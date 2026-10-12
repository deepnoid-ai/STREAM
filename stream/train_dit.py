"""STREAM Stage 1: DiT training with stochastic Riemannian flow matching on (S^{d-1})^N.

Per step:
  1. z1 = batch of UNI token maps (each token on S^{d-1}),  z0 ~ Uniform((S^{d-1})^N)
  2. t ~ logit-normal(mu, s)
  3. z_t = Exp_{SLERP(z0, z1, t)}(sigma_max * sin(pi t) * eps)       (stochastic bridge)
  4. loss = ||DiT(z_t, t) - z1||^2 / (1 - t)^2                       (chord loss, x-prediction)
  5. AdamW step with gradient-norm clipping, EMA update

Hugging Face accelerate (DDP). The run resumes automatically from the latest checkpoint
in <output-dir>/checkpoints.

Usage (8 GPUs, global batch 1024 = 128 per GPU):
    accelerate launch --num_processes 8 --num_machines 1 --mixed_precision no \
        -m stream.train_dit --config configs/brca.yaml \
        --uni-embedding-path path/to/uni_features --output-dir path/to/dit_run
"""

import argparse
import logging
import os
import re
import time
from copy import deepcopy
from glob import glob

import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import UNILatentDataset
from .dit import StreamDiT
from .geometry import sample_uniform_sphere
from .transport import bridge_perturb, chord_loss, sample_logit_normal
from .utils import atomic_torch_save, require_path, requires_grad, update_ema

LOG_EVERY = 100


def latest_checkpoint(checkpoint_dir):
    """Checkpoint with the highest step: `<step>.pt` or `final_<step>.pt` (the final one on a tie)."""
    candidates = []
    for path in glob(os.path.join(checkpoint_dir, "*.pt")):
        match = re.fullmatch(r"(final_)?(\d+)\.pt", os.path.basename(path))
        if match:
            candidates.append((int(match.group(2)), match.group(1) is not None, path))
    return max(candidates)[2] if candidates else None


def all_ranks_finite(x, device):
    """True only if `x` is finite on every rank (all ranks then stop on a bad step together)."""
    bad = torch.tensor(int(x is None or not bool(torch.isfinite(x).all().item())),
                       device=device, dtype=torch.uint8)
    if dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return not bad.item()


def main():
    torch.set_float32_matmul_precision("high")  # TF32 matmuls

    parser = argparse.ArgumentParser(description="STREAM Stage 1: DiT training.")
    parser.add_argument("--config", required=True,
                        help="STREAM YAML config (sections model, transport, dit_train).")
    parser.add_argument("--uni-embedding-path", required=True,
                        help="Directory of UNI feature shards from stream.extract_uni_features.")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory: log.txt and checkpoints/ (resumed if it has checkpoints).")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="Override dit_train.global_batch_size (e.g. a small value for a smoke test).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override dit_train.max_steps.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Micro-batches per optimizer step; per-GPU micro-batch = global batch / "
                             "num_processes / accumulation steps (default: %(default)s).")
    args = parser.parse_args()

    with open(require_path(args.config, "--config")) as f:
        config = yaml.safe_load(f)
    require_path(args.uni_embedding_path, "--uni-embedding-path")
    cfg = config["dit_train"]
    tcfg = config["transport"]
    if args.global_batch_size is not None:
        cfg["global_batch_size"] = args.global_batch_size
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(cfg["seed"], device_specific=True)  # after Accelerator(): per-rank offset
    device = accelerator.device
    rank = accelerator.local_process_index

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(os.path.join(args.output_dir, "log.txt"))])
        wandb.init(project=os.environ.get("WANDB_PROJECT", "stream"),
                   name=os.path.basename(os.path.normpath(args.output_dir)),
                   config=config, dir=os.environ.get("WANDB_DIR", args.output_dir), resume="allow")
    logger = logging.getLogger(__name__)
    if not accelerator.is_main_process:
        logger.addHandler(logging.NullHandler())

    model = StreamDiT(**config["model"])
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()
    logger.info(f"StreamDiT parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])

    dataset = UNILatentDataset(args.uni_embedding_path)
    micro_batch_size = (cfg["global_batch_size"] // accelerator.num_processes
                        // args.gradient_accumulation_steps)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(cfg["seed"] + rank)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                        generator=dl_generator)
    logger.info(f"Dataset: {len(dataset):,} feature maps; micro-batch {micro_batch_size}/GPU, "
                f"global batch {cfg['global_batch_size']}")

    # Linear warmup, then constant LR. Stepped manually once per optimizer step (not passed to
    # accelerator.prepare, whose wrapper would step it num_processes times).
    warmup_steps = cfg["warmup_steps"]
    scheduler = LambdaLR(opt, lambda step: step / warmup_steps if step < warmup_steps else 1.0)

    model, opt, loader = accelerator.prepare(model, opt, loader)
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # EMA starts as a copy

    train_steps, epoch = 0, 0
    resume_path = latest_checkpoint(checkpoint_dir)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        train_steps, epoch = ckpt["step"], ckpt["epoch"]
        for _ in range(train_steps):
            scheduler.step()
        logger.info(f"Resumed from {resume_path} at step {train_steps} "
                    f"(lr={opt.param_groups[0]['lr']:.6f})")

    def save_checkpoint(name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            path = os.path.join(checkpoint_dir, name)
            atomic_torch_save({"model": accelerator.unwrap_model(model).state_dict(),
                               "ema": ema.state_dict(), "opt": opt.state_dict(),
                               "step": train_steps, "epoch": epoch, "config": config}, path)
            logger.info(f"Saved checkpoint {path}")

    max_steps = cfg["max_steps"]
    model.train()
    running_loss, running_grad_norm, n_loss, n_grad = 0.0, 0.0, 0, 0
    start_time = time.time()
    first_step = train_steps + 1
    logger.info(f"Training from step {train_steps} to {max_steps}")

    while train_steps < max_steps:
        # As in the trainer used for the paper (UNILatentDataset has no set_epoch; the
        # prepared loader's sampler only if accelerate exposes one).
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            if train_steps >= max_steps:
                break
            z1 = batch.to(device)
            B, N, d = z1.shape
            z0 = sample_uniform_sphere(B, N, d, device)
            t = sample_logit_normal(B, mu=tcfg["logit_normal_mu"], s=tcfg["logit_normal_s"],
                                    eps_t=tcfg["eps_t"], device=device)
            z_t = bridge_perturb(z0, z1, t, sigma_max=tcfg["sigma_max"])

            grad_norm = None
            with accelerator.accumulate(model):
                loss = chord_loss(model(z_t, t), z1, t, min_one_minus_t=tcfg["min_one_minus_t"])
                # Non-finite loss or gradient norm: the check is all-reduced, so every rank
                # raises together and training stops.
                if not all_ranks_finite(loss, device):
                    raise RuntimeError(f"Non-finite loss at step {train_steps + 1} "
                                       f"(on at least one rank); training stopped.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    if not all_ranks_finite(grad_norm, device):
                        raise RuntimeError(f"Non-finite gradient norm at step {train_steps + 1} "
                                           f"(on at least one rank); training stopped.")
                opt.step()
                opt.zero_grad()

            loss_val = loss.detach()
            if dist.is_initialized():
                dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
            running_loss += loss_val.item()
            n_loss += 1
            if not accelerator.sync_gradients:
                continue

            # Optimizer step completed.
            scheduler.step()
            update_ema(ema, accelerator.unwrap_model(model), decay=cfg["ema_decay"])
            running_grad_norm += grad_norm.item()
            n_grad += 1
            train_steps += 1

            if (train_steps == first_step or train_steps % LOG_EVERY == 0
                    or train_steps == max_steps):
                torch.cuda.synchronize()
                stats = {"train/loss": running_loss / n_loss,
                         "train/grad_norm": running_grad_norm / n_grad,
                         "train/lr": opt.param_groups[0]["lr"],
                         "train/steps_per_sec": n_loss / (time.time() - start_time)}
                if accelerator.is_main_process:
                    logger.info(f"step={train_steps:07d} " + " ".join(
                        f"{k.split('/')[1]}={v:.4g}" for k, v in stats.items()))
                    wandb.log(stats, step=train_steps)
                running_loss, running_grad_norm, n_loss, n_grad = 0.0, 0.0, 0, 0
                start_time = time.time()

            if train_steps % cfg["ckpt_every"] == 0:
                save_checkpoint(f"{train_steps:07d}.pt")
        epoch += 1
        logger.info(f"Epoch {epoch} complete at step {train_steps}")

    save_checkpoint(f"final_{train_steps:07d}.pt")
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
