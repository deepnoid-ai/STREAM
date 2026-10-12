"""STREAM Stage 2: anisotropic decoder training.

Trains the pixel decoder (RAE ViT-XL GeneralDecoder + Tanh) that maps frozen UNI tokens to
256 x 256 patches. The decoder input is perturbed with anisotropic noise built from the DiT
Jacobian spectral basis (stream.extract_svd, stream.merge_svd):

    z_noisy = normalize(z + n),  n = sigma_H U U^T eta + sigma_L (I - U U^T) eta  (tangent at z)

where U holds the top-k* directions of the nearest centroid. The cosine round-trip loss uses
the CLEAN features as target.

Losses: L1 + LPIPS (VGG16) + cosine round-trip through frozen UNI + adversarial (Kaiko ViT-S/8
projected discriminator, DiffAug, hinge loss, adaptive weight).
Phases: A (reconstruction only), B (discriminator warm-up), C (adversarial term on).
bf16 autocast; gradient checkpointing on the decoder, LPIPS, UNI and discriminator; no
gradient clipping. Requires a bf16-capable GPU.

Usage (8 GPUs, global batch 512 = 64 per GPU):
    torchrun --nproc_per_node=8 -m stream.train_decoder --config configs/brca.yaml \
        --uni-embedding-path path/to/uni_features --image-dir path/to/patches \
        --svd-cache path/to/svd/per_token_svd_cache.pt --output-dir path/to/decoder_run
    # resume (continues the data order exactly): add
        --resume path/to/decoder_run/checkpoints/step-045000.pt
"""

import argparse
import logging
import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .anisotropic_noise import AnisotropicNoise
from .data import FeatureImageDataset
from .decoder import StreamDecoder
from .diffaug import DiffAug
from .discriminator import KaikoDiscriminator
from .geometry import normalize_tokens
from .lpips import LPIPS
from .uni import IMAGENET_MEAN, IMAGENET_STD, load_uni, patch_tokens
from .utils import require_path, requires_grad, update_ema

LOG_EVERY = 100
MAX_ADAPTIVE_WEIGHT = 1e4


def cosine_scheduler(optimizer, warmup_steps, total_steps, final_ratio, last_epoch=-1):
    """Linear warmup, then cosine decay to final_ratio * lr. last_epoch=N-1 resumes at step N."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def adaptive_weight(recon_loss, gan_loss, last_layer):
    """VQGAN adaptive weight ||grad L_rec|| / (||grad L_adv|| + 1e-6) on the last layer, clamped to [0.05, 1e4]."""
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    return torch.clamp(d_weight, 0.05, MAX_ADAPTIVE_WEIGHT).detach()


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(F.relu(1.0 - logits_real)) + torch.mean(F.relu(1.0 + logits_fake)))


def uni_roundtrip_features(uni_model, decoded_pm1):
    """Decoded image in [-1, 1] -> ImageNet normalization -> frozen UNI (checkpointed) -> L2-normalized tokens.

    Differentiable with respect to the image (UNI weights are frozen, no torch.no_grad).
    """
    pixels_01 = (decoded_pm1 + 1.0) * 0.5
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(pixels_01.device, dtype=pixels_01.dtype)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(pixels_01.device, dtype=pixels_01.dtype)
    all_tokens = grad_checkpoint(uni_model.forward_features, (pixels_01 - mean) / std,
                                 use_reentrant=False)
    return patch_tokens(all_tokens)


def disc_forward_fake(disc_module, fake_img):
    """Discriminator logits of fake images only (generator step; wrapped in a checkpoint)."""
    return disc_module(fake_img, None)[0]


@contextmanager
def rank0_first(rank):
    """Rank 0 runs the block first (downloading pretrained weights into the caches), then the
    other ranks; concurrent first-time torch.hub downloads into one cache collide."""
    if rank != 0:
        dist.barrier()
    yield
    if rank == 0:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="STREAM Stage 2: anisotropic decoder training.")
    parser.add_argument("--config", required=True,
                        help="STREAM YAML config (sections model, decoder, decoder_train).")
    parser.add_argument("--uni-embedding-path", required=True,
                        help="Directory of UNI feature shards with matched_index.json "
                             "(stream.build_matched_index).")
    parser.add_argument("--image-dir", required=True,
                        help="Root directory of the image patches paired in matched_index.json.")
    parser.add_argument("--svd-cache", required=True,
                        help="Merged per-token SVD cache from stream.merge_svd.")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory: train.log and checkpoints/.")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint (.pt) of this run to resume from.")
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="Override decoder_train.global_batch_size (e.g. a small value for a smoke test).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override decoder_train.max_steps (phase boundaries and LR schedules scale with it).")
    args = parser.parse_args()

    with open(require_path(args.config, "--config")) as f:
        config = yaml.safe_load(f)
    for path, flag in [(args.uni_embedding_path, "--uni-embedding-path"),
                       (args.image_dir, "--image-dir"), (args.svd_cache, "--svd-cache")]:
        require_path(path, flag)
    if args.resume is not None:
        require_path(args.resume, "--resume")
    cfg = config["decoder_train"]
    if args.global_batch_size is not None:
        cfg["global_batch_size"] = args.global_batch_size
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    # Seeds before any model or data creation.
    seed = cfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logger = logging.getLogger(__name__)
    if rank == 0:
        fh = logging.FileHandler(os.path.join(args.output_dir, "train.log"), mode="a")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    # --- Models (construction order fixes the RNG stream) ---
    decoder = StreamDecoder(config).to(device)
    decoder.train()
    ema_decoder = deepcopy(decoder).to(device)
    ema_decoder.eval()
    requires_grad(ema_decoder, False)
    update_ema(ema_decoder, decoder, decay=0)
    ddp_decoder = DDP(decoder, device_ids=[device.index], broadcast_buffers=False)

    with rank0_first(rank):
        uni_encoder = load_uni().to(device).eval()  # frozen; gradients flow to its input only
        uni_encoder.requires_grad_(False)

        disc = KaikoDiscriminator(device=device).to(device)
    disc_params = [p for p in disc.heads.parameters() if p.requires_grad]
    ddp_disc = DDP(disc, device_ids=[device.index], broadcast_buffers=False,
                   find_unused_parameters=True)
    diff_aug = DiffAug()
    with rank0_first(rank):
        lpips_fn = LPIPS().to(device).eval()
    logger.info(f"Decoder {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M params, "
                f"discriminator heads {sum(p.numel() for p in disc_params) / 1e6:.2f}M params")

    noise = AnisotropicNoise(args.svd_cache, cfg["energy_threshold"], device)
    logger.info(f"SVD cache: K={noise.n_clusters}, m={noise.m}; energy threshold "
                f"{cfg['energy_threshold']:.0%} -> k*={noise.k}")
    if noise.m < 2 * noise.k and rank == 0:
        logger.warning(f"m={noise.m} < 2k*={2 * noise.k}: the spectral basis may be limited by the "
                       f"number of probes; consider re-extracting with a larger --svd-m.")

    # --- Data ---
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    dataset = FeatureImageDataset(args.uni_embedding_path, args.image_dir, augment=True)
    batch_size = cfg["global_batch_size"] // world_size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                        persistent_workers=True, worker_init_fn=worker_init_fn)
    logger.info(f"Dataset {len(dataset):,}; batch {cfg['global_batch_size']} "
                f"({batch_size}/GPU x {world_size}); {len(loader)} steps/epoch")

    # --- Optimizers, phases ---
    opt_dec = torch.optim.AdamW(decoder.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                                weight_decay=cfg["weight_decay"])
    opt_disc = torch.optim.AdamW(disc_params, lr=cfg["disc_lr"], betas=(0.5, 0.9), weight_decay=0.0)
    total_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    disc_upd_start_step = int(cfg["disc_upd_start_frac"] * total_steps)  # Phase B starts
    gan_start_step = int(cfg["gan_start_frac"] * total_steps)            # Phase C starts
    # The discriminator schedule spans the steps it actually trains (Phases B + C).
    total_disc_steps = total_steps - disc_upd_start_step
    disc_warmup_steps = int(warmup_steps * total_disc_steps / total_steps)

    if rank == 0:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "stream"),
                   name=os.path.basename(os.path.normpath(args.output_dir)),
                   config={**cfg, "dataset_size": len(dataset)},
                   dir=os.environ.get("WANDB_DIR", args.output_dir))
    amp_dtype = torch.bfloat16

    # --- Resume ---
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ddp_decoder.module.load_state_dict(ckpt["model"])
        ema_decoder.load_state_dict(ckpt["ema"])
        opt_dec.load_state_dict(ckpt["opt_dec"])
        disc.heads.load_state_dict(ckpt["disc_heads"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        del ckpt
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    # Schedulers are (re)built for the current max_steps and fast-forwarded to global_step.
    sched_dec = cosine_scheduler(opt_dec, warmup_steps, total_steps,
                                 final_ratio=cfg["final_lr"] / cfg["lr"], last_epoch=global_step - 1)
    disc_sched_step = max(global_step - disc_upd_start_step, 0)
    sched_disc = cosine_scheduler(opt_disc, disc_warmup_steps, total_disc_steps,
                                  final_ratio=cfg["disc_final_lr"] / cfg["disc_lr"],
                                  last_epoch=disc_sched_step - 1 if disc_sched_step > 0 else -1)

    # Data resume: the loader restarts at the beginning of start_epoch, so the batches of that
    # epoch that were already trained on are drawn again and skipped (exact data order).
    resume_skip = max(global_step - start_epoch * len(loader), 0)
    if resume_skip:
        logger.info(f"Data resume: skipping {resume_skip}/{len(loader)} batches of epoch {start_epoch}")

    def save_checkpoint(name, epoch):
        torch.save({"model": ddp_decoder.module.state_dict(), "ema": ema_decoder.state_dict(),
                    "opt_dec": opt_dec.state_dict(), "disc_heads": disc.heads.state_dict(),
                    "opt_disc": opt_disc.state_dict(), "epoch": epoch, "step": global_step,
                    "config": config}, os.path.join(checkpoint_dir, name))
        logger.info(f"Saved {name}")

    logger.info(f"Steps {total_steps}: Phase A [0, {disc_upd_start_step}), "
                f"B [{disc_upd_start_step}, {gan_start_step}), C [{gan_start_step}, {total_steps})")
    dist.barrier()

    first_step = global_step + 1
    epoch = start_epoch
    while global_step < total_steps:
        sampler.set_epoch(epoch)
        ddp_decoder.train()
        for features, images in loader:
            if global_step >= total_steps:
                break
            if resume_skip > 0:
                resume_skip -= 1
                continue
            features = features.to(device, non_blocking=True)  # (B, 256, 1024)
            images = images.to(device, non_blocking=True)      # (B, 3, 256, 256) in [-1, 1]
            use_disc = global_step >= disc_upd_start_step       # Phases B and C
            use_adv = global_step >= gan_start_step             # Phase C

            features_noisy = normalize_tokens(
                features + noise.sample(features, sigma_H=cfg["sigma_H"], sigma_L=cfg["sigma_L"]))

            # --- Decoder (generator) step ---
            opt_dec.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                recon = grad_checkpoint(ddp_decoder, features_noisy, use_reentrant=False)
                loss_l1 = (recon - images).abs().mean()
                loss_lpips = grad_checkpoint(lpips_fn, recon, images, use_reentrant=False)
            # UNI at full precision; the target is the CLEAN features.
            z_roundtrip = uni_roundtrip_features(uni_encoder, recon.float())
            loss_cos = 1.0 - (z_roundtrip * features.detach()).sum(dim=-1).mean()
            loss_rec = (cfg["lambda_l1"] * loss_l1 + cfg["lambda_lpips"] * loss_lpips
                        + cfg["lambda_cos"] * loss_cos)
            if use_adv:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    # .module: discriminator weights are not updated in this step (no all-reduce).
                    logits_fake = grad_checkpoint(disc_forward_fake, ddp_disc.module,
                                                  diff_aug.aug(recon), use_reentrant=False)
                    loss_adv = -torch.mean(logits_fake)
                adapt_w = adaptive_weight(loss_rec, loss_adv, ddp_decoder.module.last_layer)
                total_loss = loss_rec + cfg["omega_G"] * adapt_w * loss_adv
            else:
                total_loss = loss_rec
            total_loss.backward()
            opt_dec.step()
            sched_dec.step()
            update_ema(ema_decoder, ddp_decoder.module, decay=cfg["ema_decay"])

            # --- Discriminator step (Phases B and C) ---
            if use_disc:
                opt_disc.zero_grad(set_to_none=True)
                # Reuse the reconstruction, quantized to uint8 levels (RAE recipe).
                recon_q = recon.detach().clamp(-1.0, 1.0)
                recon_q = torch.round((recon_q + 1.0) * 127.5) / 127.5 - 1.0
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    recon_aug = diff_aug.aug(recon_q)
                    real_aug = diff_aug.aug(images)
                    logits_fake_d, logits_real_d = ddp_disc(recon_aug, real_aug)
                    d_loss = hinge_d_loss(logits_real_d, logits_fake_d)
                d_loss.backward()
                opt_disc.step()
                sched_disc.step()

            global_step += 1
            if rank == 0 and (global_step == first_step or global_step % LOG_EVERY == 0
                              or global_step == total_steps):
                stats = {"train/total_loss": total_loss.item(), "train/L1": loss_l1.item(),
                         "train/LPIPS": loss_lpips.item(), "train/cos": loss_cos.item(),
                         "train/lr": opt_dec.param_groups[0]["lr"]}
                if use_adv:
                    stats.update({"train/adv": loss_adv.item(), "train/lambda_adapt": adapt_w.item()})
                if use_disc:
                    stats["train/D_loss"] = d_loss.item()
                phase = "C" if use_adv else ("B" if use_disc else "A")
                logger.info(f"[epoch {epoch} | step {global_step}/{total_steps} | phase {phase}] " +
                            " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in stats.items()))
                wandb.log(stats, step=global_step)
            if global_step % cfg["ckpt_every"] == 0 and rank == 0:
                save_checkpoint(f"step-{global_step:06d}.pt", epoch)
        epoch += 1
        if rank == 0:
            save_checkpoint(f"epoch-{epoch:03d}.pt", epoch)

    dist.barrier()
    if rank == 0:
        save_checkpoint("decoder_final.pt", epoch)
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
