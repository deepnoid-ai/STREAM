"""STREAM inference and gFID / rFID evaluation with a PNG round-trip (single GPU).

Generated and reconstructed images are written to disk and re-read through the same
image-folder + Inception path as the reference images.

  1. DiT sampling (EMA weights)                  -> <output-dir>/features_gen.pt
  2. decoder (EMA weights)                       -> <output-dir>/gen_png/{i:06d}.png
     (stops here with --generate-only)
  3. Inception features of gen_png/; reference features of --gt-dir (read from
     --reference-stats, or computed and saved there)  -> gFID
  4. decoder(UNI features of the reference images)  -> <output-dir>/recon_png/<reference file name>
     (same name and extension as the reference, so both share the codec) -> rFID
  5. <output-dir>/results.json
Every phase reuses its output if it already exists.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m stream.evaluate --config configs/brca.yaml \\
      --dit-checkpoint path/to/dit_run/checkpoints/final_0095000.pt \\
      --decoder-checkpoint path/to/decoder_run/checkpoints/step-090000.pt \\
      --gt-dir path/to/fid_reference_images \\
      --gt-uni-embedding-path path/to/fid_reference_uni_features \\
      --reference-stats path/to/fid_reference_inception.npz \\
      --output-dir path/to/eval
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from .data import IMAGE_EXTENSIONS, FeatureImageDataset
from .decoder import load_decoder
from .dit import load_dit
from .fid import compute_fid, inception_features
from .sampling import generate_features
from .utils import require_path

logger = logging.getLogger("stream.evaluate")


def image_names(d):
    """Basenames of all images under d (recursive)."""
    return [p.name for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def generate_and_save(args, config, device):
    """Phases 1-2: sample features, decode them and save PNGs."""
    feat_path = os.path.join(args.output_dir, "features_gen.pt")
    png_dir = os.path.join(args.output_dir, "gen_png")
    if os.path.exists(feat_path):
        logger.info(f"Reusing {feat_path}")
    else:
        dit = load_dit(args.dit_checkpoint, config, device)
        logger.info(f"Generating {args.num_gen:,} feature maps "
                    f"({config['sampling']['num_steps']} midpoint steps, seed {args.seed})")
        features = generate_features(dit, args.num_gen, args.batch_size,
                                     config["sampling"]["num_steps"], args.seed, device)
        torch.save(features, feat_path)
        del dit
        torch.cuda.empty_cache()

    n_existing = len(list(Path(png_dir).glob("*.png"))) if os.path.isdir(png_dir) else 0
    if n_existing >= args.num_gen:
        logger.info(f"Reusing {png_dir} ({n_existing} PNGs)")
        return png_dir
    decoder = load_decoder(args.decoder_checkpoint, config, device)
    features = torch.load(feat_path, map_location="cpu", weights_only=True)[:args.num_gen]
    n_total = features.shape[0]
    os.makedirs(png_dir, exist_ok=True)
    for start in range(0, n_total, args.batch_size):
        imgs = decoder(features[start:start + args.batch_size].to(device))  # tanh, [-1, 1]
        imgs_01 = ((imgs + 1.0) * 0.5).clamp(0, 1)
        for j in range(imgs_01.shape[0]):
            save_image(imgs_01[j], f"{png_dir}/{start + j:06d}.png")
    logger.info(f"Saved {n_total:,} images to {png_dir}")
    del decoder
    torch.cuda.empty_cache()
    return png_dir


def reconstruct_references(args, config, device):
    """Phase 4: decode the UNI features of the reference images, saved under the reference file names."""
    recon_dir = os.path.join(args.output_dir, "recon_png")
    dataset = FeatureImageDataset(args.gt_uni_embedding_path, args.gt_dir, augment=False)
    order = sorted(range(len(dataset.index)), key=lambda i: os.path.basename(dataset.index[i][2]))
    names = [os.path.basename(dataset.index[i][2]) for i in order]

    # Duplicate index rows are allowed: multi-GPU extraction pads the last batch by repeating
    # images, and a duplicate rewrites the same file. The unique names must equal the reference set.
    unique, gt_names = set(names), set(image_names(args.gt_dir))
    if unique != gt_names:
        raise ValueError(
            f"The matched index of --gt-uni-embedding-path does not cover the images in --gt-dir: "
            f"{len(unique - gt_names)} index entries without an image, {len(gt_names - unique)} "
            f"images without features. Extract features of exactly the --gt-dir images.")

    if os.path.isdir(recon_dir) and len(set(image_names(recon_dir))) >= len(unique):
        logger.info(f"Reusing {recon_dir}")
        return recon_dir
    decoder = load_decoder(args.decoder_checkpoint, config, device)
    loader = DataLoader(Subset(dataset, order), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)
    os.makedirs(recon_dir, exist_ok=True)
    saved = 0
    for features, _ in loader:
        recon_01 = ((decoder(features.to(device)) + 1.0) * 0.5).clamp(0, 1)
        for j in range(recon_01.shape[0]):
            # Same name and extension as the reference: save_image picks the codec from it.
            save_image(recon_01[j], os.path.join(recon_dir, names[saved + j]))
        saved += recon_01.shape[0]
    n_files = len(set(image_names(recon_dir)))
    if n_files != len(unique) or saved != len(names):
        raise RuntimeError(f"Wrote {n_files} reconstructions for {len(unique)} reference images.")
    logger.info(f"Saved {n_files:,} reconstructions to {recon_dir}")
    del decoder
    torch.cuda.empty_cache()
    return recon_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="STREAM YAML config (sections model, decoder, sampling).")
    ap.add_argument("--dit-checkpoint", required=True,
                    help="DiT checkpoint from stream.train_dit (EMA weights are used).")
    ap.add_argument("--decoder-checkpoint", required=True,
                    help="Decoder checkpoint from stream.train_decoder (EMA weights are used).")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory (features_gen.pt, gen_png/, recon_png/, results.json).")
    ap.add_argument("--gt-dir",
                    help="Folder of reference images for FID (required unless --generate-only).")
    ap.add_argument("--gt-uni-embedding-path",
                    help="UNI features of the --gt-dir images with matched_index.json, for rFID "
                         "(required unless --generate-only).")
    ap.add_argument("--reference-stats",
                    help=".npz with key 'features' = Inception features of --gt-dir. Read if it exists, "
                         "otherwise computed and saved there (default: "
                         "<output-dir>/reference_inception_features.npz).")
    ap.add_argument("--num-gen", type=int, default=50000,
                    help="Number of generated images (default: %(default)s).")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Sampling and decoding batch size (default: %(default)s). Together with "
                         "--seed it determines the generated samples.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (default: %(default)s).")
    ap.add_argument("--generate-only", action="store_true",
                    help="Stop after writing <output-dir>/gen_png (no FID).")
    args = ap.parse_args()
    if not args.generate_only and not (args.gt_dir and args.gt_uni_embedding_path):
        ap.error("--gt-dir and --gt-uni-embedding-path are required unless --generate-only is set")

    with open(require_path(args.config, "--config")) as f:
        config = yaml.safe_load(f)
    require_path(args.dit_checkpoint, "--dit-checkpoint")
    require_path(args.decoder_checkpoint, "--decoder-checkpoint")
    if not args.generate_only:
        require_path(args.gt_dir, "--gt-dir")
        require_path(args.gt_uni_embedding_path, "--gt-uni-embedding-path")
    if not torch.cuda.is_available():
        raise RuntimeError("stream.evaluate needs a CUDA GPU.")
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    gen_dir = generate_and_save(args, config, device)
    if args.generate_only:
        return

    t0 = time.time()
    gen_feats = inception_features(gen_dir, device)
    ref_path = args.reference_stats or os.path.join(args.output_dir, "reference_inception_features.npz")
    if os.path.exists(ref_path):
        gt_feats = np.load(ref_path)["features"]
    else:
        gt_feats = inception_features(args.gt_dir, device)
        os.makedirs(os.path.dirname(os.path.abspath(ref_path)), exist_ok=True)
        np.savez(ref_path, features=gt_feats)
        logger.info(f"Saved reference Inception features {gt_feats.shape} to {ref_path}")
    gfid = compute_fid(gt_feats, gen_feats)
    logger.info(f"gFID = {gfid:.4f}")

    recon_feats = inception_features(reconstruct_references(args, config, device), device)
    rfid = compute_fid(gt_feats, recon_feats)
    logger.info(f"rFID = {rfid:.4f} ({(time.time() - t0) / 60:.1f} min for FID)")

    results = {"gfid": gfid, "rfid": rfid, "num_gen": args.num_gen, "seed": args.seed,
               "batch_size": args.batch_size, "num_steps": config["sampling"]["num_steps"],
               "dit_checkpoint": args.dit_checkpoint, "decoder_checkpoint": args.decoder_checkpoint,
               "gt_dir": args.gt_dir, "reference_stats": ref_path}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()
