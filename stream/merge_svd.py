"""Merge the per_token_svd_partial_*.pt files of stream.extract_svd into one SVD cache.

Usage:
    python -m stream.merge_svd --partial-dir path/to/svd \
        --output-path path/to/svd/per_token_svd_cache.pt
"""

import argparse
import glob
import os

import torch

from .utils import atomic_torch_save, require_path


def main():
    parser = argparse.ArgumentParser(description="Merge partial per-token SVD caches.")
    parser.add_argument("--partial-dir", required=True,
                        help="Directory containing the per_token_svd_partial_*.pt files.")
    parser.add_argument("--output-path", required=True,
                        help="Output path of the merged cache (input of stream.train_decoder --svd-cache).")
    args = parser.parse_args()
    require_path(args.partial_dir, "--partial-dir")

    files = sorted(glob.glob(os.path.join(args.partial_dir, "per_token_svd_partial_*.pt")))
    if not files:
        raise FileNotFoundError(f"No per_token_svd_partial_*.pt files in {args.partial_dir}.")
    partials = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    partials.sort(key=lambda p: p["centroid_start"])

    # The partials must tile [0, n_clusters) and share m and t_eval.
    first, expected_start = partials[0], 0
    for f, p in zip(files, partials):
        if p.get("type") != "per_token_partial":
            raise ValueError(f"{f} is not a partial per-token SVD cache.")
        for key in ("n_clusters", "m", "t_eval"):
            if p[key] != first[key]:
                raise ValueError(f"Inconsistent {key} across partials: {p[key]} vs {first[key]}.")
        if p["centroid_start"] != expected_start:
            raise ValueError(f"Centroids [{expected_start}, {p['centroid_start']}) are missing.")
        expected_start = p["centroid_end"]
    if expected_start != first["n_clusters"]:
        raise ValueError(f"Centroids [{expected_start}, {first['n_clusters']}) are missing.")

    merged = {
        "centroids": torch.cat([p["centroids"] for p in partials], dim=0),
        "centroid_U_T": torch.cat([p["centroid_U_T"] for p in partials], dim=0),
        "centroid_S_top": torch.cat([p["centroid_S_top"] for p in partials], dim=0),
        "n_clusters": first["n_clusters"],
        "m": first["m"],
        "t_eval": first["t_eval"],
        "type": "per_token",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    atomic_torch_save(merged, args.output_path)
    print(f"Merged {len(partials)} partials: U {tuple(merged['centroid_U_T'].shape)} -> "
          f"{args.output_path} ({os.path.getsize(args.output_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
