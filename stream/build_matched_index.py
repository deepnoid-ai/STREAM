"""Pair every UNI feature with its image file: writes <uni-embedding-path>/matched_index.json.

Each entry is [shard file name, row in the shard, image path relative to --image-dir].
Features whose image is not found under --image-dir are skipped (and counted).

Usage:
    python -m stream.build_matched_index \
        --uni-embedding-path path/to/uni_features --image-dir path/to/patches
"""

import argparse
import json
import os

from safetensors import safe_open

from .data import IMAGE_EXTENSIONS, list_shards
from .utils import require_path


def main():
    parser = argparse.ArgumentParser(
        description="Build matched_index.json pairing UNI feature shards with image files.")
    parser.add_argument("--uni-embedding-path", required=True,
                        help="Directory of UNI feature shards from stream.extract_uni_features; "
                             "matched_index.json is written here.")
    parser.add_argument("--image-dir", required=True,
                        help="The --image-dir that was given to stream.extract_uni_features.")
    args = parser.parse_args()
    feature_dir = require_path(args.uni_embedding_path, "--uni-embedding-path")
    image_dir = require_path(args.image_dir, "--image-dir")

    available = set()
    for dirpath, _, filenames in os.walk(image_dir):
        rel_dir = os.path.relpath(dirpath, image_dir)
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS:
                available.add(fn if rel_dir == "." else os.path.join(rel_dir, fn))
    print(f"{len(available):,} images under {image_dir}")

    matched, skipped = [], 0
    shards = list_shards(feature_dir)
    for i, shard_path in enumerate(shards):
        with safe_open(str(shard_path), framework="pt") as f:
            metadata = f.metadata() or {}
            for local_idx in range(f.get_slice("features").get_shape()[0]):
                filename = metadata.get(f"filename_{local_idx}")
                if filename in available:
                    matched.append([shard_path.name, local_idx, filename])
                else:
                    skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(shards)} shards] {len(matched):,} matched, {skipped:,} skipped")

    output_path = os.path.join(feature_dir, "matched_index.json")
    with open(output_path, "w") as f:
        json.dump(matched, f)
    print(f"Matched {len(matched):,} features ({skipped:,} without an image) -> {output_path}")


if __name__ == "__main__":
    main()
