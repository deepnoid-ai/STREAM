"""Datasets over the UNI feature shards written by stream.extract_uni_features.

Shard format (safetensors): 'features' (n, 256, 1024) float16, metadata
'filename_<i>' = image path relative to the image directory.
"""

import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_shards(feature_dir: str):
    """Sorted shard files of a feature directory (error if there are none)."""
    shards = sorted(Path(feature_dir).glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.safetensors files in {feature_dir}; run stream.extract_uni_features first.")
    return shards


class UNILatentDataset(Dataset):
    """UNI token maps (256, 1024) as float32, for DiT training and k-means."""

    def __init__(self, feature_dir: str):
        self.index = []
        for shard_path in list_shards(feature_dir):
            with safe_open(str(shard_path), framework="pt") as f:
                shard_size = f.get_slice("features").get_shape()[0]  # header only
            self.index.extend((str(shard_path), i) for i in range(shard_size))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_path, local_idx = self.index[idx]
        with safe_open(shard_path, framework="pt") as f:
            return f.get_slice("features")[local_idx].to(torch.float32)


def apply_geometric_augmentation(image: torch.Tensor, tokens: torch.Tensor):
    """Random H/V flip and 90-degree rotation, applied identically to the image and the 16 x 16 token grid."""
    tokens_2d = tokens.view(16, 16, -1)
    if random.random() < 0.5:
        image = image.flip(-1)
        tokens_2d = tokens_2d.flip(1)
    if random.random() < 0.5:
        image = image.flip(-2)
        tokens_2d = tokens_2d.flip(0)
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        tokens_2d = torch.rot90(tokens_2d, k, dims=(0, 1))
    return image, tokens_2d.reshape(256, -1)


class FeatureImageDataset(Dataset):
    """(UNI tokens, image in [-1, 1]) pairs listed in <feature_dir>/matched_index.json.

    Used for decoder training (augment=True) and for the rFID reconstructions (augment=False).
    """

    def __init__(self, feature_dir: str, image_dir: str, augment: bool):
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        index_path = os.path.join(feature_dir, "matched_index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} not found; run `python -m stream.build_matched_index "
                f"--uni-embedding-path {feature_dir} --image-dir {image_dir}` first.")
        with open(index_path) as f:
            self.index = [(os.path.join(feature_dir, shard), local_idx, filename)
                          for shard, local_idx, filename in json.load(f)]
        print(f"[FeatureImageDataset] {len(self.index):,} samples from {index_path}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int, _retry: int = 0):
        shard_path, local_idx, filename = self.index[idx]
        try:
            with safe_open(shard_path, framework="pt") as f:
                feature = f.get_slice("features")[local_idx].to(torch.float32)
            image = self.image_transform(Image.open(self.image_dir / filename).convert("RGB"))
            if self.augment:
                image, feature = apply_geometric_augmentation(image, feature)
            return feature, image
        except Exception:
            # Unreadable image or shard entry: fall back to a random sample (at most 10 tries).
            if _retry >= 10:
                raise RuntimeError(f"Failed to load {filename} from {shard_path} after 10 retries.")
            return self.__getitem__(random.randint(0, len(self.index) - 1), _retry=_retry + 1)
