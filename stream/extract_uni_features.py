"""Extract UNI patch tokens for a directory of 256 x 256 image patches (STREAM feature extraction).

Every patch is encoded by frozen UNI (ToTensor + ImageNet normalization, no resize, fp16
autocast); the CLS token is dropped and each of the 256 tokens is L2-normalized. Tokens are
stored as float16 safetensors shards `shard_<rank>_<index>.safetensors` with
'features' (n, 256, 1024) and metadata 'filename_<i>' = path relative to --image-dir.

--image-dir may be flat or contain one level of sub-folders (e.g. one per slide; see
--max-depth). The sorted file list is cached in <output-dir>/filelist.txt and reused by
later runs; an interrupted run resumes after the shards that already exist.

Usage:
    torchrun --nproc_per_node=8 -m stream.extract_uni_features \
        --image-dir path/to/patches --output-dir path/to/uni_features
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from .data import IMAGE_EXTENSIONS
from .uni import IMAGENET_MEAN, IMAGENET_STD, load_uni, patch_tokens
from .utils import require_path


class OffsetDistributedSampler(DistributedSampler):
    """DistributedSampler (no shuffle) that skips the first `start_index` samples of its rank (resume)."""

    def __init__(self, *args, start_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = start_index

    def __iter__(self):
        return iter(list(super().__iter__())[self.start_index:])

    def __len__(self):
        return max(super().__len__() - self.start_index, 0)


class ImageListDataset(Dataset):
    """(image tensor, relative path) for a list of paths relative to `image_dir`; None if unreadable."""

    def __init__(self, image_dir: str, files, transform):
        self.image_dir = Path(image_dir)
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = self.files[idx]
        try:
            return self.transform(Image.open(self.image_dir / rel_path).convert("RGB")), rel_path
        except Exception as e:
            warnings.warn(f"Skipping unreadable image {rel_path}: {e}")
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


def list_images(image_dir: str, max_depth: int):
    """Sorted image paths relative to image_dir, walking at most `max_depth` directory levels."""
    files = []
    root_depth = image_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(image_dir):
        if dirpath.rstrip(os.sep).count(os.sep) - root_depth >= max_depth:
            dirnames.clear()
            continue
        rel_dir = os.path.relpath(dirpath, image_dir)
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTENSIONS:
                files.append(fn if rel_dir == "." else os.path.join(rel_dir, fn))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No {'/'.join(IMAGE_EXTENSIONS)} images found under {image_dir}.")
    return files


def save_shard(features, filenames, path):
    save_file({"features": features}, path,
              metadata={f"filename_{i}": fn for i, fn in enumerate(filenames)})


def main():
    parser = argparse.ArgumentParser(
        description="Extract UNI patch tokens (float16 safetensors shards) for a directory of patches.")
    parser.add_argument("--image-dir", required=True,
                        help="Root directory of the 256 x 256 image patches (.png/.jpg/.jpeg).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for the shards and filelist.txt.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size per GPU (default: %(default)s).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers per GPU (default: %(default)s).")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Patches per shard, about 5 GB at float16 (default: %(default)s).")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Directory levels to walk: 1 = flat directory, 2 = root plus one level "
                             "of sub-folders (default: %(default)s).")
    args = parser.parse_args()
    require_path(args.image_dir, "--image-dir")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_uni().to(device).eval()
    model.requires_grad_(False)
    # No resize/crop: 256 x 256 input gives 16 x 16 tokens (dynamic_img_size).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Rank 0 lists the images (or reads the cached list) and broadcasts the list.
    filelist_path = os.path.join(args.output_dir, "filelist.txt")
    files = None
    if rank == 0:
        if os.path.isfile(filelist_path):
            with open(filelist_path) as f:
                files = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(files):,} paths from {filelist_path}")
        else:
            files = list_images(args.image_dir, args.max_depth)
            with open(filelist_path, "w") as f:
                f.writelines(fn + "\n" for fn in files)
            print(f"Found {len(files):,} images; list saved to {filelist_path}")
    if distributed:
        obj = [files]
        dist.broadcast_object_list(obj, src=0)
        files = obj[0]
    dataset = ImageListDataset(args.image_dir, files, transform)

    # Resume: skip the samples already written to this rank's shards.
    shard_idx = sum(1 for f in os.listdir(args.output_dir)
                    if f.startswith(f"shard_{rank:02d}_") and f.endswith(".safetensors"))
    done = shard_idx * args.shard_size
    if done:
        print(f"[rank {rank}] resuming after {shard_idx} existing shards ({done:,} samples)")
    sampler = OffsetDistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                       shuffle=False, start_index=done)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_skip_none)

    buf_features, buf_names = [], []
    n_new, t0 = 0, time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in loader:
            if batch is None:  # every image of the batch was unreadable
                continue
            images, names = batch
            features = patch_tokens(model.forward_features(images.to(device, non_blocking=True)))
            buf_features.append(features.cpu().to(torch.float16))
            buf_names.extend(names)
            n_new += len(names)
            while len(buf_names) >= args.shard_size:
                all_features = torch.cat(buf_features, dim=0)
                name = f"shard_{rank:02d}_{shard_idx:06d}.safetensors"
                save_shard(all_features[:args.shard_size], buf_names[:args.shard_size],
                           os.path.join(args.output_dir, name))
                rest = all_features[args.shard_size:]
                buf_features = [rest] if rest.shape[0] > 0 else []
                buf_names = buf_names[args.shard_size:]
                shard_idx += 1
                if rank == 0:
                    print(f"[rank 0] saved {name} ({done + n_new:,} samples, "
                          f"{n_new / (time.time() - t0):.0f} img/s)")
    if buf_names:
        name = f"shard_{rank:02d}_{shard_idx:06d}.safetensors"
        save_shard(torch.cat(buf_features, dim=0), buf_names, os.path.join(args.output_dir, name))
        print(f"[rank {rank}] saved final shard {name} ({len(buf_names)} samples)")
    if rank == 0:
        print(f"Done: {n_new:,} new images in {time.time() - t0:.0f}s on rank 0.")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
