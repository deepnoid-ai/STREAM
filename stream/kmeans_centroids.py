"""K-means centroids of UNI token maps (STREAM spectral basis extraction, step 1).

Loads --n-features feature maps chosen by torch.randperm (generator seed 42), clusters their
L2-normalized mean-pooled tokens with MiniBatchKMeans(random_state=42, batch_size=1024), and
saves each cluster's per-token mean map, re-normalized onto (S^{d-1})^N, as a (K, N, d) tensor.

Usage:
    python -m stream.kmeans_centroids --uni-embedding-path path/to/uni_features \
        --output-path path/to/svd/centroids.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .data import UNILatentDataset
from .utils import require_path


def load_random_features(feature_dir: str, n_features: int, num_workers: int, seed: int = 42):
    """`n_features` feature maps (n, N, d) chosen by torch.randperm with a fixed seed."""
    dataset = UNILatentDataset(feature_dir)
    rng = torch.Generator().manual_seed(seed)
    n_load = min(n_features, len(dataset))
    indices = torch.randperm(len(dataset), generator=rng)[:n_load].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=512, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    features = torch.empty(n_load, *dataset[0].shape, dtype=torch.float32)
    offset = 0
    for batch in loader:
        features[offset:offset + batch.shape[0]] = batch
        offset += batch.shape[0]
    return features


def kmeans_centroids(features: torch.Tensor, n_centroids: int, seed: int = 42) -> torch.Tensor:
    """MiniBatchKMeans on L2-normalized mean-pooled tokens -> per-token-normalized cluster means (K, N, d)."""
    from sklearn.cluster import MiniBatchKMeans

    n, N, d = features.shape
    features_pooled = F.normalize(features.mean(dim=1), dim=-1)
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=seed, batch_size=1024)
    labels = kmeans.fit_predict(features_pooled.cpu().numpy())
    centroids = torch.zeros(n_centroids, N, d)
    for c in range(n_centroids):
        mask = torch.tensor(labels == c)
        if mask.sum() > 0:
            centroids[c] = features[mask].float().mean(dim=0)
    return F.normalize(centroids, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="K-means centroids of UNI token maps.")
    parser.add_argument("--uni-embedding-path", required=True,
                        help="Directory of UNI feature shards (the DiT training features).")
    parser.add_argument("--output-path", required=True,
                        help="Output .pt file for the (K, N, d) centroid tensor.")
    parser.add_argument("--n-centroids", type=int, default=50,
                        help="Number of clusters K (default: %(default)s).")
    parser.add_argument("--n-features", type=int, default=50000,
                        help="Number of random feature maps to cluster (default: %(default)s).")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers for loading features (default: %(default)s).")
    args = parser.parse_args()
    require_path(args.uni_embedding_path, "--uni-embedding-path")

    features = load_random_features(args.uni_embedding_path, args.n_features, args.num_workers)
    print(f"Loaded {features.shape[0]:,} feature maps from {args.uni_embedding_path}")
    centroids = kmeans_centroids(features, args.n_centroids)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(centroids, args.output_path)
    print(f"Saved centroids {tuple(centroids.shape)} to {args.output_path}")


if __name__ == "__main__":
    main()
