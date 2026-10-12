"""FID with the FID-canonical Inception V3 and a float64 matrix square root.

- Network: torchmetrics' NoTrainInceptionV3 (`weights-inception-2015-12-05-6726825d.pth`, the
  pytorch-fid / torchmetrics FID network, not torchvision's ImageNet Inception), fp32, cuDNN off.
- Images are read from disk: PIL -> Resize(299, bilinear, antialias) -> CenterCrop(299) -> uint8.
- FID = ||mu_1 - mu_2||^2 + Tr(S_1 + S_2 - 2 sqrt(S_1 S_2)) with scipy.linalg.sqrtm in float64.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data import IMAGE_EXTENSIONS


class ImageFolder(Dataset):
    """All images under `image_dir` (recursive, sorted), preprocessed for Inception."""

    def __init__(self, image_dir: str):
        self.files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.preprocess = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(299),
            transforms.PILToTensor(),  # uint8, as NoTrainInceptionV3 expects
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.preprocess(Image.open(self.files[idx]).convert("RGB"))


@torch.no_grad()
def inception_features(image_dir: str, device: torch.device, batch_size: int = 64,
                       num_workers: int = 8) -> np.ndarray:
    """2048-d Inception pool features (n, 2048) in float64 of every image under image_dir."""
    from torchmetrics.image.fid import NoTrainInceptionV3

    model = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    model.requires_grad_(False)
    dataset = ImageFolder(image_dir)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found under {image_dir}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    feats, n_done = [], 0
    for batch in loader:
        with torch.backends.cudnn.flags(enabled=False):
            feats.append(model(batch.to(device, non_blocking=True)).double().cpu().numpy())
        n_done += batch.shape[0]
        if n_done % (batch_size * 10) == 0 or n_done >= len(dataset):
            print(f"    [inception] {n_done:,}/{len(dataset):,}")
    return np.concatenate(feats, axis=0)


def compute_fid(gt_feats: np.ndarray, pred_feats: np.ndarray) -> float:
    mu1, mu2 = gt_feats.mean(0), pred_feats.mean(0)
    sigma1 = np.cov(gt_feats, rowvar=False)
    sigma2 = np.cov(pred_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        print(f"  [FID] sqrtm produced non-finite values; adding {eps} to the diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID sqrtm has a significant imaginary component: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
