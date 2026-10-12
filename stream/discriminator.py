"""Kaiko ViT-S/8 projected discriminator for decoder training.

Follows RAE's DinoDisc (Zheng et al., 2025; https://github.com/bytetriper/RAE, MIT License)
with a pathology backbone:
  - frozen Kaiko ViT-S/8 (torch.hub `kaiko-ai/towards_large_pathology_fms`, `vits8`, pinned to
    a fixed commit; its hubconf loads the weights of release 0.0.1),
  - 5 trainable heads on the tokens after blocks {2, 5, 8, 11} and after the last block,
    each SpectralConv1d(k=1) + ResidualBlock(SpectralConv1d(k=9)) + SpectralConv1d(-> 1),
    with BatchNormLocal and LeakyReLU.
Input: images in [-1, 1] (resized to 224 x 224). Output: per-token logits of all heads (B, 5 * 784).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

HEAD_KERNEL_SIZE = 9
# kaiko-ai/towards_large_pathology_fms pinned to a commit: the release tag 0.0.1 holds the weight
# files but no hubconf.py, so torch.hub cannot load the tag itself. The hubconf of this commit
# downloads `releases/download/0.0.1/vits8.pth`. skip_validation: torch.hub can only validate a
# ref that is the head of a branch or tag, and this pinned commit need not stay one.
KAIKO_HUB_REPO = "kaiko-ai/towards_large_pathology_fms:bfcde171e2003e862f9f821226788da9ee571a13"


class SpectralConv1d(nn.Conv1d):
    """Conv1d with spectral normalization applied at construction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    """Batch norm over virtual mini-batches of size 1 (per-sample normalization over tokens), as in RAE."""

    def __init__(self, num_features: int, virtual_bs: int = 1, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()
        G = int(np.ceil(x.size(0) / self.virtual_bs))
        x = x.view(G, -1, x.size(-2), x.size(-1))
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


class ResidualBlock(nn.Module):
    """(fn(x) + x) / sqrt(2)."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.ratio = 1.0 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(channels, channels, kernel_size=kernel_size,
                       padding=kernel_size // 2, padding_mode="circular"),
        BatchNormLocal(channels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class FrozenKaiko(nn.Module):
    """Frozen Kaiko ViT-S/8 returning the patch tokens [last block, block 2, 5, 8, 11] as (B, 384, L).

    The backbone is kept in a tuple so it is not registered as a submodule (never trained,
    never in a state_dict), as in RAE.
    """

    KEY_DEPTHS = (2, 5, 8, 11)
    EMBED_DIM = 384

    def __init__(self, device: torch.device):
        super().__init__()
        model = torch.hub.load(KAIKO_HUB_REPO, "vits8", trust_repo=True, skip_validation=True)
        model.eval()
        model.requires_grad_(False)
        self.model_proxy: Tuple[nn.Module, ...] = (model.to(device),)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone in float32: bf16 attention backward overflows on strong head gradients.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            model = self.model_proxy[0]
            if x.shape[-2] != 224 or x.shape[-1] != 224:
                x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = model.patch_embed(x)
            x = model._pos_embed(x)
            x = model.patch_drop(x)
            x = model.norm_pre(x)
            activations = []
            for idx, block in enumerate(model.blocks):
                x = block(x)
                if idx in self.KEY_DEPTHS:
                    activations.append(x[:, 1:, :].transpose(1, 2))
            activations.insert(0, x[:, 1:, :].transpose(1, 2))
        return activations


class KaikoDiscriminator(nn.Module):
    """forward(fake, real=None) -> (logits_fake, logits_real or None), each (B, 5 * L)."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.kaiko = FrozenKaiko(device)
        C = FrozenKaiko.EMBED_DIM
        self.heads = nn.ModuleList([
            nn.Sequential(
                make_block(C, kernel_size=1),
                ResidualBlock(make_block(C, kernel_size=HEAD_KERNEL_SIZE)),
                SpectralConv1d(C, 1, kernel_size=1, padding=0),
            )
            for _ in range(len(FrozenKaiko.KEY_DEPTHS) + 1)
        ])
        for head in self.heads:
            head.requires_grad_(True)

    def classify(self, img: torch.Tensor) -> torch.Tensor:
        activations = self.kaiko(img)
        return torch.cat([head(act).view(img.shape[0], -1)
                          for head, act in zip(self.heads, activations)], dim=1)

    def forward(self, fake: torch.Tensor, real: Optional[torch.Tensor] = None):
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real
