"""LPIPS perceptual loss (Zhang et al., 2018) with frozen VGG16 features.

Adapted from taming-transformers (https://github.com/CompVis/taming-transformers, MIT License,
Copyright (c) 2020 Patrick Esser and Robin Rombach and Bjorn Ommer) as used in RAE
(https://github.com/bytetriper/RAE, MIT License). LPIPS: https://github.com/richzhang/PerceptualSimilarity
(BSD-2-Clause). Inputs are images in [-1, 1]; returns the mean LPIPS distance.

The LPIPS linear weights (vgg.pth) are downloaded once to $TORCH_HOME/lpips (default
~/.cache/torch/lpips) and checked against their MD5.
"""

import hashlib
import os
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

LPIPS_VGG_URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
LPIPS_VGG_MD5 = "d507d7349b931f0638a25a48a722f98a"


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def lpips_vgg_weights() -> str:
    torch_home = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
    root = os.path.join(torch_home, "lpips")
    path = os.path.join(root, "vgg.pth")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        print(f"Downloading LPIPS VGG weights to {path}")
        torch.hub.download_url_to_file(LPIPS_VGG_URL, path)
    md5 = _md5(path)
    if md5 != LPIPS_VGG_MD5:
        raise RuntimeError(f"{path} has MD5 {md5}, expected {LPIPS_VGG_MD5}; delete it and retry.")
    return path


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1) -> None:
        super().__init__()
        # The Dropout is inactive (eval mode) but keeps the pretrained key names "lin*.model.1.weight".
        self.model = nn.Sequential(nn.Dropout(),
                                   nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))


class VGG16FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h_relu1_2 = self.slice1(tensor)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor ** 2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


class LPIPS(nn.Module):
    """Frozen LPIPS (VGG16): forward(input, target) -> scalar mean distance, inputs in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor()
        self.lin0 = NetLinLayer(self.chns[0])
        self.lin1 = NetLinLayer(self.chns[1])
        self.lin2 = NetLinLayer(self.chns[2])
        self.lin3 = NetLinLayer(self.chns[3])
        self.lin4 = NetLinLayer(self.chns[4])
        state = torch.load(lpips_vgg_weights(), map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feats_input = self.net(self.scaling_layer(input))
        feats_target = self.net(self.scaling_layer(target))
        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        value = torch.zeros(1, device=input.device)
        for feat_in, feat_tgt, lin in zip(feats_input, feats_target, lin_layers):
            diff = (_normalize(feat_in) - _normalize(feat_tgt)) ** 2
            value = value + lin.model(diff).mean([2, 3], keepdim=True).mean()
        return torch.mean(value)
