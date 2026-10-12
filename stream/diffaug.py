# Differentiable augmentation (DiffAugment) for the discriminator of decoder training.
# Adapted from StyleGAN-T (Sauer et al., 2023) via RAE (Zheng et al., 2025):
#   https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/training/diffaug.py
#   https://github.com/bytetriper/RAE (MIT License)
# which is in turn based on https://github.com/mit-han-lab/data-efficient-gans
# under the following license:
#
# BSD 2-Clause "Simplified" License
# Copyright (c) 2020, Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modification for STREAM: only the configuration used here (prob=1.0, no cutout) is kept.
import torch
import torch.nn.functional as F


class DiffAug:
    """Differentiable translation (up to 1/8 of the image) + color jitter (brightness, saturation,
    contrast), applied to every batch: RAE's DiffAug with prob=1.0 and cutout disabled.
    """

    def __init__(self):
        self.grids = {}

    def get_grids(self, B, x, y, dev):
        if (B, x, y) in self.grids:
            return self.grids[(B, x, y)]
        self.grids[(B, x, y)] = ret = torch.meshgrid(
            torch.arange(B, dtype=torch.long, device=dev),
            torch.arange(x, dtype=torch.long, device=dev),
            torch.arange(y, dtype=torch.long, device=dev),
            indexing='ij',
        )
        return ret

    def aug(self, BCHW: torch.Tensor) -> torch.Tensor:
        if BCHW.dtype != torch.float32:
            BCHW = BCHW.float()

        # The upstream draws three Bernoulli(prob) gates (translation, color, cutout) from the
        # CPU generator. With prob=1.0 they are always on (cutout is disabled), but the draw is
        # kept so the random-number stream is unchanged.
        torch.rand(3)
        B, dev = BCHW.shape[0], BCHW.device
        rand01 = torch.rand(7, B, 1, 1, device=dev)

        raw_h, raw_w = BCHW.shape[-2:]

        # Translation (up to 12.5% shift)
        ratio = 0.125
        delta_h = round(raw_h * ratio)
        delta_w = round(raw_w * ratio)
        translation_h = rand01[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
        translation_w = rand01[1].mul(delta_w + delta_w + 1).floor().long() - delta_w
        grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
        grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
        grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
        bchw_pad = F.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
        BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()

        # Color jitter (brightness, saturation, contrast)
        BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
        bchw_mean = BCHW.mean(dim=1, keepdim=True)
        BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
        bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
        BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)

        return BCHW
