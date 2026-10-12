"""STREAM pixel decoder: RAE's ViT-MAE-style GeneralDecoder + Tanh.

Maps UNI tokens (B, 256, 1024) to images (B, 3, 256, 256) in [-1, 1]. The architecture is
the `decoder:` section of the config (GeneralDecoder fields; the input width is model.input_dim).

The wrapper attribute `decoder` defines the checkpoint state_dict prefix ("decoder.*").
"""

import torch
import torch.nn as nn

from third_party.rae.decoder import GeneralDecoder
from third_party.rae.utils import ViTMAEConfig


class StreamDecoder(nn.Module):
    """GeneralDecoder with a Tanh output (bounded [-1, 1] pixels, the range the Kaiko discriminator expects)."""

    def __init__(self, config: dict):
        super().__init__()
        vit_cfg = ViTMAEConfig(hidden_size=config["model"]["input_dim"], **config["decoder"])
        num_patches = (vit_cfg.image_size // vit_cfg.patch_size) ** 2
        self.decoder = GeneralDecoder(vit_cfg, num_patches=num_patches)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.decoder.unpatchify(self.decoder(features).logits))

    @property
    def last_layer(self) -> torch.Tensor:
        """Weight of the final linear layer (for the adaptive GAN weight)."""
        return self.decoder.decoder_pred.weight


def load_decoder(checkpoint_path: str, config: dict, device: torch.device) -> StreamDecoder:
    """StreamDecoder with the EMA weights of a stream.train_decoder checkpoint (eval mode, frozen)."""
    decoder = StreamDecoder(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder.load_state_dict(ckpt["ema"], strict=True)  # every key of a stream.train_decoder checkpoint
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder
