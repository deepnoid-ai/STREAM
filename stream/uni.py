"""Frozen UNI encoder (ViT-L/16, https://huggingface.co/MahmoodLab/uni).

UNI is gated on the Hugging Face Hub: request access on the model page, then run
`huggingface-cli login` or export HF_TOKEN.
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_uni() -> torch.nn.Module:
    """UNI with dynamic image size, so a 256 x 256 patch gives 16 x 16 = 256 tokens (on CPU)."""
    import timm
    return timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                             init_values=1e-5, dynamic_img_size=True)


def patch_tokens(all_tokens: torch.Tensor) -> torch.Tensor:
    """forward_features output (B, 257, 1024) -> drop CLS -> per-token L2-normalized (B, 256, 1024)."""
    return F.normalize(all_tokens[:, 1:, :], dim=-1)
