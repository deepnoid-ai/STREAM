"""STREAM DiT: a LightningDiT-XL style transformer on UNI token maps.

Adapted from LightningDiT (https://github.com/hustvl/LightningDiT, MIT License). Differences:
  - input (B, 256, 1024) UNI tokens; input_proj = Linear(1024, hidden) replaces the patch embedding,
  - output Linear(hidden, 1024) + per-token L2 normalization (x-prediction on the sphere),
  - unconditional (time embedding only), no unpatchify.
Blocks: SwiGLU FFN, RMSNorm, 2D RoPE and AdaLN-Zero modulation.

Module attribute names (input_proj, t_embedder, pos_embed, feat_rope, blocks, final_layer, ...)
define the checkpoint state_dict keys and must not be renamed.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.lightningdit.pos_embed import VisionRotaryEmbeddingFast
from third_party.lightningdit.rmsnorm import RMSNorm
from third_party.lightningdit.swiglu_ffn import SwiGLUFFN


@torch.compile
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head self-attention with 2D RoPE."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = rope(q)
        k = rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP (as in LightningDiT)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class StreamDiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero modulation, RMSNorm, RoPE attention and SwiGLU."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    @torch.compile
    def forward(self, x, c, feat_rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class StreamFinalLayer(nn.Module):
    """AdaLN + Linear projection to the token dimension (normalized in StreamDiT.forward)."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed 2D sin-cos positional embedding (from LightningDiT)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class StreamDiT(nn.Module):
    """x-prediction DiT for Riemannian flow matching on (S^{d-1})^N.

    forward(z, t): z (B, num_tokens, input_dim) on (S^{d-1})^N, t (B,) in [0, 1]
                   -> per-token L2-normalized prediction of the data endpoint, same shape.
    Constructor arguments are the keys of the `model:` section of the config.
    """

    def __init__(self, input_dim: int, num_tokens: int, hidden_size: int, depth: int,
                 num_heads: int, mlp_ratio: float):
        super().__init__()
        # final_layer.linear is an affine map from hidden_size to input_dim, so with hidden_size <
        # input_dim its outputs (before normalization) would be confined to an affine subspace of
        # R^input_dim of dimension <= hidden_size (cf. RAE, arXiv:2510.11690, Sec. 4.1).
        if hidden_size < input_dim:
            raise ValueError(f"model.hidden_size ({hidden_size}) must be >= model.input_dim ({input_dim}).")
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        grid_size = int(num_tokens**0.5)  # 16 for 256 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size), requires_grad=False)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=hidden_size // num_heads // 2, pt_seq_len=grid_size)
        self.blocks = nn.ModuleList([
            StreamDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = StreamFinalLayer(hidden_size, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, int(self.num_tokens**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # AdaLN-Zero: modulation starts as the identity.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # final_layer.linear keeps its xavier init (unlike DiT): a zero output cannot be
        # normalized onto the sphere and its normalization gradient would overflow.

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(z) + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        x = self.final_layer(x, c)
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def load_dit(checkpoint_path: str, config: dict, device: torch.device) -> StreamDiT:
    """StreamDiT with the EMA weights of a stream.train_dit checkpoint (eval mode, frozen)."""
    model = StreamDiT(**config["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["ema"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model
