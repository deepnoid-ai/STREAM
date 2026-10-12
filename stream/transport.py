"""Stochastic bridge, logit-normal time sampling and the chord loss (STREAM Stage 1)."""

import math

import torch
import torch.nn.functional as F

from .geometry import slerp, sample_tangent_gaussian


def bridge_perturb(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor,
                   sigma_max: float) -> torch.Tensor:
    """Sample the stochastic bridge z_t = Exp_{mu_t}(sigma(t) eps).

    mu_t = SLERP(z0, z1, t), sigma(t) = sigma_max * sin(pi t), eps a tangent Gaussian at mu_t.
    z0, z1: (B, N, d) on (S^{d-1})^N; t: (B,).
    """
    t_expanded = t[:, None, None]
    mu_t = slerp(z0, z1, t_expanded)
    sigma_t = sigma_max * torch.sin(math.pi * t_expanded)
    return sample_tangent_gaussian(mu_t, sigma_t)


def sample_logit_normal(batch_size: int, mu: float, s: float, eps_t: float,
                        device: torch.device) -> torch.Tensor:
    """t = sigmoid(N(mu, s^2)) clamped to [eps_t, 1 - eps_t], shape (B,)."""
    normal_samples = torch.randn(batch_size, device=device) * s + mu
    t = torch.sigmoid(normal_samples)
    return t.clamp(eps_t, 1.0 - eps_t)


def chord_loss(z1_hat: torch.Tensor, z1: torch.Tensor, t: torch.Tensor,
               min_one_minus_t: float) -> torch.Tensor:
    """Chord loss ||z1_hat - z1||^2 / (1 - t)^2 for the x-prediction z1_hat (mean over B, N).

    (1 - t) is floored at min_one_minus_t and the weighted per-token loss is capped at 1e4.
    """
    z1_hat = F.normalize(z1_hat, dim=-1)
    one_minus_t = (1.0 - t).clamp(min=min_one_minus_t)[:, None]  # (B, 1)
    per_token_loss = (z1_hat - z1).pow(2).sum(dim=-1)            # (B, N)
    weighted = per_token_loss / one_minus_t.pow(2)
    weighted = weighted.clamp(max=1e4)
    return weighted.mean()
