"""Riemannian operations on the product of spheres (S^{d-1})^N.

Every token of a (B, N, d) feature map lives on its own unit sphere S^{d-1}; all
operations act per token along the last dimension.
"""

from typing import Union

import torch


def normalize_tokens(z: torch.Tensor) -> torch.Tensor:
    """Per-token L2 normalization onto S^{d-1}."""
    return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def log_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Log_x(y) in T_x S^{d-1}.

    Near x = y the ratio theta * y_proj / ||y_proj|| tends to y - x, and the clamp on
    ||y_proj|| prevents a division by zero.
    """
    cos_theta = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cos_theta)
    y_proj = y - cos_theta * x
    y_proj_norm = y_proj.norm(dim=-1, keepdim=True).clamp(min=eps)
    return theta * y_proj / y_proj_norm


def exp_map(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exp_x(v) for v in T_x S^{d-1}. For ||v|| -> 0 this tends to x + v."""
    theta = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.cos(theta) * x + torch.sin(theta) * v / theta


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
    """Geodesic interpolation SLERP(z0, z1, t); t is a float or broadcastable to (B, N, 1)."""
    cos_omega = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    w0 = torch.sin((1 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    return w0 * z0 + w1 * z1


def sample_uniform_sphere(B: int, N: int, d: int, device: torch.device) -> torch.Tensor:
    """Uniform samples on (S^{d-1})^N, shape (B, N, d)."""
    return normalize_tokens(torch.randn(B, N, d, device=device))


def project_to_tangent(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Remove the per-token radial component: v - <v, z> z."""
    dots = (v * z).sum(dim=-1, keepdim=True)
    return v - dots * z


def sample_tangent_gaussian(mu: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    """Exp_mu(sigma * eps) with eps ~ N(0, I) projected onto T_mu (S^{d-1})^N."""
    eps = torch.randn_like(mu)
    eps = eps - (eps * mu).sum(dim=-1, keepdim=True) * mu
    return exp_map(mu, sigma * eps)
