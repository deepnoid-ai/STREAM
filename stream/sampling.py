"""Sampling: Riemannian midpoint integration of the DiT flow on (S^{d-1})^N.

Velocity v(z, t) = Log_z(normalize(DiT(z, t))) / max(1 - t, 1e-3) on the uniform grid
t_i = i / n_steps, starting from z_0 ~ Uniform((S^{d-1})^N); each step is a geometrically
exact midpoint step (Exp map), with the midpoint velocity projected back to T_z.
"""

import logging
import time

import torch

from .geometry import exp_map, log_map, normalize_tokens, project_to_tangent, sample_uniform_sphere

logger = logging.getLogger(__name__)


def velocity(dit, z, t, B, device):
    x1_hat = normalize_tokens(dit(z, torch.full((B,), t, device=device)))
    return log_map(z, x1_hat) / max(1 - t, 1e-3)


@torch.no_grad()
def generate_midpoint(dit, B: int, n_steps: int, device: torch.device) -> torch.Tensor:
    """B samples (B, N, d) on (S^{d-1})^N."""
    z = sample_uniform_sphere(B, dit.num_tokens, dit.input_dim, device)
    t_grid = [i / n_steps for i in range(n_steps + 1)]
    for i in range(n_steps):
        t = t_grid[i]
        dt = t_grid[i + 1] - t_grid[i]
        z_mid = exp_map(z, velocity(dit, z, t, B, device) * dt / 2)
        v_mid = velocity(dit, z_mid, t + dt / 2, B, device)
        z = exp_map(z, project_to_tangent(v_mid * dt, z))
    return normalize_tokens(z)


@torch.no_grad()
def generate_features(dit, num_gen: int, batch_size: int, n_steps: int, seed: int,
                      device: torch.device) -> torch.Tensor:
    """num_gen feature maps as a CPU float32 tensor.

    The global torch RNG is seeded once, so the samples depend on (seed, batch_size, num_gen).
    """
    features = torch.empty(num_gen, dit.num_tokens, dit.input_dim, dtype=torch.float32)
    torch.manual_seed(seed)
    start, n_gen = time.time(), 0
    while n_gen < num_gen:
        batch = min(batch_size, num_gen - n_gen)
        features[n_gen:n_gen + batch] = generate_midpoint(dit, batch, n_steps, device).cpu()
        n_gen += batch
        if n_gen % (batch_size * 10) == 0 or n_gen >= num_gen:
            logger.info(f"  generated {n_gen:,}/{num_gen:,} ({n_gen / (time.time() - start):.1f}/s)")
    return features
