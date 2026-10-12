"""Anisotropic decoder noise from the per-token Jacobian SVD cache (stream.merge_svd).

For a feature map z the nearest centroid c (largest sum of per-token cosine similarities) is
chosen per sample, and for every token n with the top-k* left singular vectors U = U[c, n]:

    noise_n = sigma_L * eta_n + (sigma_H - sigma_L) * U U^T eta_n,   eta_n ~ N(0, I_d),

i.e. std sigma_H along the k* high-response directions and sigma_L along the rest, followed by
a projection onto the tangent space at z. k* is the smallest k whose cumulative energy of the
median spectrum (over all centroids and tokens) reaches `energy_threshold`.
"""

import torch

from .geometry import project_to_tangent


def energy_k(S: torch.Tensor, threshold: float) -> int:
    """Smallest k with sum_{i<=k} s_i^2 / sum_i s_i^2 >= threshold for the element-wise median spectrum."""
    m = S.shape[-1]
    median_S = S.float().reshape(-1, m).median(dim=0).values
    s2 = median_S ** 2
    total = s2.sum()
    if total <= 0:
        return m
    hits = (torch.cumsum(s2, dim=0) / total >= threshold).nonzero(as_tuple=True)[0]
    if len(hits) == 0:
        return m
    return min(int(hits[0].item()) + 1, m)


class AnisotropicNoise:
    """Loads the SVD cache, keeps centroids and the top-k* basis on `device`, samples noise."""

    def __init__(self, cache_path: str, energy_threshold: float, device: torch.device):
        # mmap=True: DDP ranks share one page-cached copy of the (large) cache.
        data = torch.load(cache_path, weights_only=False, mmap=True)
        if data.get("type") != "per_token":
            raise ValueError(f"{cache_path} is not a merged per-token SVD cache "
                             f"(type={data.get('type')!r}); run stream.merge_svd.")
        self.m = data["m"]
        self.n_clusters = data["n_clusters"]
        self.k = energy_k(data["centroid_S_top"], energy_threshold)
        self.centroids_flat = data["centroids"].reshape(self.n_clusters, -1).to(device)  # (K, N*d)
        self.U = data["centroid_U_T"][:, :, :, :self.k].to(device).float()  # (K, N, d, k*)

    def sample(self, z: torch.Tensor, sigma_H: float, sigma_L: float) -> torch.Tensor:
        """Anisotropic noise (B, N, d) in the tangent space at z (B, N, d)."""
        B, N, d = z.shape
        nearest = (z.reshape(B, N * d) @ self.centroids_flat.T).argmax(dim=1)  # (B,)
        eta = torch.randn(B, N, d, device=z.device)
        noise = sigma_L * eta
        for c_idx in nearest.unique().tolist():  # ascending cluster order
            mask = nearest == c_idx
            U_c = self.U[c_idx]                                       # (N, d, k)
            coeffs = torch.einsum("ndk,bnd->bnk", U_c, eta[mask])
            proj = torch.einsum("ndk,bnk->bnd", U_c, coeffs)
            noise[mask] = noise[mask] + (sigma_H - sigma_L) * proj
        return project_to_tangent(noise, z)
