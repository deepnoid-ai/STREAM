"""Per-token Jacobian SVD of the DiT velocity field at the k-means centroids (STREAM spectral basis extraction).

For each centroid z (from stream.kmeans_centroids) in [--centroid-start, --centroid-end) and
each token n:
  1. m random Gaussian tangent probes at z_n (generator seed = centroid_index * N + n, so the
     result does not depend on how centroids are split across workers), unit-normalized,
  2. forward-mode JVPs through v(z) = Log_z(normalize(DiT(z, t_eval))) / (1 - t_eval),
  3. the token-n responses are projected onto T_{z_n} S^{d-1}; their SVD gives all m left
     singular vectors U (d, m) and singular values S (m,).
Writes <output-dir>/per_token_svd_partial_<start>_<end>.pt; stream.merge_svd joins the partials.
Normally run through scripts/run_svd.sh (one worker per GPU).

Usage (one GPU, all centroids):
    python -m stream.extract_svd --config configs/brca.yaml \
        --dit-checkpoint path/to/dit_run/checkpoints/final_0095000.pt \
        --centroids-path path/to/svd/centroids.pt --output-dir path/to/svd
"""

import argparse
import logging
import os
import time

import torch
import torch._dynamo
import torch.nn.functional as F
import yaml
from torch.func import jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from .dit import load_dit
from .geometry import log_map, normalize_tokens
from .utils import atomic_torch_save, require_path

logger = logging.getLogger(__name__)


def make_velocity_fn(dit, N, d, t_eval, device):
    """v(z) = Log_z(normalize(DiT(z, t_eval))) / (1 - t_eval) on flat (N*d,) tensors, for torch.func."""
    def velocity_fn(z_flat):
        z = z_flat.reshape(1, N, d)
        t = torch.tensor([t_eval], device=device)
        x1_hat = normalize_tokens(dit(z, t))
        return (log_map(z, x1_hat) / (1 - t_eval + 1e-8)).reshape(-1)
    return velocity_fn


def per_token_svd(dit, centroids, c_start, c_end, m, t_eval, jvp_chunk_size, device):
    """Left singular vectors (K', N, d, m) and singular values (K', N, m) for centroids [c_start, c_end)."""
    _, N, d = centroids.shape
    Nd = N * d
    velocity_fn = make_velocity_fn(dit, N, d, t_eval, device)

    # torch.compile'd DiT blocks can be opaque to vmap/jvp: let dynamo fall back to eager.
    prev_suppress = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = True

    all_U, all_S = [], []
    for c_idx in range(c_start, c_end):
        z_c = centroids[c_idx].to(device)
        z_c_flat = z_c.reshape(-1)
        token_U, token_S = [], []
        t0 = time.time()
        for n in range(N):
            z_n = z_c[n]
            gen = torch.Generator(device=device).manual_seed(c_idx * N + n)
            probes = torch.randn(m, d, device=device, generator=gen)
            probes = probes - (probes @ z_n).unsqueeze(1) * z_n.unsqueeze(0)
            probes = F.normalize(probes, dim=-1)

            # Perturb token n only.
            delta_z_batch = torch.zeros(m, Nd, device=device)
            delta_z_batch[:, n * d:(n + 1) * d] = probes

            def single_jvp(tangent):
                return jvp(velocity_fn, (z_c_flat,), (tangent,))[1]

            # Math SDPA backend: the fused kernels have no forward-mode AD.
            with sdpa_kernel(SDPBackend.MATH):
                responses = vmap(single_jvp, chunk_size=jvp_chunk_size)(delta_z_batch)  # (m, Nd)

            W_n = responses[:, n * d:(n + 1) * d]
            W_n = W_n - (W_n @ z_n).unsqueeze(1) * z_n.unsqueeze(0)
            U, S, _ = torch.linalg.svd(W_n.T, full_matrices=False)  # U (d, m), S (m,)
            token_U.append(U.cpu())
            token_S.append(S.cpu())
        all_U.append(torch.stack(token_U))
        all_S.append(torch.stack(token_S))
        logger.info(f"centroid {c_idx} done ({time.time() - t0:.0f}s)")

    torch._dynamo.config.suppress_errors = prev_suppress
    return torch.stack(all_U), torch.stack(all_S)


def main():
    parser = argparse.ArgumentParser(
        description="Per-token Jacobian SVD of the STREAM DiT at k-means centroids.")
    parser.add_argument("--config", required=True,
                        help="STREAM YAML config (the model section must match the checkpoint).")
    parser.add_argument("--dit-checkpoint", required=True,
                        help="DiT checkpoint from stream.train_dit (EMA weights are used).")
    parser.add_argument("--centroids-path", required=True,
                        help="Centroid tensor (K, N, d) from stream.kmeans_centroids.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for per_token_svd_partial_<start>_<end>.pt and its log.")
    parser.add_argument("--centroid-start", type=int, default=0,
                        help="First centroid index (default: %(default)s).")
    parser.add_argument("--centroid-end", type=int, default=None,
                        help="End centroid index, exclusive (default: K, all centroids).")
    parser.add_argument("--svd-m", type=int, default=600,
                        help="Random tangent probes per (centroid, token) (default: %(default)s).")
    parser.add_argument("--svd-t-eval", type=float, default=0.99,
                        help="Time at which the Jacobian is evaluated (default: %(default)s).")
    parser.add_argument("--jvp-chunk-size", type=int, default=5,
                        help="vmap chunk size of the JVPs; lower it to save GPU memory "
                             "(default: %(default)s).")
    args = parser.parse_args()

    with open(require_path(args.config, "--config")) as f:
        config = yaml.safe_load(f)
    require_path(args.dit_checkpoint, "--dit-checkpoint")
    centroids = torch.load(require_path(args.centroids_path, "--centroids-path"),
                           map_location="cpu", weights_only=True)
    n_clusters = centroids.shape[0]
    c_start = args.centroid_start
    c_end = n_clusters if args.centroid_end is None else args.centroid_end
    if not 0 <= c_start < c_end <= n_clusters:
        raise ValueError(f"Invalid centroid range [{c_start}, {c_end}) for {n_clusters} centroids.")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format=f"[%(asctime)s] [centroids {c_start}-{c_end}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.output_dir, f"svd_{c_start}_{c_end}.log"))])
    logger.info(f"DiT {args.dit_checkpoint}; centroids [{c_start}, {c_end}) of {n_clusters}; "
                f"m={args.svd_m}, t_eval={args.svd_t_eval}")

    device = torch.device("cuda:0")
    dit = load_dit(args.dit_checkpoint, config, device)
    U, S = per_token_svd(dit, centroids, c_start, c_end, args.svd_m, args.svd_t_eval,
                         args.jvp_chunk_size, device)

    path = os.path.join(args.output_dir, f"per_token_svd_partial_{c_start}_{c_end}.pt")
    atomic_torch_save({
        "centroids": centroids[c_start:c_end],
        "centroid_U_T": U,        # (K', N, d, m)
        "centroid_S_top": S,      # (K', N, m)
        "n_clusters": n_clusters,
        "centroid_start": c_start,
        "centroid_end": c_end,
        "m": args.svd_m,
        "t_eval": args.svd_t_eval,
        "type": "per_token_partial",
    }, path)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
