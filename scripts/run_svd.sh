#!/bin/bash
# =============================================================================
# STREAM spectral basis extraction (Jacobian SVD cache), run between Stage 1 and Stage 2:
# k-means + per-token Jacobian SVD on several GPUs + merge.
# Usage:
#   bash scripts/run_svd.sh \
#     --config configs/brca.yaml \
#     --dit-checkpoint path/to/dit_run/checkpoints/final_0095000.pt \
#     --uni-embedding-path path/to/uni_features \
#     --output-dir path/to/svd
#
# Optional: --n-centroids K (50)  --n-features N (50000)  --svd-m M (600)
#           --svd-t-eval T (0.99)  --jvp-chunk-size C (5)  --num-gpus G (8)
# Output:   <output-dir>/centroids.pt, per_token_svd_partial_*.pt, per_token_svd_cache.pt
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIG=""
DIT_CHECKPOINT=""
UNI_EMBEDDING_PATH=""
OUTPUT_DIR=""
N_CENTROIDS=50
N_FEATURES=50000
SVD_M=600
SVD_T_EVAL=0.99
JVP_CHUNK_SIZE=5
NUM_GPUS=8

usage() { sed -n '2,15p' "$0"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --dit-checkpoint) DIT_CHECKPOINT="$2"; shift 2 ;;
        --uni-embedding-path) UNI_EMBEDDING_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --n-centroids) N_CENTROIDS="$2"; shift 2 ;;
        --n-features) N_FEATURES="$2"; shift 2 ;;
        --svd-m) SVD_M="$2"; shift 2 ;;
        --svd-t-eval) SVD_T_EVAL="$2"; shift 2 ;;
        --jvp-chunk-size) JVP_CHUNK_SIZE="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "${CONFIG}" || -z "${DIT_CHECKPOINT}" || -z "${UNI_EMBEDDING_PATH}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: --config, --dit-checkpoint, --uni-embedding-path and --output-dir are required."
    usage; exit 1
fi
mkdir -p "${OUTPUT_DIR}"
CENTROIDS_PATH="${OUTPUT_DIR}/centroids.pt"

echo "=== Step 1: k-means (K=${N_CENTROIDS}, ${N_FEATURES} feature maps) ==="
CUDA_VISIBLE_DEVICES=0 python -m stream.kmeans_centroids \
    --uni-embedding-path "${UNI_EMBEDDING_PATH}" \
    --n-centroids "${N_CENTROIDS}" \
    --n-features "${N_FEATURES}" \
    --output-path "${CENTROIDS_PATH}"

echo "=== Step 2: per-token Jacobian SVD (m=${SVD_M}, t=${SVD_T_EVAL}) on ${NUM_GPUS} GPUs ==="
BASE=$(( N_CENTROIDS / NUM_GPUS ))
EXTRA=$(( N_CENTROIDS % NUM_GPUS ))
PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    if [ "${gpu}" -lt "${EXTRA}" ]; then
        C_START=$(( gpu * (BASE + 1) ))
        C_END=$(( C_START + BASE + 1 ))
    else
        C_START=$(( EXTRA * (BASE + 1) + (gpu - EXTRA) * BASE ))
        C_END=$(( C_START + BASE ))
    fi
    [ "${C_START}" -ge "${C_END}" ] && continue
    echo "GPU ${gpu}: centroids [${C_START}, ${C_END})"
    CUDA_VISIBLE_DEVICES=${gpu} python -m stream.extract_svd \
        --config "${CONFIG}" \
        --dit-checkpoint "${DIT_CHECKPOINT}" \
        --centroids-path "${CENTROIDS_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --centroid-start "${C_START}" \
        --centroid-end "${C_END}" \
        --svd-m "${SVD_M}" \
        --svd-t-eval "${SVD_T_EVAL}" \
        --jvp-chunk-size "${JVP_CHUNK_SIZE}" &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAILED=$((FAILED + 1))
done
if [ "${FAILED}" -gt 0 ]; then
    echo "ERROR: ${FAILED} SVD worker(s) failed; not merging."
    exit 1
fi

echo "=== Step 3: merge ==="
python -m stream.merge_svd \
    --partial-dir "${OUTPUT_DIR}" \
    --output-path "${OUTPUT_DIR}/per_token_svd_cache.pt"
echo "Spectral basis: ${OUTPUT_DIR}/per_token_svd_cache.pt"
