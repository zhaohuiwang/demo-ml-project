#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Stable ML Dev Loop for WSL + PyTorch (CPU-only)
# Using entr instead of watchexec
# ==========================================================

echo "[watch_train] Starting ML dev loop (WSL, CPU-only)"

# ------------------------------
# Activate virtual environment
# ------------------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found"
    exit 1
fi

# ------------------------------
# PyTorch stability settings
# ------------------------------
export CUDA_VISIBLE_DEVICES=""
export TORCH_DISABLE_SHARED_MEMORY=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# Optional: prevents some OpenMP fork issues
export KMP_INIT_AT_FORK=FALSE
export KMP_BLOCKTIME=0

# ------------------------------
# Build file watch list
# ------------------------------
WATCH_PATHS="src scripts conf data/processed"

echo "[watch_train] Watching paths: $WATCH_PATHS"

# ------------------------------
# Dev Loop
# ------------------------------
while true; do
    echo ""
    echo "============================================"
    echo "[watch_train] Restart triggered at $(date)"
    echo "============================================"
    
    # Run training (exec replaces shell process cleanly)
    exec python -X faulthandler -u scripts/train.py
done < <(
    find $WATCH_PATHS -type f \
    ! -path "*/mlruns/*" \
    ! -path "*/model_export/*" \
    ! -path "*/data/raw/*" \
    ! -name "*.log" \
    ! -name "*.log.*" \
    | entr -d -p echo "change detected"
)
