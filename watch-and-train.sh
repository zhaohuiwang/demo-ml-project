#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Watch & Train Script for watchexec v2
# ==========================================================


# ------------------------------
# Activate virtual environment
# ------------------------------
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: virtual environment not found. Using system Python."
fi

# ------------------------------
# Environment variables for PyTorch
# ------------------------------
export TORCH_DISABLE_SHARED_MEMORY=1   # Avoid shared memory issues
export OMP_NUM_THREADS=1               # Limit OpenMP threads
export MKL_NUM_THREADS=1               # Limit MKL threads
export TORCH_NUM_THREADS=1             # Limit PyTorch threads

# ------------------------------
# Directories & extensions
# ------------------------------
# Directories to watch
WATCH_DIRS=("src" "scripts" "conf" "data/processed")

# Directories / files to ignore
IGNORE_DIRS=("mlruns" "model_export" "data/raw")
IGNORE_GLOBS=("**/*.log" "**/*.log.*")

# File extensions to watch
EXTS=("py" "yaml" "yml" "toml" "parquet")

# Build the --watch flags
WATCH_FLAGS=()
for dir in "${WATCH_DIRS[@]}"; do
    WATCH_FLAGS+=(--watch "$dir")
done

# Build the --ignore flags
IGNORE_FLAGS=()
for dir in "${IGNORE_DIRS[@]}"; do
    IGNORE_FLAGS+=(--ignore "$dir")
done
for glob in "${IGNORE_GLOBS[@]}"; do
    IGNORE_FLAGS+=(--ignore "$glob")
done

# Build the --ext flags
EXT_FLAGS=()
for ext in "${EXTS[@]}"; do
    EXT_FLAGS+=(--exts "$ext")
done


# ------------------------------
# Command to run on changes
# ------------------------------
# Using -X faulthandler to debug import hangs
CMD=(python -X faulthandler -u scripts/train.py)
# CMD=(python -c "import time; print('hello'); time.sleep(10)") # dry-run
# CMD=(python -X faulthandler -c "import torch; print('ok')")



# ------------------------------
# Launch watchexec
# ------------------------------
echo "Launching watchexec with command: ${CMD[*]}"
watchexec \
    "${WATCH_FLAGS[@]}" \
    "${IGNORE_FLAGS[@]}" \
    "${EXT_FLAGS[@]}" \
    --clear \
    --restart \
    --verbose \
    --shell=none \
    -- \
    "${CMD[@]}"
    
