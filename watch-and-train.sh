#!/usr/bin/env bash
set -euo pipefail
# source .venv/bin/activate
# ==========================================================
# Watch & Train Script for watchexec v2
# ==========================================================

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

# Command to run on changes
# CMD=(python -c "import time; print('hello'); time.sleep(10)") # dry-run
CMD=(.venv/bin/python -u scripts/train.py)
# CMD=(python -X faulthandler -c "import torch; print('ok')")



# Run watchexec
echo "Launching watchexec with command: ${CMD[*]}"
TORCH_DISABLE_SHARED_MEMORY=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
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
    
