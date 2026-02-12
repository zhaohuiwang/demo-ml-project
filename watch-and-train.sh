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

# Make sure CUDA libraries are visible (adjust path if needed)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# ------------------------------
# Command to safely run training with GPU fallback
# ------------------------------
# This small Python snippet detects CUDA and falls back to CPU if needed
PY_CMD=$(cat << 'EOF'
import os
import torch
import sys

try:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available, using CPU")
except Exception as e:
    print("CUDA init failed, falling back to CPU:", e)

# Run the actual training script
training_script = os.path.join("scripts", "train.py")
sys.argv = [training_script] + sys.argv[1:]
with open(training_script, "rb") as f:
    code = compile(f.read(), training_script, "exec")
    exec(code)
EOF
)

CMD=(python -X faulthandler -u -c "$PY_CMD")

# ------------------------------
# Directories & extensions to watch
# ------------------------------
WATCH_DIRS=("src" "scripts" "conf" "data/processed")
IGNORE_DIRS=("mlruns" "model_export" "data/raw")
IGNORE_GLOBS=("**/*.log" "**/*.log.*")
EXTS=("py" "yaml" "yml" "toml" "parquet")

# Build watch flags
WATCH_FLAGS=()
for dir in "${WATCH_DIRS[@]}"; do
    WATCH_FLAGS+=(--watch "$dir")
done

# Build ignore flags
IGNORE_FLAGS=()
for dir in "${IGNORE_DIRS[@]}"; do
    IGNORE_FLAGS+=(--ignore "$dir")
done
for glob in "${IGNORE_GLOBS[@]}"; do
    IGNORE_FLAGS+=(--ignore "$glob")
done

# Build extension flags
EXT_FLAGS=()
for ext in "${EXTS[@]}"; do
    EXT_FLAGS+=(--exts "$ext")
done

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



# # ------------------------------
# # Activate virtual environment
# # ------------------------------
# if [ -f ".venv/bin/activate" ]; then
#     echo "Activating virtual environment..."
#     source .venv/bin/activate
# else
#     echo "Warning: virtual environment not found. Using system Python."
# fi

# # ------------------------------
# # Environment variables for PyTorch
# # ------------------------------
# export TORCH_DISABLE_SHARED_MEMORY=1   # Avoid shared memory issues
# export OMP_NUM_THREADS=1               # Limit OpenMP threads
# export MKL_NUM_THREADS=1               # Limit MKL threads
# export TORCH_NUM_THREADS=1             # Limit PyTorch threads

# # ------------------------------
# # Directories & extensions
# # ------------------------------
# # Directories to watch
# WATCH_DIRS=("src" "scripts" "conf" "data/processed")

# # Directories / files to ignore
# IGNORE_DIRS=("mlruns" "model_export" "data/raw")
# IGNORE_GLOBS=("**/*.log" "**/*.log.*")

# # File extensions to watch
# EXTS=("py" "yaml" "yml" "toml" "parquet")

# # Build the --watch flags
# WATCH_FLAGS=()
# for dir in "${WATCH_DIRS[@]}"; do
#     WATCH_FLAGS+=(--watch "$dir")
# done

# # Build the --ignore flags
# IGNORE_FLAGS=()
# for dir in "${IGNORE_DIRS[@]}"; do
#     IGNORE_FLAGS+=(--ignore "$dir")
# done
# for glob in "${IGNORE_GLOBS[@]}"; do
#     IGNORE_FLAGS+=(--ignore "$glob")
# done

# # Build the --ext flags
# EXT_FLAGS=()
# for ext in "${EXTS[@]}"; do
#     EXT_FLAGS+=(--exts "$ext")
# done


# # ------------------------------
# # Command to run on changes
# # ------------------------------
# # Using -X faulthandler to debug import hangs
# CMD=(python -X faulthandler -u scripts/train.py)
# # CMD=(python -c "import time; print('hello'); time.sleep(10)") # dry-run
# # CMD=(python -X faulthandler -c "import torch; print('ok')")



# # ------------------------------
# # Launch watchexec
# # ------------------------------
# echo "Launching watchexec with command: ${CMD[*]}"
# watchexec \
#     "${WATCH_FLAGS[@]}" \
#     "${IGNORE_FLAGS[@]}" \
#     "${EXT_FLAGS[@]}" \
#     --clear \
#     --restart \
#     --verbose \
#     --shell=none \
#     -- \
#     "${CMD[@]}"
    
