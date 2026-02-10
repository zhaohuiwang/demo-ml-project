

# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime AS builder

# ── Install standalone uv binary (official recommendation) ──────────────────────
ARG UV_VERSION=0.6.1
ADD https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (caching)
COPY pyproject.toml uv.lock* requirements.txt* ./

# Install deps with uv (fast & cached)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache .[dev]   # or uv sync --frozen if using uv.lock

# ── Final runtime stage ────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Non-root user (security)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages + project code
COPY --from=builder /opt/conda /opt/conda
COPY --chown=appuser:appuser . .

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns

CMD ["python", "scripts/train.py"]