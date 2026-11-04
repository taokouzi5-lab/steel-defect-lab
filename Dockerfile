# syntax=docker/dockerfile:1

# ============================================================================
# DefectLens — production single-container image (T5.1)
# ----------------------------------------------------------------------------
# One container serves the FastAPI API and the built SPA (gunicorn +
# uvicorn workers, non-root user). Target: ModelScope 创空间 Docker
# deployment — port 7860, 8 GB memory budget, Linux/amd64.
#
# Asset policy (no conditional COPY or build-arg switches needed):
#   ONNX models (artifacts/onnx) and Grad-CAM checkpoints
#   (artifacts/checkpoints) are gitignored but baked in whenever they exist
#   in the BUILD CONTEXT (local checkout / ModelScope upload). A context
#   without them — e.g. a bare git clone or the CI checkout — builds the
#   SAME image: the app already degrades gracefully at runtime
#   (/api/v1/health stays 200, /api/v1/models reports available=false,
#   POST /predict answers 503 model_not_ready). The CI docker-build job
#   asserts exactly this boot + degradation contract, so the production
#   Dockerfile itself is what gets verified — no drift-prone CI variant.
#
# Dependency rules (docs/deployment.md):
#   torch/torchvision — CPU wheels from download.pytorch.org; the container
#     must never inherit CUDA builds.
#   onnxruntime — backend/pyproject.toml pins ==1.20.1 for the Windows dev
#     machine (1.21+ wheels fail to import there); containers install
#     >=1.21.2 explicitly, AFTER the editable install (installing it first
#     would be downgraded back to the pin).
# ============================================================================


# -- stage 1: frontend build --------------------------------------------------
# ModelScope builds have no npm cache; building the SPA in-image is the
# deterministic choice (no reliance on a host-built, gitignored dist).
FROM node:24-alpine AS frontend-build

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# -- stage 2: runtime ---------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime required by the torch CPU wheels.
# curl: healthcheck probe (the slim base does not ship it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch first, in its own layer (~1 GB): cached across code changes.
# Version floor comes from backend/pyproject.toml (torch>=2.3.0,
# torchvision>=0.18.0); the CPU index resolves the newest matching pair.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/ /app/backend/

# Runtime deps only (no [dev] extras). The onnxruntime override must come
# AFTER the editable install: the 1.20.1 pin is a Windows-dev-machine
# workaround (see header / docs/deployment.md), while containers need
# >=1.21.2 (security advisories + dynamic-quantizer fixes in the 1.20.x
# line). gunicorn is the container process manager; it is not a backend
# dependency.
RUN pip install -e backend/ \
    && pip install "onnxruntime>=1.21.2" "gunicorn>=23"

# Baked-in assets. artifacts/ exists in any git checkout (metrics.json is
# tracked); the onnx/ and checkpoints/ subdirectories ride along only when
# present in the build context (see header).
COPY artifacts/ /app/artifacts/

COPY --from=frontend-build /build/dist /app/frontend/dist

# Non-root runtime user. With PYTHONDONTWRITEBYTECODE=1 /app is import-only
# at runtime, so ownership is fixed once and nothing needs write access.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app

# Production defaults (docs/deployment.md). CORS_ORIGINS="" (explicit empty
# string) turns the CORS middleware off: the SPA and API share one origin.
ENV PORT=7860 \
    PRELOAD_MODELS=1 \
    RATE_LIMIT_ENABLED=1 \
    DEFECTLENS_ENABLE_DOCS=0 \
    CORS_ORIGINS="" \
    CARBON_COUNTRY=China

EXPOSE 7860

# /api/v1/health always returns 200 while the process is up (it reports the
# preloaded models, not the availability of any specific asset), so the
# probe works with or without baked models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/api/v1/health || exit 1

CMD ["gunicorn", "--config", "/app/backend/gunicorn.conf.py", "app.asgi:app"]
