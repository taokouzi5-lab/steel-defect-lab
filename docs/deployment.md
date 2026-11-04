# DefectLens Backend — Deployment Guide

Deploying the `backend/` FastAPI service to a free ModelScope container (or
any single-instance Linux container). Follow these rules to keep the
security posture from the security review intact.

The repo-root `Dockerfile` (T5.1) is the standard deployment path — it runs
gunicorn with uvicorn workers; see
[Container image (Dockerfile)](#container-image-dockerfile-t51) below. The
uvicorn recipe here remains the reference for bare-uvicorn deployments and
for the worker-count rationale.

## uvicorn parameters (fixed recipe)

```bash
uvicorn app.main:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Every flag matters:

| Flag | Why it must be this |
|------|---------------------|
| `--workers 1` (single worker) | The rate limiter uses slowapi **in-memory** storage (`memory://`): counters live in one process. Multiple workers would each keep their own counters, so per-client budgets would be multiplied by the worker count. Single-worker also keeps the ONNX session cache, the bounded inference executor and the Grad-CAM checkpoint cache warm in one process (no per-worker duplicate cold starts). |
| no `--forwarded-allow-ips=*` / no wildcard XFF trust | The platform gateway in front of your container sets `X-Forwarded-For`. Do **not** pass `--forwarded-allow-ips="*"`: trusting XFF from any peer lets a client spoof its remote address and evade the per-client rate limit. Either omit the flag entirely (uvicorn then ignores XFF and rate-limits by the gateway address) or set it to the platform gateway's specific address/range only. |
| `--factory` | `create_app` is an app factory; `uvicorn` calls it per worker. |

## Environment variables for the container

```bash
export DEFECTLENS_ENABLE_DOCS=0   # /docs, /redoc, /openapi.json off in production
export MAX_UPLOAD_MB=5            # upload cap (ASGI precheck = this + 1 MB multipart overhead)
export PRELOAD_MODELS=1           # warm the ONNX sessions at startup
export REQUEST_TIMEOUT_S=10       # per-inference timeout
export CORS_ORIGINS=              # empty: no browser CORS in production
```

- `DEFECTLENS_ENABLE_DOCS=0` disables the documentation endpoints
  (`/docs`, `/redoc`, `/openapi.json`); they default to enabled for
  development.

## Container image (Dockerfile, T5.1)

The repo-root `Dockerfile` builds the single-container production image for
ModelScope 创空间 Docker deployment (port 7860, 8 GB memory budget).

- **Entrypoint**: gunicorn with uvicorn workers
  (`gunicorn --config /app/backend/gunicorn.conf.py app.asgi:app`),
  **2 workers by default** (each worker holds its own torch runtime and
  ONNX session cache; both fit in the 8 GB budget), overridable via
  `WORKERS`. The worker-count trade-off from the recipe above still holds:
  the slowapi limiter is in-memory per process, so per-client budgets scale
  with the worker count — set `WORKERS=1` to restore the strict
  single-process budget semantics.
- **Port**: listens on `0.0.0.0:7860` (ModelScope Docker convention),
  overridable via `PORT`.
- **Runtime user**: non-root (`app`, uid 10001); `/app` is read-only at
  runtime (`PYTHONDONTWRITEBYTECODE=1`).
- **Frontend**: built inside the image (node:24-alpine stage, `npm ci` +
  `npm run build`) and served by FastAPI from `frontend/dist` — no separate
  web server and no dependence on a host-built `dist`.
- **Health probe**: `curl http://localhost:7860/api/v1/health` — always
  200 while the process is up; `models_loaded` reports how many ONNX
  sessions were preloaded at boot.
- **Dependencies**: torch/torchvision CPU wheels and the
  [onnxruntime container override](#onnxruntime-pinning-l2-policy); system
  packages are only `libgomp1` (torch CPU OpenMP) and `curl` (healthcheck).
- **CORS**: the image sets `CORS_ORIGINS=""` — an explicit empty string
  turns the CORS middleware off (the SPA and the API share one origin);
  leaving the variable unset keeps the dev-server defaults.

### Model assets are baked from the build context

`artifacts/onnx` and `artifacts/checkpoints` are gitignored; the image bakes
them in **when they exist in the build context** (local checkout or
ModelScope upload). Nothing in the Dockerfile hard-fails without them:

- With assets: all four model ids report `available=true` and
  `PRELOAD_MODELS=1` warms the ONNX sessions at boot.
- Without assets (e.g. a bare `git clone` build): the image still builds
  and serves — `/api/v1/health` returns 200 with an empty `models_loaded`,
  `/api/v1/models` lists the four metrics-known models with
  `available=false`, and `POST /predict` answers `503 model_not_ready`
  (envelope). The CI `docker-build` job asserts this boot + degradation
  contract on every push.

## Cold start

- ONNX sessions load lazily on first request per model id unless
  `PRELOAD_MODELS=1`; the first `POST /predict` after boot is therefore
  slower than steady-state latency.
- The Grad-CAM checkpoint loads on the first explain request and is cached
  per architecture afterwards.
- The rate limiter (memory storage) resets when the container restarts or
  is recycled by the platform.

## onnxruntime pinning (L2 policy)

Two distinct environments, two rules:

| Environment | Rule | Why |
|-------------|------|-----|
| **Container (Linux)** | `onnxruntime>=1.21.2` recommended | Containers install their own ORT; newer 1.21+ wheels fix bugs in the older dynamic-quantizer output. Do not inherit the dev pin. |
| **Dev machine (Windows)** | keep `onnxruntime==1.20.1` | 1.21+ Windows wheels fail to import on the dev box (DLL init error) — see `backend/pyproject.toml` and `training/pyproject.toml`. 1.20.x's dynamic quantizer emits 4-input `ConvInteger` nodes that `scripts/quantize.py` rewrites at export time. |

The `backend/pyproject.toml` pin exists for the dev machine; the container
build should install ORT explicitly (e.g. `pip install "onnxruntime>=1.21.2"`)
so the pin is not carried into the image.

## Concurrency model (what the container gets for free)

- Inference: bounded to 2 concurrent `session.run` calls per process via a
  module-level executor + semaphore (`app/inference/onnx_engine.py`).
- Grad-CAM: serialized per architecture by an in-process lock
  (`app/services/gradcam_backend.py`) — required for correctness, since
  concurrent autograd on a shared model corrupts state.
- Both semantics assume a single process — another reason for
  `--workers 1`.

## Known residual risk

`Transfer-Encoding: chunked` uploads carry no `Content-Length`, so the
ASGI precheck (which rejects oversized bodies before multipart parsing)
cannot see them. They are still bounded by the stream cap in
`app/core/validation.py` once parsing starts, but the framework-level
spooling happens first. Prefer rejecting chunked requests at the platform
gateway if available.
