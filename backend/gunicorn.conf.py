"""Gunicorn configuration for the DefectLens container (T5.1).

Serves the FastAPI app (``app.asgi:app``) through uvicorn workers on the
ModelScope 创空间 Docker convention port 7860 within the 8 GB memory
budget. The bare-uvicorn recipe and the single-process rationale are
documented in docs/deployment.md.
"""

import os

# ModelScope 创空间 Docker convention: listen on 7860 unless PORT is set.
bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"

# Default 1 worker: the slowapi rate limiter uses in-memory storage
# (memory://), so counters live in one process and N workers would multiply
# per-client budgets by N — the single-worker recipe documented in
# docs/deployment.md. It also keeps torch runtime + ONNX session memory to a
# single copy (~0.7 GB RSS) inside the 8 GB budget. Raise WORKERS only when
# a shared rate-limit backend is configured.
workers = int(os.environ.get("WORKERS", "1"))

worker_class = "uvicorn.workers.UvicornWorker"

# Generous heartbeat: a first request may cold-load an ONNX session or a
# Grad-CAM checkpoint on a shared-CPU platform.
timeout = 120
graceful_timeout = 30
keepalive = 5

# The app package lives in /app/backend; chdir makes ``app.asgi``
# importable and anchors relative path resolution there.
chdir = "/app/backend"

# Logs to stdout/stderr (the container runtime captures both).
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
