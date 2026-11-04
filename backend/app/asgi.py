"""ASGI entrypoint for the gunicorn/uvicorn container (T5.1).

Gunicorn's UvicornWorker does not reliably resolve application factories
(``create_app``) from the ``WSGI_APP`` string, so the container runs this
module instead: the app is built once at import time, per worker process,
from the process environment (see docs/deployment.md for the container
environment variables).
"""

from app.main import create_app

app = create_app()
