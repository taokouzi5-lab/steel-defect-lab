"""FastAPI application assembly (T3.5).

``create_app`` wires everything: settings, model registry, ONNX engine,
sample catalog, predict service, the ``/api/v1`` router, rate limiting,
request-id middleware, CORS (development), and the envelope-producing
exception handlers.

Error policy: clients get a stable ``{code, message, status}`` envelope and
never stack traces; details go to the server log.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import routes
from app.config import Settings, set_active_settings
from app.core import envelope
from app.core.body_limit import BodySizeLimitMiddleware
from app.core.errors import AppError
from app.core.rate_limit import limiter
from app.core.spa_static import SPAStaticFiles
from app.inference.model_registry import ModelRegistry
from app.inference.onnx_engine import OnnxEngine
from app.services.gradcam_backend import GradCAMBackend
from app.services.predict_service import PredictService
from app.services.sample_catalog import SampleCatalog

logger = logging.getLogger(__name__)

_VALID_CLIENT_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# Multipart framing overhead allowance on top of the file cap: boundary,
# content-disposition headers and form fields must fit too.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application (state, middleware, handlers, routes)."""
    settings = settings or Settings.from_env()
    set_active_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload_models:
            ids = [model.id for model in app.state.model_registry.list()]
            loaded = app.state.onnx_engine.load_all(ids)
            logger.info("preloaded %d/%d onnx sessions", len(loaded), len(ids))
        yield

    app = FastAPI(
        title="DefectLens Steel Defect API",
        version="0.1.0",
        description="Steel surface defect classification with Grad-CAM "
        "explainability and carbon estimation.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    # -- application state ------------------------------------------------
    registry = ModelRegistry(settings.metrics_path, settings.model_dir, settings.carbon_country)
    engine = OnnxEngine(settings.model_dir, timeout_s=settings.request_timeout_s)
    catalog = SampleCatalog(settings.samples_dir)
    gradcam_backend = GradCAMBackend(settings.checkpoint_dir)
    service = PredictService(
        engine, registry, settings, catalog=catalog, gradcam_backend=gradcam_backend
    )
    app.state.settings = settings
    app.state.model_registry = registry
    app.state.onnx_engine = engine
    app.state.sample_catalog = catalog
    app.state.predict_service = service

    # -- rate limiting -----------------------------------------------------
    limiter.enabled = settings.rate_limit_enabled
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # -- CORS (development) ------------------------------------------------
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,  # wildcard-incompatible; no cookies here
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- request body precheck (H1) -----------------------------------------
    # Starlette spools multipart bodies to a SpooledTemporaryFile (disk past
    # ~1 MB) before app validation runs, so reject declared-oversized bodies
    # from the Content-Length header alone, before any parsing. Chunked
    # bodies without Content-Length are a residual risk bounded by the
    # stream cap in app.core.validation.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_upload_bytes() + _MULTIPART_OVERHEAD_BYTES,
    )

    # -- exception handlers ------------------------------------------------
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    # -- request id middleware ---------------------------------------------
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        client_id = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            client_id if _VALID_CLIENT_REQUEST_ID.match(client_id) else envelope.new_request_id()
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.include_router(routes.router)

    # -- frontend static hosting (single-container topology) ----------------
    # Mounted last: /api/v1/* routes win, everything else falls through to
    # the SPA build with an index.html fallback for client-side routes.
    dist_dir = settings.frontend_dist_dir
    if (dist_dir / "index.html").is_file():
        app.mount(
            "/",
            SPAStaticFiles(directory=str(dist_dir), html=True),
            name="frontend",
        )
        logger.info("serving frontend from %s", dist_dir)
    else:
        logger.info(
            "frontend dist not found at %s; API-only mode (build the "
            "frontend or set FRONTEND_DIST_DIR)",
            dist_dir,
        )

    return app


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _current_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", envelope.new_request_id())


def _json_error(request: Request, code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=envelope.failure(code, message, status_code, _current_request_id(request)),
    )


async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("request %s failed with %s: %s", _current_request_id(request), exc.code, exc)
    return _json_error(request, exc.code, exc.message, exc.status_code)


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning("request %s rate limited: %s", _current_request_id(request), exc.detail)
    return _json_error(request, "rate_limited", "request rate limit exceeded", 429)


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info(
        "request %s failed schema validation: %s", _current_request_id(request), exc.errors()
    )
    return _json_error(request, "validation_error", "invalid request parameters", 422)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _json_error(request, "http_error", str(exc.detail), exc.status_code)


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error for request %s", _current_request_id(request))
    return _json_error(request, "internal_error", "an unexpected error occurred", 500)
