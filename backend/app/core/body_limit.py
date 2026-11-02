"""ASGI precheck that rejects oversized request bodies before parsing (H1).

Starlette's multipart parser spools the entire request body to a
``SpooledTemporaryFile`` — which rolls over to a real temp file past ~1 MB —
*before* application code runs, so the stream cap in ``app.core.validation``
cannot protect the framework layer. This middleware inspects the
``Content-Length`` header and answers 413 without reading the body when the
declared size already exceeds the upload cap plus a multipart-overhead
allowance.

Residual risk: ``Transfer-Encoding: chunked`` bodies carry no
Content-Length and are not prechecked here; they are still bounded by the
stream cap in ``app.core.validation`` once parsing starts.
"""

from __future__ import annotations

import logging

from starlette.responses import JSONResponse

from app.core import envelope
from app.core.errors import PayloadTooLargeError

logger = logging.getLogger(__name__)


def content_length_exceeds(raw_header: bytes | None, limit: int) -> bool:
    """True when a Content-Length header declares a body over ``limit``.

    Missing or malformed headers return False — the request proceeds and is
    bounded by downstream validation instead.
    """
    if raw_header is None:
        return False
    try:
        declared = int(raw_header.strip())
    except ValueError:
        return False
    return declared > limit


class BodySizeLimitMiddleware:
    """Reject declared-oversized requests with a 413 envelope, pre-parsing."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        raw_length = headers.get(b"content-length")
        if content_length_exceeds(raw_length, self.max_bytes):
            logger.info(
                "rejected request with Content-Length %s exceeding the %d-byte body cap",
                raw_length,
                self.max_bytes,
            )
            response = JSONResponse(
                status_code=413,
                content=envelope.failure(
                    "payload_too_large", PayloadTooLargeError.default_message(), 413
                ),
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
