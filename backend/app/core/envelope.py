"""API response envelope and request-id helpers.

Every endpoint except ``/health`` answers with the same envelope::

    {"success": true, "data": <payload>, "error": null, "meta": {...}}
    {"success": false, "data": null, "error": {"code", "message", "status"},
     "meta": {"request_id": "..."}}
"""

from __future__ import annotations

import uuid
from typing import Any

_REQUEST_ID_PATTERN_KEYS = "0123456789abcdef"


def new_request_id() -> str:
    """Return a fresh 32-char hex request id."""
    return uuid.uuid4().hex


def success(data: Any, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the success envelope."""
    return {"success": True, "data": data, "error": None, "meta": meta or {}}


def failure(
    code: str,
    message: str,
    status_code: int,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the error envelope."""
    return {
        "success": False,
        "data": None,
        "error": {"code": code, "message": message, "status": status_code},
        "meta": {"request_id": request_id or new_request_id()},
    }
