"""slowapi rate limiting wrapper (T3.6).

A single module-level ``Limiter`` instance decorates the routes; its
``enabled`` flag is set per app in ``create_app`` (so the test environment
can disable it via ``RATE_LIMIT_ENABLED=0``) and the per-minute budget is
resolved dynamically at request time from the active settings, so tests can
lower it with ``RATE_LIMIT_PER_MIN`` without rebuilding decorators.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_active_settings

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
    enabled=False,
    # Header injection requires endpoints that return a Response object;
    # ours return dicts (FastAPI builds the response), so header injection
    # must stay off.
    headers_enabled=False,
)


def dynamic_limit_value() -> str:
    """Return the per-minute budget as a rate-limit string."""
    return f"{get_active_settings().rate_limit_per_min}/minute"
