"""Single-page-app static hosting for the built frontend.

Serves ``frontend/dist`` from the API process (single container topology).
On a 404 the built ``index.html`` is returned instead, so client-side
React Router paths (``/samples``, ``/compare``, ``/about``) work without
a separate web server.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    """StaticFiles with an index.html fallback for unknown SPA paths."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise
