"""Tests for single-container SPA static hosting (T4.4).

When ``frontend/dist`` exists it is served at "/" with an index.html
fallback for client-side routes; API routes always win; without a dist
directory the app runs API-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

INDEX_HTML = "<!doctype html><html><body>DEFECTLENS_MARKER</body></html>"


@pytest.fixture()
def spa_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a synthetic frontend build."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dist / "assets" / "app.js").write_text("// asset marker", encoding="utf-8")
    return Settings.from_env(frontend_dist_dir=dist)


class TestSpaStatic:
    def test_root_serves_index(self, make_client, spa_settings) -> None:
        with make_client(spa_settings) as test_client:
            response = test_client.get("/")
            assert response.status_code == 200
            assert "DEFECTLENS_MARKER" in response.text

    def test_spa_route_falls_back_to_index(self, make_client, spa_settings) -> None:
        with make_client(spa_settings) as test_client:
            response = test_client.get("/samples")
            assert response.status_code == 200
            assert "DEFECTLENS_MARKER" in response.text

    def test_assets_are_served(self, make_client, spa_settings) -> None:
        with make_client(spa_settings) as test_client:
            response = test_client.get("/assets/app.js")
            assert response.status_code == 200
            assert "asset marker" in response.text

    def test_api_routes_still_win(self, make_client, spa_settings) -> None:
        with make_client(spa_settings) as test_client:
            response = test_client.get("/api/v1/health")
            assert response.status_code == 200
            assert "models_loaded" in response.json()

    def test_missing_dist_means_api_only(self, make_client, tmp_path: Path) -> None:
        settings = Settings.from_env(frontend_dist_dir=tmp_path / "does-not-exist")
        with make_client(settings) as test_client:
            response = test_client.get("/")
            assert response.status_code == 404
            assert response.json()["success"] is False
