"""Container ASGI entrypoint tests (T5.1)."""

from app.asgi import app


def test_container_entrypoint_builds_the_fastapi_app():
    """The gunicorn entrypoint assembles the same app as the factory."""
    assert app.title == "DefectLens Steel Defect API"
    assert app.state.settings is not None
    paths = app.openapi()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/models" in paths
