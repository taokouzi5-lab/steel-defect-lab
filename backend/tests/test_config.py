"""Settings environment parsing (T5.1 container deployment work)."""

from app.config import Settings

DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def test_cors_origins_unset_keeps_dev_defaults(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    settings = Settings.from_env()
    assert settings.cors_origins == DEV_ORIGINS


def test_cors_origins_empty_string_disables_cors(monkeypatch):
    # Deployment contract (docs/deployment.md): CORS_ORIGINS="" means no
    # browser CORS in the production container.
    monkeypatch.setenv("CORS_ORIGINS", "")
    settings = Settings.from_env()
    assert settings.cors_origins == ()


def test_cors_origins_whitespace_only_disables_cors(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "   ")
    settings = Settings.from_env()
    assert settings.cors_origins == ()


def test_cors_origins_parses_comma_separated_list(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    settings = Settings.from_env()
    assert settings.cors_origins == ("https://a.example", "https://b.example")
