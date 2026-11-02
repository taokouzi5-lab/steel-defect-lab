"""Unit tests for envelope, error mapping, and carbon disclosure (T3.6)."""

from __future__ import annotations

import pytest

from app.carbon import constants, meta as carbon_meta
from app.core import envelope
from app.core.errors import (
    AppError,
    InferenceTimeoutError,
    ModelNotFoundError,
    ModelNotReadyError,
    PayloadTooLargeError,
    SampleNotFoundError,
    UnprocessableImageError,
    UnsupportedMediaTypeError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_success_envelope_shape():
    body = envelope.success({"class_id": 0}, {"model": "resnet18-int8"})
    assert body == {
        "success": True,
        "data": {"class_id": 0},
        "error": None,
        "meta": {"model": "resnet18-int8"},
    }


def test_failure_envelope_shape():
    body = envelope.failure("payload_too_large", "upload exceeds 5 MB", 413, "abc123")
    assert body == {
        "success": False,
        "data": None,
        "error": {"code": "payload_too_large", "message": "upload exceeds 5 MB", "status": 413},
        "meta": {"request_id": "abc123"},
    }


def test_request_id_is_32_hex_chars():
    first, second = envelope.new_request_id(), envelope.new_request_id()
    assert len(first) == 32
    assert all(char in "0123456789abcdef" for char in first)
    assert first != second


# ---------------------------------------------------------------------------
# Error classes -> HTTP semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "status", "code"),
    [
        (UnsupportedMediaTypeError, 415, "unsupported_media_type"),
        (PayloadTooLargeError, 413, "payload_too_large"),
        (UnprocessableImageError, 422, "unprocessable_image"),
        (ModelNotFoundError, 404, "model_not_found"),
        (ModelNotReadyError, 503, "model_not_ready"),
        (InferenceTimeoutError, 503, "inference_timeout"),
        (SampleNotFoundError, 404, "sample_not_found"),
    ],
)
def test_error_status_code_semantics(error_class, status, code):
    error = error_class()
    assert error.status_code == status
    assert error.code == code
    assert isinstance(error.message, str) and error.message
    assert isinstance(error, AppError)


# ---------------------------------------------------------------------------
# Carbon method disclosure (GET /meta)
# ---------------------------------------------------------------------------


def test_carbon_meta_has_three_declarations():
    disclosure = carbon_meta.build_carbon_meta(country="China")

    assert set(disclosure) == {"formula", "constants", "boundaries", "methods"}
    assert "co2eq_g" in disclosure["formula"]
    assert (
        disclosure["constants"]["grid_carbon_intensity_g_per_kwh"]
        == constants.GRID_CARBON_INTENSITY["China"]
    )
    assert disclosure["constants"]["pue"] == constants.DEFAULT_PUE
    assert len(disclosure["boundaries"]) >= 3
    assert "mace-f-methodD-measured" in disclosure["methods"]
    assert "tier-representative-fallback" in disclosure["methods"]


def test_carbon_meta_respects_country():
    uk = carbon_meta.build_carbon_meta(country="UK")
    assert uk["constants"]["grid_carbon_intensity_g_per_kwh"] == 231.0


def test_carbon_meta_unknown_country_raises():
    with pytest.raises(ValueError):
        carbon_meta.build_carbon_meta(country="Atlantis")
