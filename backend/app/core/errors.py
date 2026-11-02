"""Business exceptions and their HTTP semantics for the DefectLens API.

Every exception raised by services/validation carries a stable ``code``
string (machine-readable, shown to clients) and a client-safe ``message``.
Details that could leak internals go to the server log instead (see the
global exception handler in ``app.main``).
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, client-visible API errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message()
        super().__init__(self.message)

    @classmethod
    def default_message(cls) -> str:
        return "request could not be processed"


class UnsupportedMediaTypeError(AppError):
    """415: upload content is not a whitelisted image format (JPEG/PNG)."""

    status_code = 415
    code = "unsupported_media_type"

    @classmethod
    def default_message(cls) -> str:
        return "only JPEG and PNG images are accepted"


class PayloadTooLargeError(AppError):
    """413: upload exceeds the configured size cap."""

    status_code = 413
    code = "payload_too_large"

    @classmethod
    def default_message(cls) -> str:
        return "upload exceeds the maximum allowed size"


class UnprocessableImageError(AppError):
    """422: image is corrupted, truncated, or exceeds pixel limits."""

    status_code = 422
    code = "unprocessable_image"

    @classmethod
    def default_message(cls) -> str:
        return "image could not be decoded; the file may be corrupted or truncated"


class InvalidRequestError(AppError):
    """422: request parameters failed schema validation."""

    status_code = 422
    code = "invalid_request"

    @classmethod
    def default_message(cls) -> str:
        return "invalid request parameters"


class ModelNotFoundError(AppError):
    """404: the requested model id is not in the whitelist."""

    status_code = 404
    code = "model_not_found"

    @classmethod
    def default_message(cls) -> str:
        return "unknown model id"


class SampleNotFoundError(AppError):
    """404: the requested sample id is not in the server-side whitelist."""

    status_code = 404
    code = "sample_not_found"

    @classmethod
    def default_message(cls) -> str:
        return "unknown sample id"


class ModelNotReadyError(AppError):
    """503: a known model exists but its artifacts are not loaded/serving."""

    status_code = 503
    code = "model_not_ready"

    @classmethod
    def default_message(cls) -> str:
        return "the requested model is not available yet"


class InferenceTimeoutError(AppError):
    """503: inference exceeded the configured request timeout."""

    status_code = 503
    code = "inference_timeout"

    @classmethod
    def default_message(cls) -> str:
        return "inference timed out"
