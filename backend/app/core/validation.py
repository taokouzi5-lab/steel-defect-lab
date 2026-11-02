"""Upload validation — the security core of the API (T3.5).

Checks, in order (cheapest first):

1. size cap (413) — streamed, stops reading past the cap;
2. magic-byte whitelist (415) — JPEG ``FF D8 FF`` / PNG ``89 50 4E 47``;
   the file extension is never trusted;
3. Pillow ``verify()`` + reopen + pixel-budget check *before* the full
   ``load()`` decode (422) — catches truncated and fake files and rejects
   oversized images without paying the decode cost;
   ``Image.DecompressionBombError`` also maps to 422 (Pillow's own guard is
   lowered to 4096x4096 at import);
4. EXIF transposition (orientation tag applied, not ignored);
5. conversion to RGB.

The application-level pipeline is memory-only: this code never writes
files. One layer up, Starlette's multipart parser may spool request bodies
larger than ~1 MB into a ``SpooledTemporaryFile`` on disk before this code
runs; the HTTP layer's Content-Length precheck (``app.core.body_limit``,
wired in ``app.main``) rejects oversized uploads before that parsing starts.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, ImageOps

from app.core.errors import (
    PayloadTooLargeError,
    UnprocessableImageError,
    UnsupportedMediaTypeError,
)

logger = logging.getLogger(__name__)

# Pixel budget: 4096x4096. Also lowers Pillow's internal decompression-bomb
# guard for this process (Pillow's default is ~178M pixels, far too generous).
MAX_IMAGE_PIXELS = 4096 * 4096
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG"

_STREAM_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class ValidatedImage:
    """A decoded, EXIF-corrected, RGB image plus provenance facts."""

    image: Image.Image
    format: str
    size: tuple[int, int]
    byte_count: int


def read_upload_stream(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read an upload stream into memory, failing fast past ``max_bytes``.

    Stops reading as soon as the cap is exceeded so oversized uploads are
    rejected before their full body is buffered.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(_STREAM_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(
                f"upload exceeds the {max_bytes}-byte cap (received at least {total} bytes)"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _detect_format(data: bytes) -> str:
    if data.startswith(_JPEG_MAGIC):
        return "jpeg"
    if data.startswith(_PNG_MAGIC):
        return "png"
    raise UnsupportedMediaTypeError("file content does not start with a JPEG or PNG signature")


def _decode(data: bytes, image_format: str, budget: int) -> Image.Image:
    """verify() + reopen + pixel-budget check before the full decode.

    The budget is enforced on the image header as soon as the second open
    succeeds and *before* ``load()`` decodes the pixel buffer, so oversized
    images are rejected without paying the decode cost.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # integrity check without full pixel decode
    except Image.DecompressionBombError as exc:
        raise UnprocessableImageError("image exceeds the decompression-bomb pixel limit") from exc
    except Exception as exc:  # UnidentifiedImageError/OSError/SyntaxError...
        logger.info("verify() rejected a %s upload: %s", image_format, exc)
        raise UnprocessableImageError("file fails image integrity verification") from exc

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > budget:
            raise UnprocessableImageError(
                f"image is {width}x{height} ({width * height} px), exceeding the "
                f"{budget}-pixel limit"
            )
        image.load()  # force the full decode so truncation cannot hide
        return image
    except UnprocessableImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise UnprocessableImageError("image exceeds the decompression-bomb pixel limit") from exc
    except Exception as exc:
        logger.info("full decode rejected a %s upload: %s", image_format, exc)
        raise UnprocessableImageError("image could not be fully decoded") from exc


def validate_upload(
    data: bytes,
    max_bytes: int,
    max_pixels: int | None = None,
) -> ValidatedImage:
    """Validate an in-memory upload and return a decoded RGB image.

    Args:
        data: the raw upload bytes (already read from the stream).
        max_bytes: the size cap; exceeding it raises 413.
        max_pixels: pixel budget (defaults to :data:`MAX_IMAGE_PIXELS`);
            exceeding it raises 422, enforced before the full decode.

    Raises:
        PayloadTooLargeError: 413, upload too large.
        UnsupportedMediaTypeError: 415, magic bytes not JPEG/PNG.
        UnprocessableImageError: 422, decode failure or pixel budget exceeded.
    """
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"upload exceeds the {max_bytes}-byte cap ({len(data)} bytes received)"
        )

    image_format = _detect_format(data)
    budget = max_pixels if max_pixels is not None else MAX_IMAGE_PIXELS
    image = _decode(data, image_format, budget)

    image = ImageOps.exif_transpose(image)  # apply EXIF orientation
    return ValidatedImage(
        image=image.convert("RGB"),
        format=image_format,
        size=image.size,
        byte_count=len(data),
    )
