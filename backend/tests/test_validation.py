"""Unit tests for upload validation (T3.5).

Covers the security-critical upload pipeline: size caps, magic-byte
whitelisting (extension never trusted), double-decode integrity checks,
decompression-bomb protection, EXIF transposition, and the guarantee that
validation performs zero disk writes.
"""

from __future__ import annotations

import io
import tempfile

import pytest
from PIL import Image

from app.core import validation
from app.core.errors import (
    PayloadTooLargeError,
    UnprocessableImageError,
    UnsupportedMediaTypeError,
)

pytestmark = pytest.mark.unit

ONE_MB = 1024 * 1024


def _fixture_bytes(fixtures_dir, name: str) -> bytes:
    return (fixtures_dir / name).read_bytes()


def _png_bytes(width: int = 20, height: int = 20, mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), (120, 60, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_accepts_valid_jpeg(fixtures_dir):
    result = validation.validate_upload(_fixture_bytes(fixtures_dir, "valid_jpeg.jpg"), ONE_MB)

    assert result.format == "jpeg"
    assert result.size == (40, 30)
    assert result.image.mode == "RGB"
    assert result.byte_count == 1010


def test_accepts_valid_png(fixtures_dir):
    result = validation.validate_upload(_fixture_bytes(fixtures_dir, "valid_png.png"), ONE_MB)

    assert result.format == "png"
    assert result.size == (24, 16)


def test_grayscale_image_is_converted_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (20, 20), 128).save(buffer, format="PNG")
    result = validation.validate_upload(buffer.getvalue(), ONE_MB)
    assert result.image.mode == "RGB"


# ---------------------------------------------------------------------------
# 415: magic-byte whitelist (extension is never trusted)
# ---------------------------------------------------------------------------


def test_rejects_text_file_with_jpeg_extension_415(fixtures_dir):
    with pytest.raises(UnsupportedMediaTypeError) as excinfo:
        validation.validate_upload(_fixture_bytes(fixtures_dir, "fake_image.jpg"), ONE_MB)
    assert excinfo.value.status_code == 415


def test_rejects_empty_upload_415():
    with pytest.raises(UnsupportedMediaTypeError) as excinfo:
        validation.validate_upload(b"", ONE_MB)
    assert excinfo.value.status_code == 415


def test_rejects_gif_magic_415():
    with pytest.raises(UnsupportedMediaTypeError):
        validation.validate_upload(b"GIF89a" + b"\x00" * 64, ONE_MB)


# ---------------------------------------------------------------------------
# 413: size cap (checked before magic so oversized junk is rejected fast)
# ---------------------------------------------------------------------------


def test_rejects_5mb_plus_one_byte_413():
    data = b"\x00" * (5 * ONE_MB + 1)
    with pytest.raises(PayloadTooLargeError) as excinfo:
        validation.validate_upload(data, max_bytes=5 * ONE_MB)
    assert excinfo.value.status_code == 413


def test_size_check_precedes_magic_check():
    # Oversized junk whose first bytes are valid JPEG magic: the size cap must
    # win (DoS protection runs before any content parsing).
    data = b"\xff\xd8\xff" + b"\x00" * (5 * ONE_MB)
    with pytest.raises(PayloadTooLargeError):
        validation.validate_upload(data, max_bytes=5 * ONE_MB)


def test_exactly_max_bytes_is_not_413():
    # Boundary: a file exactly at the cap passes the size check and is then
    # rejected for its content (magic ok, decode fails) — not for its size.
    data = b"\xff\xd8\xff" + b"\x00" * (ONE_MB - 3)
    with pytest.raises(UnprocessableImageError):
        validation.validate_upload(data, max_bytes=ONE_MB)


# ---------------------------------------------------------------------------
# 422: decode failures, truncation, pixel limits
# ---------------------------------------------------------------------------


def test_rejects_truncated_jpeg_422(fixtures_dir):
    with pytest.raises(UnprocessableImageError) as excinfo:
        validation.validate_upload(_fixture_bytes(fixtures_dir, "truncated.jpg"), ONE_MB)
    assert excinfo.value.status_code == 422


def test_rejects_truncated_png_422(fixtures_dir):
    raw = _fixture_bytes(fixtures_dir, "valid_png.png")
    with pytest.raises(UnprocessableImageError):
        validation.validate_upload(raw[: len(raw) // 2], ONE_MB)


def test_rejects_jpeg_magic_with_garbage_body_422():
    with pytest.raises(UnprocessableImageError):
        validation.validate_upload(b"\xff\xd8\xff" + b"definitely not jpeg data" * 4, ONE_MB)


def test_rejects_above_pixel_limit_422(monkeypatch):
    # Monkeypatching the module-level limit keeps the test fast (constructing
    # a real 4097x4096 image would be slow and memory-heavy).
    monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(UnprocessableImageError) as excinfo:
        validation.validate_upload(_png_bytes(20, 20), ONE_MB)  # 400 px > 100
    assert excinfo.value.status_code == 422


def test_pillow_decompression_bomb_guard_maps_to_422(monkeypatch):
    # Pillow's own MAX_IMAGE_PIXELS guard raises DecompressionBombError at
    # open() time; it must surface as 422, not 500.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(UnprocessableImageError) as excinfo:
        validation.validate_upload(_png_bytes(200, 200), ONE_MB)
    assert excinfo.value.status_code == 422


def test_max_image_pixels_constant_covers_4096_square():
    assert validation.MAX_IMAGE_PIXELS == 4096 * 4096


def test_pixel_budget_is_checked_before_full_decode(monkeypatch):
    # The pixel budget must be enforced on the image header (second open)
    # BEFORE image.load() decodes the full pixel buffer: verify() alone is
    # the only decode performed when the budget rejects the image.
    from PIL import ImageFile

    png = _png_bytes(20, 20)  # 400 px
    monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 100)

    loads: list[tuple[int, int]] = []
    original_load = ImageFile.ImageFile.load

    def spy_load(self):
        loads.append(self.size)
        return original_load(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", spy_load)
    with pytest.raises(UnprocessableImageError) as excinfo:
        validation.validate_upload(png, ONE_MB)

    assert "pixel" in str(excinfo.value)
    assert loads == [], "the budget check must pre-empt every full pixel decode"


# ---------------------------------------------------------------------------
# EXIF transposition
# ---------------------------------------------------------------------------


def test_exif_orientation_is_transposed(fixtures_dir):
    # Raw canvas is 80x60 with orientation=6; the correctly displayed image
    # (PIL transposes with ROTATE_270, i.e. 90 degrees CCW) must be 60x80.
    result = validation.validate_upload(_fixture_bytes(fixtures_dir, "exif_rotate6.jpg"), ONE_MB)

    assert result.size == (60, 80)
    assert result.format == "jpeg"


def test_exif_transposition_applies_rotation_to_pixels(fixtures_dir):
    # Raw canvas: left half green (x < 40), right half red. After the 90-CW
    # display transform the top-left column is the raw top row: green near the
    # top (raw x = y' < 40) and red lower down (raw x = y' >= 40). JPEG
    # artifacts allow a small per-channel tolerance.
    result = validation.validate_upload(_fixture_bytes(fixtures_dir, "exif_rotate6.jpg"), ONE_MB)

    top = result.image.getpixel((0, 0))
    lower = result.image.getpixel((0, 50))
    assert top[1] > 200 and top[0] < 80, f"expected green, got {top}"
    assert lower[0] > 200 and lower[1] < 80, f"expected red, got {lower}"


# ---------------------------------------------------------------------------
# Zero disk writes
# ---------------------------------------------------------------------------


def test_validation_writes_no_files(monkeypatch, tmp_path, fixtures_dir):
    # Redirect the OS temp dir to an empty directory: if validation touched
    # the filesystem, files would appear here.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    validation.validate_upload(_fixture_bytes(fixtures_dir, "valid_jpeg.jpg"), ONE_MB)
    validation.validate_upload(_fixture_bytes(fixtures_dir, "valid_png.png"), ONE_MB)

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------


def test_read_upload_stream_collects_chunks():
    payload = b"abc" * 100
    assert validation.read_upload_stream(io.BytesIO(payload), max_bytes=1000) == payload


def test_read_upload_stream_empty_yields_empty_bytes():
    assert validation.read_upload_stream(io.BytesIO(b""), max_bytes=ONE_MB) == b""


def test_read_upload_stream_stops_at_cap():
    with pytest.raises(PayloadTooLargeError):
        validation.read_upload_stream(io.BytesIO(b"x" * 1000), max_bytes=100)


def test_read_upload_stream_accepts_exact_cap():
    assert validation.read_upload_stream(io.BytesIO(b"x" * 100), max_bytes=100) == b"x" * 100
