"""Unit tests for the sample-image catalog (T3.6).

The catalog is the server-side whitelist behind ``GET /samples`` and
``POST /predict/sample/{id}``: user input is only ever compared against
manifest entries, never joined into filesystem paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.sample_catalog import SampleCatalog

pytestmark = pytest.mark.unit


def _make_catalog(tmp_path: Path, samples: list[dict] | None = None) -> SampleCatalog:
    manifest = {
        "source": "unit test",
        "class_names": ["crazing", "scratches"],
        "samples": samples
        if samples is not None
        else [
            {"id": "crazing_101", "label": "crazing", "name": "crazing_101.jpg"},
            {"id": "scratches_126", "label": "scratches", "name": "scratches_126.jpg"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for entry in manifest["samples"]:
        try:
            (tmp_path / entry["label"]).mkdir(exist_ok=True)
            (tmp_path / entry["label"] / entry["name"]).write_bytes(b"placeholder")
        except OSError:
            # Unsafe names (separators, "..", empty) cannot be materialized —
            # exactly what the catalog must reject.
            continue
    return SampleCatalog(tmp_path)


def test_lists_manifest_entries(tmp_path):
    catalog = _make_catalog(tmp_path)
    infos = catalog.list()
    assert [(info.id, info.label) for info in infos] == [
        ("crazing_101", "crazing"),
        ("scratches_126", "scratches"),
    ]


def test_get_by_id(tmp_path):
    catalog = _make_catalog(tmp_path)
    info = catalog.get("scratches_126")
    assert info is not None
    assert info.name == "scratches_126.jpg"
    assert catalog.path_of(info) == tmp_path / "scratches" / "scratches_126.jpg"


def test_get_unknown_id_returns_none(tmp_path):
    catalog = _make_catalog(tmp_path)
    assert catalog.get("nope") is None
    assert catalog.get("") is None


def test_traversal_ids_are_simply_not_found(tmp_path):
    catalog = _make_catalog(tmp_path)
    for evil in ("../crazing_101", "..\\secret", "crazing_101/../../x", "C:\\windows\\win.ini"):
        assert catalog.get(evil) is None, evil


def test_manifest_entries_with_separators_are_rejected(tmp_path):
    catalog = _make_catalog(
        tmp_path,
        [
            {"id": "ok", "label": "crazing", "name": "ok.jpg"},
            {"id": "evil", "label": "crazing", "name": "../evil.jpg"},
            {"id": "evil2", "label": "crazing", "name": "a/b.jpg"},
            {"id": "evil3", "label": "..", "name": "x.jpg"},
            {"id": "empty", "label": "crazing", "name": ""},
        ],
    )
    assert [info.id for info in catalog.list()] == ["ok"]


def test_entries_whose_file_is_missing_are_skipped(tmp_path):
    # The file must be gone before the catalog loads the manifest.
    manifest = {
        "source": "unit test",
        "class_names": ["crazing", "scratches"],
        "samples": [
            {"id": "crazing_101", "label": "crazing", "name": "crazing_101.jpg"},
            {"id": "scratches_126", "label": "scratches", "name": "scratches_126.jpg"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "crazing").mkdir()
    (tmp_path / "crazing" / "crazing_101.jpg").write_bytes(b"placeholder")
    (tmp_path / "scratches").mkdir()
    (tmp_path / "scratches" / "scratches_126.jpg").write_bytes(b"placeholder")
    (tmp_path / "crazing" / "crazing_101.jpg").unlink()

    catalog = SampleCatalog(tmp_path)
    assert [info.id for info in catalog.list()] == ["scratches_126"]
    assert catalog.get("crazing_101") is None


def test_missing_manifest_yields_empty_catalog(tmp_path):
    catalog = SampleCatalog(tmp_path)
    assert catalog.list() == []
    assert catalog.get("anything") is None


def test_malformed_manifest_yields_empty_catalog(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    catalog = SampleCatalog(tmp_path)
    assert catalog.list() == []


def test_path_of_never_escapes_samples_dir(tmp_path):
    catalog = _make_catalog(tmp_path)
    for info in catalog.list():
        path = catalog.path_of(info)
        assert path.parent.parent == tmp_path
