"""Tests for the test-set evaluation script (T2.4).

The pure helpers in ``scripts/evaluate.py`` (``_per_class_metrics``,
``_update_metrics_json``, ``_write_metrics_json``, ``_read_carbon_csv``,
``_extract_training_duration``, ``_iso_timestamp``, ``_load_checkpoint``,
``_read_metrics_json``) are unit-tested directly; ``main`` is covered by
smoke tests with monkeypatched data loaders and tmp paths. No test depends
on the real ``artifacts/checkpoints`` contents, on the background training
run, or on GPU training.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from scripts import evaluate
from steel_training.models.factory import create_resnet

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CARBON_HEADER = "timestamp,project_name,duration,emissions,energy_consumed"


@pytest.fixture(scope="session")
def tiny_checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Trainer-format checkpoint holding a resnet18 (fc=6) state dict."""
    model = create_resnet("resnet18", num_classes=6)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 3,
        "metrics": {
            "train_loss": 0.8,
            "train_accuracy": 0.9,
            "val_loss": 0.7,
            "val_accuracy": 0.85,
            "val_macro_f1": 0.72,
        },
        "config": {"model_name": "resnet18", "epochs": 20},
    }
    path = tmp_path_factory.mktemp("checkpoints") / "resnet18_best.pth"
    torch.save(checkpoint, path)
    return path


def _write_carbon_csv(path: Path, header: str, rows: list[str]) -> Path:
    """Write a CodeCarbon-style CSV; returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_minimal_manifest(path: Path) -> Path:
    """Write a manifest with the 6 real class names and no image entries."""
    manifest = {
        "seed": 42,
        "class_names": CLASS_NAMES,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "entries": [],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _fake_test_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One batch of four random 224x224 images with labels 0..3."""
    torch.manual_seed(0)
    return [(torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 3]))]


# ---------------------------------------------------------------------------
# _per_class_metrics
# ---------------------------------------------------------------------------


def test_per_class_metrics_perfect_predictions() -> None:
    # Arrange
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5])

    # Act
    result = evaluate._per_class_metrics(y_true, y_pred, num_classes=6)

    # Assert
    assert len(result) == 6
    for row in result:
        assert row["precision"] == 1.0
        assert row["recall"] == 1.0
        assert row["f1"] == 1.0
    assert [row["support"] for row in result] == [1, 1, 1, 1, 1, 1]


def test_per_class_metrics_known_values() -> None:
    # Arrange: class 0 -> tp=1 fp=0 fn=1 (P=1, R=0.5, F1=2/3);
    #          class 1 -> tp=1 fp=1 fn=0 (P=0.5, R=1, F1=2/3).
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]

    # Act
    result = evaluate._per_class_metrics(y_true, y_pred, num_classes=2)

    # Assert
    class0, class1 = result
    assert class0["precision"] == pytest.approx(1.0)
    assert class0["recall"] == pytest.approx(0.5)
    assert class0["f1"] == pytest.approx(2 / 3)
    assert class0["support"] == 2
    assert class1["precision"] == pytest.approx(0.5)
    assert class1["recall"] == pytest.approx(1.0)
    assert class1["f1"] == pytest.approx(2 / 3)
    assert class1["support"] == 1


def test_per_class_metrics_zero_sample_class_f1_is_zero_not_nan() -> None:
    # Arrange: classes 2..5 have no true and no predicted samples.
    y_true = [0, 1]
    y_pred = [0, 1]

    # Act
    result = evaluate._per_class_metrics(y_true, y_pred, num_classes=6)

    # Assert: documented convention - unseen class contributes 0.0, not NaN.
    for index in (2, 3, 4, 5):
        row = result[index]
        assert row["precision"] == 0.0
        assert row["recall"] == 0.0
        assert row["f1"] == 0.0
        assert row["support"] == 0


def test_per_class_metrics_class_with_only_false_positives() -> None:
    # Arrange: class 1 is predicted once but never true (tp=0, fp=1, fn=0).
    y_true = [0]
    y_pred = [1]

    # Act
    result = evaluate._per_class_metrics(y_true, y_pred, num_classes=2)

    # Assert: no true positives -> recall 0 (convention) -> F1 0.
    assert result[1]["precision"] == 0.0
    assert result[1]["recall"] == 0.0
    assert result[1]["f1"] == 0.0
    assert result[1]["support"] == 0


def test_per_class_metrics_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate._per_class_metrics([], [], num_classes=3)


def test_per_class_metrics_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="length"):
        evaluate._per_class_metrics([0, 1], [0], num_classes=2)


def test_per_class_metrics_out_of_range_label_raises() -> None:
    with pytest.raises(ValueError, match="range"):
        evaluate._per_class_metrics([0, 3], [0, 3], num_classes=3)


def test_per_class_metrics_non_integer_input_raises() -> None:
    with pytest.raises(TypeError, match="integer"):
        evaluate._per_class_metrics([0.0, 1.0], [0.0, 1.0], num_classes=2)


# ---------------------------------------------------------------------------
# _update_metrics_json
# ---------------------------------------------------------------------------


def test_update_metrics_json_from_scratch() -> None:
    # Arrange / Act
    result = evaluate._update_metrics_json(None, "resnet18", {"params": 1})

    # Assert
    assert result == {"models": {"resnet18": {"params": 1}}}


def test_update_metrics_json_preserves_other_models() -> None:
    # Arrange
    existing = {"models": {"resnet50": {"params": 2, "test_acc": 0.9}}}

    # Act
    result = evaluate._update_metrics_json(existing, "resnet18", {"params": 1})

    # Assert
    assert result["models"]["resnet50"] == {"params": 2, "test_acc": 0.9}
    assert result["models"]["resnet18"] == {"params": 1}
    # The input document is not mutated.
    assert existing == {"models": {"resnet50": {"params": 2, "test_acc": 0.9}}}
    assert result["models"] is not existing["models"]


def test_update_metrics_json_replaces_existing_entry_wholesale() -> None:
    # Arrange
    existing = {"models": {"resnet18": {"params": 1, "test_acc": 0.5}}}

    # Act
    result = evaluate._update_metrics_json(existing, "resnet18", {"params": 3})

    # Assert: stale fields of the replaced entry do not survive.
    assert result == {"models": {"resnet18": {"params": 3}}}


def test_update_metrics_json_preserves_other_top_level_keys() -> None:
    # Arrange
    existing = {"schema_version": 1, "models": {}}

    # Act
    result = evaluate._update_metrics_json(existing, "resnet18", {"params": 1})

    # Assert
    assert result["schema_version"] == 1
    assert result["models"] == {"resnet18": {"params": 1}}


# ---------------------------------------------------------------------------
# _write_metrics_json (atomic write)
# ---------------------------------------------------------------------------


def test_write_metrics_json_creates_file_and_no_tmp_residue(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "nested" / "metrics.json"
    data = {"models": {"resnet18": {"params": 1, "test_acc": 0.9}}}

    # Act
    evaluate._write_metrics_json(path, data)

    # Assert
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert list(path.parent.glob("*.tmp")) == []


def test_write_metrics_json_overwrites_existing_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "metrics.json"
    evaluate._write_metrics_json(path, {"models": {"resnet18": {"params": 1}}})

    # Act
    evaluate._write_metrics_json(path, {"models": {"resnet18": {"params": 2}}})

    # Assert
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "models": {"resnet18": {"params": 2}}
    }
    assert list(path.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# _read_carbon_csv
# ---------------------------------------------------------------------------


def test_read_carbon_csv_missing_file_returns_none(tmp_path: Path) -> None:
    # Arrange / Act
    result = evaluate._read_carbon_csv(tmp_path / "B_defectlens_resnet18_emissions.csv")

    # Assert
    assert result is None


def test_read_carbon_csv_single_row_converts_kg_to_g(tmp_path: Path) -> None:
    # Arrange
    path = _write_carbon_csv(
        tmp_path / "B_defectlens_resnet18_emissions.csv",
        CARBON_HEADER,
        ["2026-11-02T00:40:24,macef_run1,20.0,0.0001335592288394,0.0005"],
    )

    # Act
    result = evaluate._read_carbon_csv(path)

    # Assert: 0.0001335592288394 kg -> 0.1335592288394 g.
    assert result == pytest.approx(0.1335592288394)


def test_read_carbon_csv_multiple_rows_takes_last_row(tmp_path: Path) -> None:
    # Arrange: CodeCarbon appends one row per tracker session; the last row
    # belongs to the session that produced the evaluated checkpoint.
    path = _write_carbon_csv(
        tmp_path / "B_defectlens_resnet18_emissions.csv",
        CARBON_HEADER,
        [
            "2026-11-02T00:40:24,macef_run1,20.0,0.0001335592288394,0.0005",
            "2026-11-02T00:43:45,macef_run2,33.48,0.00030854184932307483,0.0013",
        ],
    )

    # Act
    result = evaluate._read_carbon_csv(path)

    # Assert: the last row's emissions, converted to grams.
    assert result == pytest.approx(0.30854184932307483)


def test_read_carbon_csv_without_emissions_column_returns_none(tmp_path: Path) -> None:
    # Arrange
    path = _write_carbon_csv(
        tmp_path / "odd.csv", "timestamp,project_name,duration", ["2026-11-02T00:40:24,run1,20.0"]
    )

    # Act / Assert
    assert evaluate._read_carbon_csv(path) is None


def test_read_carbon_csv_header_only_returns_none(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    path = _write_carbon_csv(tmp_path / "empty.csv", CARBON_HEADER, [])
    assert evaluate._read_carbon_csv(path) is None


def test_read_carbon_csv_non_numeric_emissions_returns_none(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    path = _write_carbon_csv(
        tmp_path / "bad.csv", CARBON_HEADER, ["2026-11-02T00:40:24,run1,20.0,n/a,0.0"]
    )
    assert evaluate._read_carbon_csv(path) is None


# ---------------------------------------------------------------------------
# _extract_training_duration
# ---------------------------------------------------------------------------


def test_extract_training_duration_from_metrics_duration_key() -> None:
    # Arrange / Act
    result = evaluate._extract_training_duration({"metrics": {"duration": 123.4}})

    # Assert
    assert result == pytest.approx(123.4)


def test_extract_training_duration_from_training_duration_s_key() -> None:
    # Arrange / Act
    result = evaluate._extract_training_duration({"metrics": {"training_duration_s": 5}})

    # Assert
    assert result == 5.0


def test_extract_training_duration_absent_returns_none() -> None:
    # Arrange / Act / Assert
    assert evaluate._extract_training_duration({}) is None
    assert evaluate._extract_training_duration({"metrics": {"val_macro_f1": 0.7}}) is None


def test_extract_training_duration_non_numeric_ignored() -> None:
    # Arrange / Act / Assert
    assert evaluate._extract_training_duration({"metrics": {"duration": "fast"}}) is None


# ---------------------------------------------------------------------------
# _iso_timestamp
# ---------------------------------------------------------------------------


def test_iso_timestamp_epoch_zero() -> None:
    # Arrange / Act
    result = evaluate._iso_timestamp(0.0)

    # Assert
    assert result == "1970-01-01T00:00:00+00:00"
    datetime.fromisoformat(result)


# ---------------------------------------------------------------------------
# _load_checkpoint
# ---------------------------------------------------------------------------


def test_load_checkpoint_applies_state_dict(tiny_checkpoint_path: Path) -> None:
    # Arrange
    model = create_resnet("resnet18", num_classes=6)

    # Act
    checkpoint = evaluate._load_checkpoint(tiny_checkpoint_path, model)

    # Assert
    expected = torch.load(
        str(tiny_checkpoint_path), map_location="cpu", weights_only=True
    )["model_state_dict"]
    loaded = model.state_dict()
    assert loaded.keys() == expected.keys()
    for name in expected:
        assert torch.equal(loaded[name], expected[name])
    assert checkpoint["epoch"] == 3


def test_load_checkpoint_missing_file_raises(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        evaluate._load_checkpoint(tmp_path / "nope.pth", nn.Linear(2, 2))


def test_load_checkpoint_not_a_dict_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "bad.pth"
    torch.save([1, 2, 3], path)

    # Act / Assert
    with pytest.raises(ValueError, match="model_state_dict"):
        evaluate._load_checkpoint(path, nn.Linear(2, 2))


def test_load_checkpoint_missing_state_dict_key_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "bad.pth"
    torch.save({"epoch": 1}, path)

    # Act / Assert
    with pytest.raises(ValueError, match="model_state_dict"):
        evaluate._load_checkpoint(path, nn.Linear(2, 2))


def test_load_checkpoint_wrong_architecture_raises(tmp_path: Path) -> None:
    # Arrange: a state dict whose shapes do not match the target model.
    path = tmp_path / "wrong.pth"
    torch.save({"model_state_dict": nn.Linear(3, 3).state_dict()}, path)

    # Act / Assert
    with pytest.raises(ValueError, match="model_state_dict"):
        evaluate._load_checkpoint(path, nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# _read_metrics_json
# ---------------------------------------------------------------------------


def test_read_metrics_json_missing_file_returns_empty(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    assert evaluate._read_metrics_json(tmp_path / "metrics.json") == {}


def test_read_metrics_json_valid_document(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "metrics.json"
    path.write_text('{"models": {"resnet50": {"params": 2}}}', encoding="utf-8")

    # Act / Assert
    assert evaluate._read_metrics_json(path) == {"models": {"resnet50": {"params": 2}}}


def test_read_metrics_json_corrupt_json_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="not valid JSON"):
        evaluate._read_metrics_json(path)


def test_read_metrics_json_unexpected_structure_raises(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "metrics.json"
    path.write_text('{"models": [1, 2]}', encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="models"):
        evaluate._read_metrics_json(path)


# ---------------------------------------------------------------------------
# _build_test_loader (integration with SteelDefectDataset on tiny images)
# ---------------------------------------------------------------------------


def test_build_test_loader_reads_manifest_test_split(tmp_path: Path) -> None:
    # Arrange: three tiny real images and a manifest pointing at them.
    image_root = tmp_path / "images"
    for class_name in CLASS_NAMES[:2]:
        (image_root / class_name).mkdir(parents=True)
    for index in range(3):
        image = Image.new("RGB", (32, 32), color=(10 * index, 20 * index, 30 * index))
        image.save(image_root / CLASS_NAMES[index % 2] / f"img_{index}.jpg")
    manifest = {
        "seed": 42,
        "class_names": CLASS_NAMES,
        "split_counts": {"train": 0, "val": 0, "test": 3},
        "entries": [
            {"path": "crazing/img_0.jpg", "label": "crazing", "split": "test"},
            {"path": "inclusion/img_1.jpg", "label": "inclusion", "split": "test"},
            {"path": "crazing/img_2.jpg", "label": "crazing", "split": "test"},
        ],
    }

    # Act
    loader = evaluate._build_test_loader(manifest, image_root, num_workers=0)

    # Assert: eval transform applied (224x224, normalized), labels indexed.
    assert len(loader.dataset) == 3
    images, labels = next(iter(loader))
    assert images.shape == (3, 3, 224, 224)
    assert labels.tolist() == [0, 1, 0]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults_derive_from_model() -> None:
    # Arrange / Act
    args = evaluate.parse_args(["--model", "resnet18"])

    # Assert
    assert args.model == "resnet18"
    assert args.config == evaluate.CONFIG_DIR / "resnet18.yaml"
    assert args.checkpoint == evaluate.CHECKPOINT_DIR / "resnet18_best.pth"


def test_parse_args_explicit_paths() -> None:
    # Arrange / Act
    args = evaluate.parse_args(
        ["--model", "resnet50", "--config", "x.yaml", "--checkpoint", "y.pth"]
    )

    # Assert
    assert args.model == "resnet50"
    assert args.config == Path("x.yaml")
    assert args.checkpoint == Path("y.pth")


def test_parse_args_rejects_unknown_model() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--model", "resnet101"])


# ---------------------------------------------------------------------------
# main smoke tests (monkeypatched loaders and paths; no real dataset/GPU)
# ---------------------------------------------------------------------------


def test_main_writes_metrics_json_smoke(
    tmp_path: Path, tiny_checkpoint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: hermetic paths + a fake loader with one batch of four samples.
    monkeypatch.setattr(
        evaluate, "MANIFEST_PATH", _write_minimal_manifest(tmp_path / "manifest.json")
    )
    monkeypatch.setattr(evaluate, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(evaluate, "CARBON_DIR", tmp_path / "carbon")
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(evaluate, "_build_test_loader", lambda *args, **kwargs: _fake_test_loader())

    # Act
    exit_code = evaluate.main(["--model", "resnet18", "--checkpoint", str(tiny_checkpoint_path)])

    # Assert
    assert exit_code == 0
    doc = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    entry = doc["models"]["resnet18"]
    assert entry["params"] > 0
    assert isinstance(entry["test_acc"], float)
    assert isinstance(entry["test_macro_f1"], float)
    assert list(entry["per_class_f1"].keys()) == CLASS_NAMES
    assert all(isinstance(value, float) for value in entry["per_class_f1"].values())
    assert len(entry["confusion_matrix"]) == 6
    assert all(len(row) == 6 for row in entry["confusion_matrix"])
    assert isinstance(entry["confusion_matrix"][0][0], int)
    assert entry["training_epochs"] == 3
    assert entry["best_val_macro_f1"] == pytest.approx(0.72)
    assert entry["carbon_co2eq_g"] is None  # no carbon CSV under the fake dir
    assert entry["training_duration_s"] is None
    datetime.fromisoformat(entry["trained_at"])  # ISO-8601 parseable
    assert list(tmp_path.glob("*.tmp")) == []


def test_main_preserves_existing_model_entries(
    tmp_path: Path, tiny_checkpoint_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: an existing metrics doc with a resnet50 entry must survive.
    (tmp_path / "metrics.json").write_text(
        json.dumps({"models": {"resnet50": {"params": 25_557_032}}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        evaluate, "MANIFEST_PATH", _write_minimal_manifest(tmp_path / "manifest.json")
    )
    monkeypatch.setattr(evaluate, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(evaluate, "CARBON_DIR", tmp_path / "carbon")
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(evaluate, "_build_test_loader", lambda *args, **kwargs: _fake_test_loader())

    # Act
    exit_code = evaluate.main(["--model", "resnet18", "--checkpoint", str(tiny_checkpoint_path)])

    # Assert
    assert exit_code == 0
    doc = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert doc["models"]["resnet50"] == {"params": 25_557_032}
    assert "resnet18" in doc["models"]


def test_main_empty_test_set_exits_nonzero_without_writing(
    tmp_path: Path, tiny_checkpoint_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange: the loader yields no batches at all.
    monkeypatch.setattr(
        evaluate, "MANIFEST_PATH", _write_minimal_manifest(tmp_path / "manifest.json")
    )
    monkeypatch.setattr(evaluate, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(evaluate, "CARBON_DIR", tmp_path / "carbon")
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(evaluate, "_build_test_loader", lambda *args, **kwargs: [])

    # Act
    exit_code = evaluate.main(["--model", "resnet18", "--checkpoint", str(tiny_checkpoint_path)])

    # Assert
    assert exit_code == 1
    assert "no samples" in caplog.text
    assert not (tmp_path / "metrics.json").exists()


def test_main_missing_checkpoint_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setattr(
        evaluate, "MANIFEST_PATH", _write_minimal_manifest(tmp_path / "manifest.json")
    )
    monkeypatch.setattr(evaluate, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(evaluate, "CARBON_DIR", tmp_path / "carbon")
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path / "data")

    # Act
    exit_code = evaluate.main(
        ["--model", "resnet18", "--checkpoint", str(tmp_path / "nope.pth")]
    )

    # Assert
    assert exit_code == 1
    assert "checkpoint does not exist" in caplog.text
    assert not (tmp_path / "metrics.json").exists()
