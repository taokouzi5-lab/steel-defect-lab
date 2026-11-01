"""Tests for the ResNet model factory and parameter splitting (T2.3)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from steel_training.models.factory import create_resnet, split_params

# Total parameter counts for torchvision resnets with a 6-class fc head
# (ImageNet fc replaced): resnet18 11_689_512 - 513_000 + 3_078 = 11_179_590,
# resnet50 25_557_032 - 2_049_000 + 12_294 = 23_520_326.
RESNET18_TOTAL_RANGE = (11_000_000, 12_000_000)
RESNET50_TOTAL_RANGE = (23_000_000, 26_000_000)


def test_create_resnet18_structure() -> None:
    # Arrange / Act
    model = create_resnet("resnet18", num_classes=6)

    # Assert
    assert model.fc.out_features == 6
    total = sum(p.numel() for p in model.parameters())
    assert RESNET18_TOTAL_RANGE[0] < total < RESNET18_TOTAL_RANGE[1]


def test_create_resnet50_structure() -> None:
    # Arrange / Act
    model = create_resnet("resnet50", num_classes=6)

    # Assert
    assert model.fc.out_features == 6
    total = sum(p.numel() for p in model.parameters())
    assert RESNET50_TOTAL_RANGE[0] < total < RESNET50_TOTAL_RANGE[1]


def test_create_resnet_custom_num_classes() -> None:
    # Arrange / Act
    model = create_resnet("resnet18", num_classes=10)

    # Assert
    assert model.fc.out_features == 10


def test_create_resnet_unknown_name_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="unknown model"):
        create_resnet("resnet101")


def test_create_resnet_invalid_num_classes_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="num_classes"):
        create_resnet("resnet18", num_classes=0)


def test_pretrained_state_dict_loads_and_fc_is_fresh(
    pretrained_dir: Path,
) -> None:
    # Arrange
    weights = pretrained_dir / "resnet18.pth"
    if not weights.is_file():
        pytest.skip("local resnet18.pth cache is not present")

    # Act
    model = create_resnet("resnet18", num_classes=6, pretrained_path=weights)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)

    # Assert: backbone keys were copied verbatim (strict load succeeded) ...
    assert torch.equal(model.conv1.weight, state["conv1.weight"])
    assert torch.equal(model.layer4[1].conv2.weight, state["layer4.1.conv2.weight"])
    # ... while fc was replaced by a freshly (randomly) initialized 6-class head.
    assert model.fc.in_features == 512
    assert model.fc.out_features == 6
    # Kaiming-uniform scale for fan_in 512: bound 1/sqrt(512) ~= 0.044.
    assert 0.005 < model.fc.weight.std().item() < 0.25
    assert 0.005 < model.fc.bias.std().item() < 0.25


def test_pretrained_resnet50_loads(pretrained_dir: Path) -> None:
    # Arrange
    weights = pretrained_dir / "resnet50.pth"
    if not weights.is_file():
        pytest.skip("local resnet50.pth cache is not present")

    # Act
    model = create_resnet("resnet50", num_classes=6, pretrained_path=weights)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)

    # Assert
    assert torch.equal(model.conv1.weight, state["conv1.weight"])
    assert model.fc.in_features == 2048
    assert model.fc.out_features == 6


def test_pretrained_fc_random_init_differs_between_builds(pretrained_dir: Path) -> None:
    # Arrange
    weights = pretrained_dir / "resnet18.pth"
    if not weights.is_file():
        pytest.skip("local resnet18.pth cache is not present")

    # Act
    first = create_resnet("resnet18", num_classes=6, pretrained_path=weights)
    second = create_resnet("resnet18", num_classes=6, pretrained_path=weights)

    # Assert: backbone identical, but each build's fc is freshly randomized.
    assert torch.equal(first.conv1.weight, second.conv1.weight)
    assert not torch.equal(first.fc.weight, second.fc.weight)


def test_strict_load_mismatched_keys_raises(tmp_path: Path) -> None:
    # Arrange: a valid torch file whose keys do not match a resnet.
    bad_path = tmp_path / "mismatched.pth"
    torch.save({"not_a_resnet.weight": torch.zeros(3)}, bad_path)

    # Act / Assert: strict=True must reject the mismatched state dict.
    with pytest.raises(RuntimeError):
        create_resnet("resnet18", pretrained_path=bad_path)


def test_strict_load_garbage_file_raises(tmp_path: Path) -> None:
    # Arrange: bytes that are not a serialized torch state dict.
    bad_path = tmp_path / "garbage.pth"
    bad_path.write_bytes(b"this is not a torch checkpoint")

    # Act / Assert: torch 2.5.1's weights_only unpickler surfaces garbage as
    # IndexError (pop_mark on an empty stack); older versions raise
    # UnpicklingError/EOFError. All are acceptable "not a weight file" signs.
    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError, IndexError)):
        create_resnet("resnet18", pretrained_path=bad_path)


def test_strict_load_missing_file_raises(tmp_path: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(FileNotFoundError):
        create_resnet("resnet18", pretrained_path=tmp_path / "absent.pth")


def test_split_params_partitions_all_parameters() -> None:
    # Arrange
    model = create_resnet("resnet18", num_classes=6)

    # Act
    backbone, head = split_params(model)

    # Assert: the two groups are a partition of every parameter of the model.
    all_ids = {id(p) for p in model.parameters()}
    assert {id(p) for p in backbone} | {id(p) for p in head} == all_ids
    assert len(backbone) + len(head) == len(all_ids)
    # The fc head contributes exactly weight + bias.
    assert len(head) == 2
    assert all(any(id(p) == id(param) for param in head) for p in [model.fc.weight, model.fc.bias])


def test_split_params_returns_parameter_objects() -> None:
    # Arrange
    model = create_resnet("resnet50", num_classes=6)

    # Act
    backbone, head = split_params(model)

    # Assert
    assert all(isinstance(p, torch.nn.Parameter) for p in backbone)
    assert all(isinstance(p, torch.nn.Parameter) for p in head)
    # fc params are never duplicated into the backbone group.
    head_ids = {id(p) for p in head}
    assert all(id(p) not in head_ids for p in backbone)
