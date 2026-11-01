"""ResNet model factory for the steel surface defect classifier.

Builds torchvision ResNet architectures without bundled weights
(``weights=None``), optionally loading a locally cached ImageNet
``state_dict`` with ``strict=True`` (so key mismatches fail loudly), and
replaces the classification head with a freshly initialized ``num_classes``
output. Also splits parameters into backbone/head groups for differential
learning rates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torchvision.models as models

SUPPORTED_MODELS = ("resnet18", "resnet50")

_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def create_resnet(
    model_name: str,
    num_classes: int = 6,
    pretrained_path: Optional[Path] = None,
) -> nn.Module:
    """Create a ResNet model with a ``num_classes`` classification head.

    Args:
        model_name: one of ``"resnet18"`` or ``"resnet50"``.
        num_classes: number of outputs of the (replaced) ``fc`` head.
        pretrained_path: optional path to a torch ``state_dict`` file. When
            given, the dict is loaded with ``strict=True`` into the
            ``weights=None`` architecture, so any key mismatch raises a
            ``RuntimeError``; the ``fc`` head is then replaced with a fresh
            random ``Linear`` regardless of the loaded weights.

    Returns:
        The constructed model (on CPU; move it to a device afterwards).

    Raises:
        ValueError: for an unsupported ``model_name`` or ``num_classes < 1``.
        RuntimeError: when ``pretrained_path`` exists but its state dict does
            not exactly match the model's keys.
        FileNotFoundError: when ``pretrained_path`` does not exist.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model '{model_name}'; expected one of {SUPPORTED_MODELS}")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")

    model = _BUILDERS[model_name](weights=None)
    if pretrained_path is not None:
        state_dict = torch.load(str(pretrained_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split a ResNet's parameters into ``(backbone, fc_head)`` groups.

    Every parameter whose name starts with ``"fc."`` goes to the head group
    (used with a higher learning rate); everything else belongs to the
    backbone group (fine-tuned with a lower learning rate). The two groups
    form an exact partition of the model's parameters.

    Returns:
        ``(backbone_params, head_params)`` lists of ``nn.Parameter`` objects.
    """
    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            head.append(param)
        else:
            backbone.append(param)
    return backbone, head
