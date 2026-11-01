"""Pure NumPy evaluation metrics (no sklearn, no torch).

All functions validate their inputs and fail fast with clear errors:

- empty inputs raise :class:`ValueError`;
- non-integer label arrays raise :class:`TypeError`;
- labels outside ``[0, num_classes)`` raise :class:`ValueError`
  (only checked where ``num_classes`` is part of the signature);
- ``y_true``/``y_pred`` of different lengths raise :class:`ValueError`.

Labels are per-class integer indices following the manifest's
``class_names`` order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_int_labels(values: ArrayLike, name: str, num_classes: int | None) -> NDArray[np.int64]:
    """Convert ``values`` to a validated 1-D int64 label array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of class labels, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty; metrics require at least one sample")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer class labels, got dtype {array.dtype}")
    labels = array.astype(np.int64)
    if num_classes is not None and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            f"{name} contains labels outside the range [0, {num_classes}): "
            f"min={labels.min()}, max={labels.max()}"
        )
    return labels


def _validate_pair(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int | None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Validate and return the ``(y_true, y_pred)`` label pair."""
    true_labels = _as_int_labels(y_true, "y_true", num_classes)
    pred_labels = _as_int_labels(y_pred, "y_pred", num_classes)
    if true_labels.shape != pred_labels.shape:
        raise ValueError(
            "y_true and y_pred must have the same length, got "
            f"{true_labels.shape[0]} and {pred_labels.shape[0]}"
        )
    return true_labels, pred_labels


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of correctly predicted samples in ``[0, 1]``."""
    true_labels, pred_labels = _validate_pair(y_true, y_pred, num_classes=None)
    return float(np.mean(true_labels == pred_labels))


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> float:
    """Unweighted mean of per-class F1 scores in ``[0, 1]``.

    A class with no predicted positives has precision 0; a class with no true
    positives has recall 0; a class with neither contributes F1 = 0.
    """
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")
    true_labels, pred_labels = _validate_pair(y_true, y_pred, num_classes)

    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(np.sum((true_labels == class_index) & (pred_labels == class_index)))
        false_positive = int(np.sum((true_labels != class_index) & (pred_labels == class_index)))
        false_negative = int(np.sum((true_labels == class_index) & (pred_labels != class_index)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class_f1.append(f1)

    return float(np.mean(per_class_f1))


def confusion_matrix_counts(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int
) -> NDArray[np.int64]:
    """Count-based confusion matrix of shape ``(num_classes, num_classes)``.

    ``matrix[i, j]`` is the number of samples with true class ``i`` that were
    predicted as class ``j``; row sums therefore equal per-class true counts.
    """
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")
    true_labels, pred_labels = _validate_pair(y_true, y_pred, num_classes)

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true_labels, pred_labels), 1)
    return matrix
