"""Tests for the hand-rolled NumPy evaluation metrics (T2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from steel_training.eval.metrics import accuracy, confusion_matrix_counts, macro_f1


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_perfect_prediction() -> None:
    # Arrange / Act
    result = accuracy([0, 1, 2, 2, 1, 0], [0, 1, 2, 2, 1, 0])

    # Assert
    assert result == 1.0


def test_accuracy_half_correct() -> None:
    # Arrange / Act
    result = accuracy([0, 0, 1, 1], [0, 1, 0, 1])

    # Assert
    assert result == 0.5


def test_accuracy_accepts_numpy_arrays() -> None:
    # Arrange
    y_true = np.array([2, 0, 1])
    y_pred = np.array([2, 1, 1])

    # Act
    result = accuracy(y_true, y_pred)

    # Assert
    assert result == pytest.approx(2 / 3)


def test_accuracy_empty_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


def test_accuracy_mismatched_lengths_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="length"):
        accuracy([0, 1, 2], [0, 1])


def test_accuracy_non_integer_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="integer"):
        accuracy([0.0, 1.0], [0.0, 1.0])


def test_accuracy_string_labels_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="integer"):
        accuracy(["a", "b"], ["a", "b"])


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------


def test_macro_f1_perfect_single_class() -> None:
    # Arrange / Act
    result = macro_f1([0, 0, 0], [0, 0, 0], num_classes=1)

    # Assert
    assert result == 1.0


def test_macro_f1_perfect_multiclass() -> None:
    # Arrange
    y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    y_pred = list(y_true)

    # Act
    result = macro_f1(y_true, y_pred, num_classes=6)

    # Assert
    assert result == 1.0


def test_macro_f1_all_wrong_is_zero() -> None:
    # Arrange: every prediction is wrong, no class gets any true positive.
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [1, 1, 2, 2, 0, 0]

    # Act
    result = macro_f1(y_true, y_pred, num_classes=3)

    # Assert
    assert result == 0.0


def test_macro_f1_unseen_class_contributes_zero() -> None:
    # Arrange: class 2 appears in neither y_true nor y_pred.
    # Act
    result = macro_f1([0, 1], [0, 1], num_classes=3)

    # Assert: classes 0 and 1 are perfect (1.0 each), class 2 counts as 0.
    assert result == pytest.approx(2 / 3)


def test_macro_f1_known_value() -> None:
    # Arrange: class 0 -> tp=1 fp=0 fn=1 (P=1, R=0.5, F1=2/3);
    #          class 1 -> tp=1 fp=1 fn=0 (P=0.5, R=1, F1=2/3).
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]

    # Act
    result = macro_f1(y_true, y_pred, num_classes=2)

    # Assert
    assert result == pytest.approx(2 / 3)


def test_macro_f1_empty_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="empty"):
        macro_f1([], [], num_classes=3)


def test_macro_f1_mismatched_lengths_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="length"):
        macro_f1([0, 1], [0], num_classes=2)


def test_macro_f1_invalid_num_classes_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="num_classes"):
        macro_f1([0, 1], [0, 1], num_classes=0)


def test_macro_f1_non_integer_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="integer"):
        macro_f1([0.5, 1.5], [0.5, 1.5], num_classes=2)


def test_macro_f1_out_of_range_label_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="range"):
        macro_f1([0, 3], [0, 3], num_classes=3)


def test_macro_f1_negative_label_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="range"):
        macro_f1([-1, 0], [-1, 0], num_classes=2)


# ---------------------------------------------------------------------------
# confusion_matrix_counts
# ---------------------------------------------------------------------------


def test_confusion_matrix_shape_and_row_sums() -> None:
    # Arrange
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 0]

    # Act
    matrix = confusion_matrix_counts(y_true, y_pred, num_classes=3)

    # Assert
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix.sum(axis=1), np.array([2, 2, 1]))
    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [1, 0, 0],
            ]
        ),
    )


def test_confusion_matrix_perfect_diagonal() -> None:
    # Arrange
    y_true = [0, 1, 2, 0, 1, 2]

    # Act
    matrix = confusion_matrix_counts(y_true, y_true, num_classes=3)

    # Assert
    np.testing.assert_array_equal(matrix, np.diag([2, 2, 2]))


def test_confusion_matrix_total_sum_matches_length() -> None:
    # Arrange
    y_true = np.random.default_rng(7).integers(0, 6, size=500)
    y_pred = np.random.default_rng(8).integers(0, 6, size=500)

    # Act
    matrix = confusion_matrix_counts(y_true, y_pred, num_classes=6)

    # Assert
    assert matrix.shape == (6, 6)
    assert matrix.sum() == 500


def test_confusion_matrix_accepts_tuples() -> None:
    # Arrange / Act
    matrix = confusion_matrix_counts((0, 1), (1, 0), num_classes=2)

    # Assert
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [1, 0]]))


def test_confusion_matrix_empty_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="empty"):
        confusion_matrix_counts([], [], num_classes=3)


def test_confusion_matrix_non_integer_input_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(TypeError, match="integer"):
        confusion_matrix_counts([0.0, 1.0], [0.0, 1.0], num_classes=2)


def test_confusion_matrix_out_of_range_label_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="range"):
        confusion_matrix_counts([0, 5], [0, 5], num_classes=3)


def test_confusion_matrix_mismatched_lengths_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="length"):
        confusion_matrix_counts([0, 1], [0, 1, 2], num_classes=3)
