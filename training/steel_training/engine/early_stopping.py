"""Early stopping helper, kept separate so its logic is trivially testable."""

from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """Tracks a monitored metric and signals when to stop training.

    ``update`` returns True once ``patience`` consecutive epochs have passed
    without improving the best score (``"max"``: higher is better, default;
    ``"min"``: lower is better). ``patience == 0`` disables stopping while
    still tracking the best score. ``is_best`` is True after an update that
    established a new best score, which the caller uses to persist
    checkpoints.
    """

    def __init__(self, patience: int, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        self.patience = patience
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.counter = 0
        self.is_best = False

    def update(self, metric: float, epoch: int) -> bool:
        """Record ``metric`` for ``epoch``; return True if training should stop."""
        self.is_best = False
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            self.is_best = True
            return False
        improved = (
            metric > self.best_score if self.mode == "max" else metric < self.best_score
        )
        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            self.is_best = True
            return False
        self.counter += 1
        return self.patience > 0 and self.counter >= self.patience

    def reset(self) -> None:
        """Forget all recorded history."""
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.is_best = False
