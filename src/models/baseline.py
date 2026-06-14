"""Gradient-boosted-tree baseline (AGENT.md §5.4, §7).

A LightGBM classifier over flattened per-window engineered features. This is the
benchmark the ST-MoE network must beat before the deep model earns its keep
(AGENT.md §1, §7). Falls back to scikit-learn's HistGradientBoosting if LightGBM
is unavailable, so the layer never hard-fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import BaselineConfig


@dataclass
class GBTClassifier:
    """Thin wrapper giving a stable fit/predict API over the chosen GBT backend.

    Existing as a deliberate, hard-to-beat reference point: the ST-MoE only earns
    its complexity by beating this on macro-F1 *and* time-to-detection (AGENT.md §7).
    """

    cfg: BaselineConfig
    # Seeded so the must-beat benchmark is reproducible across runs (AGENT.md §2).
    seed: int = 42

    def __post_init__(self) -> None:
        self._model: Any | None = None
        self._classes: np.ndarray | None = None

    def _build(self) -> Any:
        # Prefer LightGBM (fast, strong on tabular), but never make the layer's
        # availability depend on an optional dep: fall through to sklearn below.
        if self.cfg.backend == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(
                    n_estimators=self.cfg.n_estimators,
                    learning_rate=self.cfg.learning_rate,
                    num_leaves=self.cfg.num_leaves,
                    max_depth=self.cfg.max_depth,
                    class_weight=self.cfg.class_weight,
                    random_state=self.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                pass
        # Drop-in fallback so the baseline (and thus the §7 must-beat gate) always
        # runs; HistGradientBoosting is the closest stdlib-stack equivalent.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=None if self.cfg.max_depth < 0 else self.cfg.max_depth,
            class_weight=self.cfg.class_weight,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBTClassifier:
        self._model = self._build()
        self._model.fit(X, y)
        self._classes = np.asarray(self._model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        assert self._classes is not None
        return self._classes
