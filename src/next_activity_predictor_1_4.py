"""
src/next_activity_predictor_1_4.py

Loads the trained bigram next-activity model and exposes sample_next().
Wraps next_activity_TRAIN_1_4.py so the engine doesn't need to import it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from next_activity_TRAIN_1_4 import NextActivityPredictor
except ImportError:
    # fallback — define the class inline if the training module isn't importable
    import joblib
    import numpy as np
    from typing import Optional, List

    class NextActivityPredictor:

        def __init__(self, model_path: str, seed: int = 42):
            self.model = joblib.load(model_path)
            self.rng = np.random.default_rng(seed)

        def _sample_from(
            self,
            next_list: List[str],
            prob_array: np.ndarray,
            allowed_next: Optional[List[str]] = None,
        ) -> str:
            next_list = list(next_list)
            prob = np.array(prob_array, dtype=float)

            if allowed_next is None or len(allowed_next) == 0:
                return str(self.rng.choice(next_list, p=prob))

            allowed_set = set(allowed_next)
            mask = np.array([n in allowed_set for n in next_list], dtype=bool)

            if not mask.any():
                # Fallback: if model predicts forbidden paths, pick any allowed path
                return str(self.rng.choice(list(allowed_set)))

            next_f = [n for n, m in zip(next_list, mask) if m]
            prob_f = prob[mask]
            
            # Re-normalize
            s = float(prob_f.sum())
            if not np.isfinite(s) or s <= 0:
                return str(self.rng.choice(next_f))

            prob_f = prob_f / s
            return str(self.rng.choice(next_f, p=prob_f))

        def sample_next(
            self,
            prev2: Optional[str],
            prev1: Optional[str],
            allowed_next: Optional[List[str]] = None,
        ) -> str:
            # 1. Bigram
            if prev2 is not None and prev1 is not None:
                key = (prev2, prev1)
                if key in self.model.get("bigram", {}):
                    info = self.model["bigram"][key]
                    return self._sample_from(info["next"], info["prob"], allowed_next)

            # 2. Unigram
            if prev1 is not None and prev1 in self.model.get("unigram", {}):
                info = self.model["unigram"][prev1]
                return self._sample_from(info["next"], info["prob"], allowed_next)

            # 3. Global
            info = self.model["global"]
            return self._sample_from(info["next"], info["prob"], allowed_next)


def load_next_activity_predictor(
    model_path: str,
    peer_py_path: str = "next_activity_TRAIN_1_4.py",
    seed: int = 42
) -> Any:
    """Load the predictor; tries models/ folder if path isn't found directly."""
    mpath = Path(model_path)
    if not mpath.exists():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / "models" / model_path
        if candidate.exists():
            mpath = candidate
        else:
            candidate_name = project_root / "models" / Path(model_path).name
            if candidate_name.exists():
                mpath = candidate_name

    if not mpath.exists():
        raise FileNotFoundError(
            f"Cannot find next-activity model file.\n"
            f"Given: {model_path}\n"
            f"Resolved: {mpath}\n"
            f"Hint: Train the model first using 'next_activity_TRAIN_1_4.py'."
        )

    return NextActivityPredictor(str(mpath), seed=seed)