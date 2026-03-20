"""
src/rl_allocator.py

Loads a MaskablePPO policy from disk and gives the engine a select_action() call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------
# RL Inference Wrapper
# ----------------------------

class RLAllocator:

    def __init__(self, model_path: str | Path, *, deterministic: bool = True) -> None:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "sb3_contrib is required for the RL allocator. "
                "Install it with: pip install sb3-contrib"
            ) from exc

        mpath = Path(model_path)
        if not mpath.exists():
            # try models/ folder as fallback
            project_root = Path(__file__).resolve().parents[1]
            candidate = project_root / "models" / mpath.name
            if candidate.exists():
                mpath = candidate
            else:
                raise FileNotFoundError(
                    f"RL policy not found: {model_path}\n"
                    f"Train first using:  python src/rl_train.py"
                )

        self.model = MaskablePPO.load(str(mpath))
        self._deterministic = deterministic

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        action, _ = self.model.predict(
            state,
            action_masks=mask,
            deterministic=self._deterministic,
        )
        return int(action)
