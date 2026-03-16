"""
src/rl_allocator.py

Runtime wrapper for the trained RL policy (Method C).

Loads a MaskablePPO model from disk and exposes a single
select_action(state, mask) → int interface that SimulationEngine
calls inside _allocate_rl().

Usage
-----
    from rl_allocator import RLAllocator
    allocator = RLAllocator("models/rl_policy.zip")
    engine.rl_allocator = allocator

Dependencies
------------
    pip install stable-baselines3 sb3-contrib gymnasium
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ----------------------------
# RL Inference Wrapper
# ----------------------------

class RLAllocator:
    """
    Thin inference wrapper around a trained MaskablePPO policy.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved MaskablePPO model (``models/rl_policy.zip``).
    deterministic : bool
        If True (default) the policy always picks the highest-probability
        action.  Set False to sample from the policy distribution.
    """

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
            # Allow a bare filename — try the project models/ directory as fallback.
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
        """
        Choose an action given the current state and valid-action mask.

        Parameters
        ----------
        state : np.ndarray, shape (2*|R|+|A|,)
            Observation vector built by SimulationEngine._rl_get_state().
        mask : np.ndarray, shape (|R|*|A|+1,), dtype bool
            Valid-action mask built by SimulationEngine._rl_get_action_mask().

        Returns
        -------
        int
            Action index in [0, |R|*|A|].  |R|*|A| means Postpone.
        """
        action, _ = self.model.predict(
            state,
            action_masks=mask,
            deterministic=self._deterministic,
        )
        return int(action)
