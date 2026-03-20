"""
src/rl_environment.py

Gymnasium wrapper around the DES engine for RL training.
Control returns to the agent at RECHECK events where there's work and a free resource.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gymnasium = None  # type: ignore[assignment]
    spaces = None     # type: ignore[assignment]


def _require_gym():
    if not _HAS_GYM:
        raise ImportError(
            "gymnasium is required for RL training. "
            "Install it with: pip install gymnasium stable-baselines3 sb3-contrib"
        )


def build_resources_and_activities(csv_path: str) -> Tuple[List[str], List[str]]:
    """Return sorted resource and activity lists; indices must be stable across runs."""
    import pandas as _pd
    df = _pd.read_csv(csv_path, low_memory=False)

    resources: List[str] = []
    if "org:resource" in df.columns:
        resources = sorted(
            df["org:resource"].dropna().astype(str).str.strip().unique().tolist()
        )
        resources = [r for r in resources if r and r != "nan"]

    activities: List[str] = []
    if "concept:name" in df.columns:
        activities = sorted(
            df["concept:name"].dropna().astype(str).str.strip().unique().tolist()
        )
        # only keep A_*, O_*, W_* — skip lifecycle noise rows
        import re
        activities = [a for a in activities if re.match(r"^[AOW]_", a)]

    return resources, activities


class BPSimEnv:

    metadata = {"render_modes": []}

    def __init__(
        self,
        engine_kwargs: Dict[str, Any],
        all_resources: List[str],
        all_activities: List[str],
    ) -> None:
        _require_gym()
        super().__init__()  # type: ignore[call-arg]

        self._engine_kwargs: Dict[str, Any] = engine_kwargs
        self._all_resources: List[str] = sorted(all_resources)
        self._all_activities: List[str] = sorted(all_activities)

        nR = len(self._all_resources)
        nA = len(self._all_activities)
        self._nR = nR
        self._nA = nA

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(2 * nR + nA,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(nR * nA + 1)

        self._engine: Any = None
        self._t: Optional[pd.Timestamp] = None
        self._done: bool = False
        self._pending_reward: float = 0.0
        self._rewarded_cases: Set[str] = set()

    # ---

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        _require_gym()
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        from simulation_engine_1_1 import SimulationEngine

        kwargs = dict(self._engine_kwargs)
        kwargs["allocation_method"] = "rl"
        self._engine = SimulationEngine(**kwargs)
        self._engine._all_resources_list = self._all_resources
        self._engine._all_activities_list = self._all_activities

        self._done = False
        self._pending_reward = 0.0
        self._rewarded_cases = set()
        self._t = self._engine.start_time

        self._engine._init_run()
        self._run_to_decision()

        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_state(), 0.0, True, False, {}

        t   = self._t
        eng = self._engine
        nR, nA = self._nR, self._nA
        postpone = nR * nA

        if action != postpone:
            r_idx    = action // nA
            a_idx    = action % nA
            resource = self._all_resources[r_idx]
            activity = self._all_activities[a_idx]

            # suspended queue has priority
            task, is_resumed = self._find_and_remove_task(activity)
            if (
                task is not None
                and resource not in eng.busy_resources
                and eng.permissions.can_execute(resource, activity)
            ):
                eng._start_or_resume_task(t, task, resource, resumed=is_resumed)

        self._pending_reward -= 0.001  # small penalty per step

        self._run_to_decision()

        reward = self._pending_reward
        self._pending_reward = 0.0

        return self._get_state(), float(reward), self._done, False, {}

    def action_masks(self) -> np.ndarray:
        if self._engine is None or self._t is None:
            mask = np.zeros(self._nR * self._nA + 1, dtype=bool)
            mask[-1] = True
            return mask
        return self._engine._rl_get_action_mask(self._t)

    def render(self):
        pass  # no visual rendering

    def close(self):
        self._engine = None

    # ---

    def _find_and_remove_task(self, activity: str):
        eng = self._engine
        for task in list(eng.suspended):
            if task.activity == activity:
                eng.suspended = type(eng.suspended)(
                    tk for tk in eng.suspended if tk.uid != task.uid
                )
                return task, True
        for task in list(eng.waiting):
            if task.activity == activity:
                eng.waiting = type(eng.waiting)(
                    tk for tk in eng.waiting if tk.uid != task.uid
                )
                return task, False
        return None, False

    def _run_to_decision(self) -> None:
        from simulation_engine_1_1 import EventType

        eng = self._engine

        while True:
            # harvest cycle-time rewards for cases that just finished
            for case_id, ct_h in list(eng._rl_completed_cases.items()):
                if case_id not in self._rewarded_cases:
                    self._rewarded_cases.add(case_id)
                    self._pending_reward -= ct_h  # minimise cycle time
            eng._rl_completed_cases.clear()

            result = eng._step_one_event()
            if result is None:
                self._done = True
                return

            etype, t = result
            self._t = t

            if etype == EventType.RECHECK and self._is_decision_point(t):
                return

    def _is_decision_point(self, t: pd.Timestamp) -> bool:
        eng = self._engine
        if not eng.suspended and not eng.waiting:
            return False

        available = set(eng.availability.get_available_resources(t))
        free = available - eng.busy_resources
        if not free:
            return False

        for task in list(eng.suspended) + list(eng.waiting):
            for r in free:
                if eng.permissions.can_execute(r, task.activity):
                    return True
        return False

    def _get_state(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(2 * self._nR + self._nA, dtype=np.float32)
        t = self._t or self._engine.start_time
        return self._engine._rl_get_state(t)


# inherit from gymnasium.Env when available so SB3's VecEnv machinery works
if _HAS_GYM:
    class BPSimEnv(BPSimEnv, gymnasium.Env):  # type: ignore[no-redef]
        pass
