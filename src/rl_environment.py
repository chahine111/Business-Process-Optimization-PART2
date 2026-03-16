"""
src/rl_environment.py

Gymnasium environment wrapping the BPI2017 Discrete Event Simulation Engine
for Reinforcement Learning training (Method C — Middelhuis et al. 2025).

The environment exposes a standard Gym interface:
    env = BPSimEnv(engine_kwargs, all_resources, all_activities)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Decision points
---------------
The event loop runs autonomously until it reaches a RECHECK event where
    (a) at least one task is waiting/suspended, AND
    (b) at least one free resource is authorised for that task type.
At that point control returns to the agent.

Action space
------------
Discrete(|R| × |A| + 1):
    action = r_idx * |A| + a_idx  → assign resource r to the next task of type a
    action = |R| × |A|            → Postpone (no assignment this step)

Observation space
-----------------
Box(2*|R| + |A|,) — see SimulationEngine._rl_get_state() for the layout.

Reward
------
    -cycle_time_hours  accumulated over all cases completed since last step.
    -0.001             per decision step (encourages fewer postponements).

Episode ends when simulation reaches end_time.

Dependencies
------------
    pip install gymnasium stable-baselines3 sb3-contrib
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


# ----------------------------
# Utility — Space Construction
# ----------------------------

def build_resources_and_activities(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract the full sorted list of resources and activities from the event log.

    Returns fixed-length, deterministically-ordered lists so that the indices
    used in the state vector and action space are stable across resets and runs.

    Filters:
        - Resources: all unique 'org:resource' values (non-null, non-empty).
        - Activities: 'concept:name' values matching ^[AOW]_ (BPI 2017 standard
          prefix convention) to exclude lifecycle noise rows.
    """
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
        # Keep only standard BPI 2017 activity names (A_*, O_*, W_*)
        import re
        activities = [a for a in activities if re.match(r"^[AOW]_", a)]

    return resources, activities


# ----------------------------
# Gymnasium Environment
# ----------------------------

class BPSimEnv:
    """
    Gymnasium environment wrapping SimulationEngine for RL training.

    Parameters
    ----------
    engine_kwargs : dict
        All keyword arguments passed to SimulationEngine.__init__().
        ``allocation_method`` is forced to ``"rl"`` internally.
    all_resources : list[str]
        Sorted list of all resource IDs (defines state/action dimensionality).
    all_activities : list[str]
        Sorted list of all activity names.
    """

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

        # Observation: [busy × nR, remaining_norm × nR, task_count × nA]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(2 * nR + nA,), dtype=np.float32
        )
        # Action: nR×nA assignments + 1 Postpone
        self.action_space = spaces.Discrete(nR * nA + 1)

        # Runtime state — re-initialised on every reset()
        self._engine: Any = None                     # active SimulationEngine instance
        self._t: Optional[pd.Timestamp] = None       # current simulation clock
        self._done: bool = False                     # episode terminated flag
        self._pending_reward: float = 0.0            # accumulated reward since last step()
        self._rewarded_cases: Set[str] = set()       # cases already counted in reward

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        _require_gym()
        # gymnasium.Env.reset() seeds the action/obs spaces when seed is given.
        # We don't inherit from gymnasium.Env to keep the import optional, so
        # we handle seeding manually.
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        # Build a fresh engine in RL mode.
        from simulation_engine_1_1 import SimulationEngine

        kwargs = dict(self._engine_kwargs)
        kwargs["allocation_method"] = "rl"          # force RL mode regardless of caller setting
        self._engine = SimulationEngine(**kwargs)
        # Inject the fixed resource/activity lists so state/action indices stay stable.
        self._engine._all_resources_list = self._all_resources
        self._engine._all_activities_list = self._all_activities

        self._done = False
        self._pending_reward = 0.0
        self._rewarded_cases = set()
        self._t = self._engine.start_time

        # Seed the event queue and advance to the first decision point.
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

        # ── Execute action ────────────────────────────────────────────────
        if action != postpone:
            # Decode flat action index → (resource index, activity index).
            r_idx    = action // nA
            a_idx    = action % nA
            resource = self._all_resources[r_idx]
            activity = self._all_activities[a_idx]

            # Find the first matching task (suspended queue has priority).
            task, is_resumed = self._find_and_remove_task(activity)
            if (
                task is not None
                and resource not in eng.busy_resources
                and eng.permissions.can_execute(resource, activity)
            ):
                eng._start_or_resume_task(t, task, resource, resumed=is_resumed)

        # Per-step penalty (encourages avoiding unnecessary postponements).
        self._pending_reward -= 0.001

        # ── Continue until next decision point ────────────────────────────
        self._run_to_decision()

        reward = self._pending_reward
        self._pending_reward = 0.0

        return self._get_state(), float(reward), self._done, False, {}

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get the valid action mask."""
        if self._engine is None or self._t is None:
            mask = np.zeros(self._nR * self._nA + 1, dtype=bool)
            mask[-1] = True
            return mask
        return self._engine._rl_get_action_mask(self._t)

    def render(self):
        pass  # no visual rendering

    def close(self):
        self._engine = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_and_remove_task(self, activity: str):
        """
        Find and remove the first task matching *activity* from the engine queues.
        Suspended queue has priority.
        Returns (task, is_resumed) or (None, False).
        """
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
        """
        Advance the event loop until a decision point is reached or
        the episode ends.

        A decision point is a RECHECK event where:
            - at least one task is waiting/suspended, AND
            - at least one free resource is authorised for any of them.

        Cycle-time rewards for completed cases are harvested here.
        """
        from simulation_engine_1_1 import EventType

        eng = self._engine

        while True:
            # ── Harvest completed-case rewards ────────────────────────────
            for case_id, ct_h in list(eng._rl_completed_cases.items()):
                if case_id not in self._rewarded_cases:
                    self._rewarded_cases.add(case_id)
                    self._pending_reward -= ct_h  # minimise cycle time
            eng._rl_completed_cases.clear()

            # ── Process one event ─────────────────────────────────────────
            result = eng._step_one_event()
            if result is None:
                self._done = True
                return

            etype, t = result
            self._t = t

            # ── Check for decision point at RECHECK events ────────────────
            if etype == EventType.RECHECK and self._is_decision_point(t):
                return

    def _is_decision_point(self, t: pd.Timestamp) -> bool:
        """True iff there is work to do AND a free authorised resource for it."""
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
        """Delegate to engine's RL state builder (same layout as during training)."""
        if self._engine is None:
            return np.zeros(2 * self._nR + self._nA, dtype=np.float32)
        t = self._t or self._engine.start_time
        return self._engine._rl_get_state(t)


# Make BPSimEnv inherit from gymnasium.Env when gymnasium is available.
# We do this at class definition time so that SB3's VecEnv machinery works.
if _HAS_GYM:
    class BPSimEnv(BPSimEnv, gymnasium.Env):  # type: ignore[no-redef]
        pass
