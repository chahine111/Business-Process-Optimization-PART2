"""
src/resource_selector_1_7.py

Three heuristic selectors (random, round-robin, shortest-queue) plus
batch_assign() for K-Batching with Hungarian algorithm.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Optional scipy for Hungarian algorithm in batch_assign.
try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Prefix that marks dummy resource columns in the cost matrix.
_DUMMY_PREFIX = "__dummy_"


class ResourceSelector:

    VALID_STRATEGIES = {"random", "round_robin", "shortest_queue"}

    def __init__(self, strategy: str = "random", seed: int = 42) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self.rng = random.Random(seed)

        # sorted list of every resource seen so far, plus a cyclic pointer for RR
        self._all_resources: List[str] = []
        self._rr_pointer: int = 0

        # cumulative task counts — used as tiebreaker in SHQ
        self._task_counts: Dict[str, int] = {}

    # ---

    def select(
        self,
        candidates: List[str],
        *,
        executing: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not candidates:
            return None

        cands = list(candidates)
        self._register_resources(cands)

        if self.strategy == "random":
            chosen = self.rng.choice(cands)

        elif self.strategy == "round_robin":
            chosen = self._round_robin(cands)

        elif self.strategy == "shortest_queue":
            chosen = self._shortest_queue(cands, executing)

        else:
            # Unreachable — __init__ validates strategy.
            chosen = self.rng.choice(cands)

        self._task_counts[chosen] = self._task_counts.get(chosen, 0) + 1
        return chosen

    # ---

    def _round_robin(self, cands: List[str]) -> str:
        cands_set = set(cands)
        n = len(self._all_resources)
        if n == 0:
            return sorted(cands)[0]

        for offset in range(n):
            idx = (self._rr_pointer + offset) % n
            r = self._all_resources[idx]
            if r in cands_set:
                self._rr_pointer = (idx + 1) % n
                return r

        # fallback — shouldn't happen after _register_resources
        return sorted(cands)[0]

    def _shortest_queue(
        self,
        cands: List[str],
        executing: Optional[Dict[str, Any]],
    ) -> str:
        # count tasks currently executing per resource
        exec_load: Dict[str, int] = {}
        if executing:
            for ex in executing.values():
                r = ex.resource
                exec_load[r] = exec_load.get(r, 0) + 1

        def load_key(r: str) -> Tuple[int, int]:
            return (exec_load.get(r, 0), self._task_counts.get(r, 0))

        min_key = min(load_key(r) for r in cands)
        tied = [r for r in cands if load_key(r) == min_key]
        return self.rng.choice(tied)

    # ---

    def _register_resources(self, cands: List[str]) -> None:
        existing = set(self._all_resources)
        new = [r for r in cands if r not in existing]
        if new:
            self._all_resources = sorted(existing | set(new))

    # ---
    # K-Batching
    # ---

    def batch_assign(
        self,
        tasks: List[Any],
        candidates: List[str],
        duration_model: Any,
        case_attrs: Dict[str, Any],
        current_time: Any,
        mode: str = "basic",
        permissions_model: Optional[Any] = None,
        delta: float = 1.0,
    ) -> List[Tuple[Any, str]]:
        """
        Build a predicted-duration cost matrix and solve it.
        Tasks assigned to dummy columns stay in the queue for the next round.
        """
        if not tasks or not candidates:
            return []

        sample_method = "sample" if mode == "advanced" else "median"
        cur_ts = (
            current_time.to_pydatetime()
            if hasattr(current_time, "to_pydatetime")
            else current_time
        )

        # per-task authorised candidate lists
        task_candidates: List[List[str]] = []
        for task in tasks:
            if permissions_model is not None:
                tc = [r for r in candidates
                      if permissions_model.can_execute(r, task.activity)]
            else:
                tc = list(candidates)
            task_candidates.append(tc)

        real_resources = sorted({r for tc in task_candidates for r in tc})
        if not real_resources:
            return []

        n_tasks = len(tasks)
        n_real  = len(real_resources)
        res_idx = {r: j for j, r in enumerate(real_resources)}

        # build cost matrix — inf for impermissible pairs
        cost = np.full((n_tasks, n_real), np.inf)

        for i, task in enumerate(tasks):
            attrs = case_attrs.get(task.case_id, {})
            for r in task_candidates[i]:
                j = res_idx[r]
                if task.remaining_s is not None:
                    dur = float(task.remaining_s)
                else:
                    try:
                        if hasattr(duration_model, "sample_duration"):
                            dur = float(duration_model.sample_duration(
                                activity=task.activity,
                                case_attributes=attrs,
                                resource=r,
                                current_time=cur_ts,
                                method=sample_method,
                            ))
                        elif hasattr(duration_model, "predict_duration"):
                            dur = float(duration_model.predict_duration(
                                activity=task.activity,
                                case_attributes=attrs,
                                resource=r,
                                current_time=cur_ts,
                                quantile=0.5,
                            ))
                        else:
                            dur = 60.0
                    except Exception:
                        dur = 60.0
                cost[i][j] = max(1.0, dur)

        # dummy columns: cost = δ × (1/|R_auth|) × Σ c(t,r)
        all_resources = list(real_resources)

        if n_tasks > n_real:
            n_dummies = n_tasks - n_real
            row_dummy = np.zeros(n_tasks)
            for i in range(n_tasks):
                finite_vals = cost[i][np.isfinite(cost[i])]
                if finite_vals.size > 0:
                    # δ × (1/|R_auth_t|) × Σ c(t,r)
                    row_dummy[i] = delta * finite_vals.sum() / finite_vals.size
                else:
                    row_dummy[i] = delta * 3600.0  # fallback: 1 h
            dummy_matrix = np.outer(row_dummy, np.ones(n_dummies))
            cost = np.hstack([cost, dummy_matrix])
            for k in range(n_dummies):
                all_resources.append(f"{_DUMMY_PREFIX}{k}__")

        n_res = len(all_resources)
        original_cost = cost.copy()

        if _HAS_SCIPY:
            assignments = self._solve_hungarian(
                tasks, all_resources, original_cost, n_real
            )
        else:
            assignments = self._solve_greedy(
                tasks, all_resources, original_cost, n_real, n_res
            )

        return assignments

    # ---

    def _solve_hungarian(
        self,
        tasks: List[Any],
        all_resources: List[str],
        cost: np.ndarray,
        n_real: int,
    ) -> List[Tuple[Any, str]]:
        # replace inf with large sentinel so scipy can run
        finite_vals = cost[np.isfinite(cost)]
        M = float(finite_vals.max()) * 1000.0 if finite_vals.size > 0 else 1e9
        cost_hp = np.where(np.isfinite(cost), cost, M)

        row_ind, col_ind = _hungarian(cost_hp)

        assignments: List[Tuple[Any, str]] = []
        for i, j in zip(row_ind, col_ind):
            if not np.isfinite(cost[i, j]):
                continue
            r = all_resources[j]
            if r.startswith(_DUMMY_PREFIX):
                continue
            assignments.append((tasks[i], r))
        return assignments

    def _solve_greedy(
        self,
        tasks: List[Any],
        all_resources: List[str],
        cost: np.ndarray,
        n_real: int,
        n_res: int,
    ) -> List[Tuple[Any, str]]:
        # greedy fallback (no scipy): most-constrained task first
        n_tasks = len(tasks)
        task_order = sorted(
            range(n_tasks),
            key=lambda i: int(np.isfinite(cost[i, :n_real]).sum()),
        )

        assignments: List[Tuple[Any, str]] = []
        processed: Set[int] = set()
        taken: Set[str] = set()

        for i in task_order:
            if i in processed:
                continue
            best_j, best_c = -1, np.inf
            for j in range(n_res):
                r = all_resources[j]
                if r in taken:
                    continue
                if cost[i][j] < best_c:
                    best_c = cost[i][j]
                    best_j = j

            if best_j >= 0 and np.isfinite(best_c):
                r = all_resources[best_j]
                processed.add(i)
                taken.add(r)
                if not r.startswith(_DUMMY_PREFIX):
                    assignments.append((tasks[i], r))
        return assignments
