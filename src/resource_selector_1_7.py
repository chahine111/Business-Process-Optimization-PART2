"""
src/resource_selector_1_7.py

Module 1.7 — Resource Selection Strategy.

Implements the three base heuristics from Russell et al. "Workflow Resource
Patterns" and a batch assignment method for K-Batching:

  "random"         R-RMA — pick a uniformly random free resource (baseline).
  "round_robin"    R-RRA — rotate through all known resources in sorted order,
                   advancing the pointer past each picked resource.
  "shortest_queue" R-SHQ — pick the free resource with the least historical
                   workload (total tasks assigned so far), breaking ties at
                   random.  Accepts the engine's `executing` dict so it can
                   also factor in currently active tasks per resource.

Shift-aware candidate filtering
--------------------------------
Filtering happens in the engine's _allocate() BEFORE select() is called, so
it applies uniformly to all three strategies.  select() simply receives the
already-filtered candidate list.

K-Batching
----------
batch_assign() builds an (n_tasks × n_machines) predicted-duration cost matrix
and solves it with the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
When tasks outnumber free resources, dummy machine columns are appended with
cost = δ × (1/|R_auth|) × Σ c(t,r)  (per the lecture-slide formula).
Tasks assigned to dummy machines stay in the queue for the next round.
Falls back to a greedy solver when scipy is not available.
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
    """
    Multi-strategy resource dispatcher for the BPI2017 simulation.

    Parameters
    ----------
    strategy : str
        One of ``"random"``, ``"round_robin"``, or ``"shortest_queue"``.
    seed : int
        RNG seed (used by random choices in all strategies).
    """

    VALID_STRATEGIES = {"random", "round_robin", "shortest_queue"}

    def __init__(self, strategy: str = "random", seed: int = 42) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self.rng = random.Random(seed)

        # ── Round-Robin state ─────────────────────────────────────────────────
        # Sorted list of every resource ever seen + a cyclic pointer.
        self._all_resources: List[str] = []
        self._rr_pointer: int = 0

        # ── Shortest-Queue state ──────────────────────────────────────────────
        # Cumulative task count per resource — used as a long-run load proxy
        # when all candidates have identical current executing-task counts
        # (which is always the case for free candidates in a single-server model).
        self._task_counts: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — single-task selection
    # ──────────────────────────────────────────────────────────────────────────

    def select(
        self,
        candidates: List[str],
        *,
        executing: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Choose one resource from *candidates*.

        Parameters
        ----------
        candidates :
            Free, permission-filtered (and shift-filtered) resource IDs.
            Pre-filtering happens in ``_allocate()`` before this call.
        executing :
            The engine's ``self.executing`` dict ``{task_id: Execution}``.
            Used by ``"shortest_queue"`` to count currently active tasks per
            resource.  Ignored by the other strategies.

        Returns
        -------
        str or None
            Selected resource ID, or ``None`` when *candidates* is empty.
        """
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

        # Track cumulative assignments for R-SHQ load balancing.
        self._task_counts[chosen] = self._task_counts.get(chosen, 0) + 1
        return chosen

    # ──────────────────────────────────────────────────────────────────────────
    # Strategy implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _round_robin(self, cands: List[str]) -> str:
        """
        R-RRA: Cycle through all known resources in sorted order, picking the
        next one that currently appears in *cands*.  The pointer advances past
        the chosen resource for the next call.
        """
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

        # Fallback: _all_resources stale (shouldn't happen after _register).
        return sorted(cands)[0]

    def _shortest_queue(
        self,
        cands: List[str],
        executing: Optional[Dict[str, Any]],
    ) -> str:
        """
        R-SHQ: Pick the free resource with the least workload.

        Sorting key (ascending):
          1. Current executing-task count  (from the engine's executing dict)
          2. Cumulative historical task count  (long-run load balancing)
          3. Random tiebreak

        In a single-server model all free candidates have 0 executing tasks,
        so the historical count drives differentiation between candidates.
        """
        # Count tasks currently executing per resource.
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

    # ──────────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────────

    def _register_resources(self, cands: List[str]) -> None:
        """Register newly seen resource IDs into the sorted global list."""
        existing = set(self._all_resources)
        new = [r for r in cands if r not in existing]
        if new:
            self._all_resources = sorted(existing | set(new))

    # ──────────────────────────────────────────────────────────────────────────
    # K-Batching: Parallel Machines Scheduling (Hungarian / greedy fallback)
    # ──────────────────────────────────────────────────────────────────────────

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
        K-Batch assignment via predicted-duration cost matrix.

        Cost matrix
        -----------
        c(t, r) = predicted processing time of task t on resource r.
        c(t, r) = ∞  for impermissible (task, resource) pairs.

        Dummy columns (when n_tasks > n_real_resources)
        ------------------------------------------------
        For each dummy column d:
          c(t, d) = δ × (1 / |R_auth_t|) × Σ_{r ∈ R_auth_t} c(t, r)

        where R_auth_t is the set of authorised real resources for task t.
        Tasks assigned to dummy columns are not started — they stay in the
        queue for the next batch window.

        Solver
        ------
        Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm) when
        scipy is available.  Falls back to a greedy most-constrained-first
        heuristic otherwise.

        Parameters
        ----------
        tasks :
            Batch of Task objects (``.activity``, ``.case_id``, ``.remaining_s``).
        candidates :
            Currently free resource IDs (before permission filtering).
        duration_model :
            Module 1.3 model; must expose ``sample_duration()`` or
            ``predict_duration()``.
        case_attrs :
            ``{case_id: attribute_dict}`` from the engine.
        current_time :
            Current simulation timestamp (``pd.Timestamp`` or ``datetime``).
        mode :
            ``"basic"`` (median) or ``"advanced"`` (sampled) duration method.
        permissions_model :
            Module 1.6 model with ``can_execute(resource, activity)``.
            When ``None`` every candidate is valid for every task.
        delta : float
            Dummy-column cost multiplier (default 1.0).

        Returns
        -------
        List of (task, resource) pairs for real assignments.
        Tasks matched to dummy columns are excluded (left in queue).
        """
        if not tasks or not candidates:
            return []

        sample_method = "sample" if mode == "advanced" else "median"
        cur_ts = (
            current_time.to_pydatetime()
            if hasattr(current_time, "to_pydatetime")
            else current_time
        )

        # ── Per-task authorised candidate lists ─────────────────────────────
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

        # ── Build cost matrix for real resources ─────────────────────────────
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

        # ── Dummy columns: cost = δ × (1/|R_auth|) × Σ c(t,r) ──────────────
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
        original_cost = cost.copy()  # preserve inf markers before replacement

        # ── Solve ─────────────────────────────────────────────────────────────
        if _HAS_SCIPY:
            assignments = self._solve_hungarian(
                tasks, all_resources, original_cost, n_real
            )
        else:
            assignments = self._solve_greedy(
                tasks, all_resources, original_cost, n_real, n_res
            )

        return assignments

    # ── Solvers ───────────────────────────────────────────────────────────────

    def _solve_hungarian(
        self,
        tasks: List[Any],
        all_resources: List[str],
        cost: np.ndarray,
        n_real: int,
    ) -> List[Tuple[Any, str]]:
        """
        Solve the assignment problem with the Hungarian algorithm.

        inf entries are replaced with a large finite sentinel M so that
        linear_sum_assignment can operate.  Assignments where the original
        cost was inf (impermissible) are discarded post-solve.
        """
        finite_vals = cost[np.isfinite(cost)]
        M = float(finite_vals.max()) * 1000.0 if finite_vals.size > 0 else 1e9
        cost_hp = np.where(np.isfinite(cost), cost, M)

        row_ind, col_ind = _hungarian(cost_hp)

        assignments: List[Tuple[Any, str]] = []
        for i, j in zip(row_ind, col_ind):
            if not np.isfinite(cost[i, j]):
                continue  # impermissible real pair chosen by algorithm
            r = all_resources[j]
            if r.startswith(_DUMMY_PREFIX):
                continue  # deferred task
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
        """
        Greedy fallback (no scipy): most-constrained task first.
        """
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
