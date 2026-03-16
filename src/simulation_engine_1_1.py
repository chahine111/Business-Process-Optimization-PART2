"""
simulation_engine_1_1.py

Module 1.1 — Discrete Event Simulation Engine.

This module acts as the central orchestrator for the simulation. It implements a
Discrete Event Simulation (DES) loop that advances time based on events rather
than fixed ticks.

Key Features:
1.  **Event-Driven**: Jumps efficiently between timestamps (Arrivals, Completions, Resource Checks).
2.  **Resource Buckets**: Manages resource availability in fixed time windows (1h or 2h) to
    simulate shifts accurately.
3.  **Priority Queueing**: Prioritizes suspended tasks (resuming work) over new waiting tasks.
4.  **Smart Logging**:
    - Logs standard BPI attributes (lifecycle:transition, org:resource).
    - Merges case attributes (LoanGoal, Amount) into every log row.
    - Only logs 'schedule' events if a task actually experiences a wait time.
5.  **Modular Integration**: Connects Arrival, Duration, Availability, and Permission models.
6.  **Park & Song Allocation** (Method A): Prediction-based assignment with strategic idling.
    Predicts each running case's next activity, adds predicted tasks to the Hungarian
    cost matrix at a discounted cost, and reserves resources for them (Park & Song 2019).
7.  **Deep RL Allocation** (Method C): MaskablePPO neural-network policy that observes
    the simulation state and selects (resource, task) pairs or postpones.  Training is
    done externally via rl_train.py; inference is handled by RLAllocator (rl_allocator.py).

Allocation Methods (allocation_method parameter):
    "standard"  — existing heuristics or K-batching (default, backward-compatible).
    "park_song" — prediction-based strategic idling (Park & Song 2019).
    "rl"        — deep RL policy (Middelhuis et al. 2025).

Usage:
    engine = SimulationEngine(...)
    csv_path = engine.run()

    # RL step-by-step (used by rl_environment.BPSimEnv):
    engine._init_run()
    while True:
        result = engine._step_one_event()
        if result is None:
            break
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np
import pandas as pd

# Optional scipy — needed by Park & Song cost-matrix solver.
try:
    from scipy.optimize import linear_sum_assignment as _ps_hungarian
    _HAS_SCIPY = True
except ImportError:
    _ps_hungarian = None  # type: ignore[assignment]
    _HAS_SCIPY = False

# ----------------------------
# Optional XES export support
# ----------------------------
try:
    import pm4py
except Exception:
    pm4py = None


# ----------------------------
# Event types & Priority
# ----------------------------
class EventType:
    """
    Defines the distinct types of events handled by the simulation loop.
    """
    ACTIVITY_COMPLETE = "ACTIVITY_COMPLETE"  # A task finished executing
    RESOURCE_CHECK    = "RESOURCE_CHECK"     # End of a time bucket (shift change check)
    CASE_ARRIVAL      = "CASE_ARRIVAL"       # A new case enters the system
    RECHECK           = "RECHECK"            # Trigger to re-evaluate resource allocation


# Priority map for events occurring at the exact same timestamp.
# Logic: Finish work first -> Check resources -> Accept new work -> Allocate.
_EVENT_PRIORITY = {
    EventType.ACTIVITY_COMPLETE: 0,
    EventType.RESOURCE_CHECK: 1,
    EventType.CASE_ARRIVAL: 2,
    EventType.RECHECK: 3,
}


@dataclass
class Event:
    """Represents a single event in the priority queue."""
    ts: pd.Timestamp
    type: str
    payload: dict


@dataclass
class Task:
    """
    Represents a unit of work waiting to be performed.
    
    Attributes:
        pending_schedule: If True, we log a 'schedule' event only after
                          determining the task cannot start immediately.
    """
    uid: str
    case_id: str
    activity: str
    ready_ts: pd.Timestamp
    remaining_s: Optional[float] = None  # Only populated if task was suspended
    pending_schedule: bool = False       


@dataclass
class Execution:
    """Represents a task currently being executed by a resource."""
    task_id: str
    case_id: str
    activity: str
    resource: str
    slice_start: pd.Timestamp
    remaining_s: float
    complete_ts: Optional[pd.Timestamp] = None  # Scheduled finish time (if within bucket)


# ----------------------------
# Case Attribute Sampler
# ----------------------------
class CaseAttributeSampler:
    """
    Generates realistic case attributes (LoanGoal, ApplicationType, Amount) 
    by sampling from the historical distributions found in the input CSV.
    
    Logic:
    - Uses joint distributions (Goal + AppType) to sample RequestedAmount intelligently.
    - Falls back to global distributions if specific combinations are rare.
    """

    def __init__(self, csv_path: str, seed: int = 42):
        self.csv_path = str(csv_path)
        self.rng = np.random.default_rng(seed)

        # Storage for distributions
        self.loan_goals: List[str] = []
        self.loan_goal_p: np.ndarray = np.array([], dtype=float)

        self.app_types: List[str] = []
        self.app_type_p: np.ndarray = np.array([], dtype=float)

        self.amounts_global: np.ndarray = np.array([], dtype=float)
        self.global_lo: float = 1000.0
        self.global_hi: float = 50000.0

        # Conditional maps for specific sampling
        self.amounts_by_goal: Dict[str, np.ndarray] = {}
        self.bounds_by_goal: Dict[str, Tuple[float, float]] = {}
        self.amounts_by_goal_app: Dict[Tuple[str, str], np.ndarray] = {}
        self.bounds_by_goal_app: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._load()

        # Validation to prevent running without data
        if len(self.loan_goals) == 0:
            raise ValueError(
                "CaseAttributeSampler Error: No 'case:LoanGoal' found in CSV. "
                "Ensure the input data is valid."
            )
        if len(self.app_types) == 0:
            raise ValueError(
                "CaseAttributeSampler Error: No 'case:ApplicationType' found in CSV. "
                "Ensure the input data is valid."
            )

    def _safe_probs(self, counts: pd.Series) -> np.ndarray:
        """Normalizes counts to probabilities."""
        p = counts.to_numpy(dtype=float)
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones(len(p), dtype=float) / max(1, len(p))
        return p / s

    def _clean_str(self, s: pd.Series) -> pd.Series:
        """Cleans string columns (removes NaNs and whitespace)."""
        s = s.dropna().astype(str).str.strip()
        return s[s != ""]

    def _compute_bounds(self, x: np.ndarray, lo_q=0.05, hi_q=0.95) -> Tuple[float, float]:
        """Computes quantile bounds to filter outliers during sampling."""
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size == 0:
            return (self.global_lo, self.global_hi)

        lo = float(np.quantile(x, lo_q))
        hi = float(np.quantile(x, hi_q))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (self.global_lo, self.global_hi)

        return (lo, hi)

    def _load(self):
        """Parses the CSV to build probability distributions."""
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
        except Exception:
            return

        # 1. Load Goals
        if "case:LoanGoal" in df.columns:
            col = self._clean_str(df["case:LoanGoal"])
            if len(col) > 0:
                counts = col.value_counts()
                self.loan_goals = counts.index.tolist()
                self.loan_goal_p = self._safe_probs(counts)

        # 2. Load Application Types
        if "case:ApplicationType" in df.columns:
            col = self._clean_str(df["case:ApplicationType"])
            if len(col) > 0:
                counts = col.value_counts()
                self.app_types = counts.index.tolist()
                self.app_type_p = self._safe_probs(counts)

        # 3. Load Amounts (Global)
        if "case:RequestedAmount" in df.columns:
            amt = pd.to_numeric(df["case:RequestedAmount"], errors="coerce").dropna()
            amt = amt[(amt > 0) & np.isfinite(amt)]
            if len(amt) > 0:
                self.amounts_global = amt.to_numpy(dtype=float)
                self.global_lo, self.global_hi = self._compute_bounds(self.amounts_global, 0.01, 0.99)

        # 4. Load Amounts (Conditional)
        if "case:RequestedAmount" in df.columns and self.amounts_global.size > 0:
            tmp = df.copy()
            tmp["case:RequestedAmount"] = pd.to_numeric(tmp["case:RequestedAmount"], errors="coerce")
            tmp = tmp.dropna(subset=["case:RequestedAmount"])
            tmp = tmp[(tmp["case:RequestedAmount"] > 0) & np.isfinite(tmp["case:RequestedAmount"])]

            has_goal = "case:LoanGoal" in tmp.columns
            has_app = "case:ApplicationType" in tmp.columns

            if has_goal:
                tmp["case:LoanGoal"] = tmp["case:LoanGoal"].astype(str).str.strip()
                for g, gg in tmp.groupby("case:LoanGoal"):
                    x = gg["case:RequestedAmount"].to_numpy(dtype=float)
                    if len(x) >= 80:  # Threshold for statistical significance
                        self.amounts_by_goal[g] = x
                        self.bounds_by_goal[g] = self._compute_bounds(x, 0.05, 0.95)

            if has_goal and has_app:
                tmp["case:ApplicationType"] = tmp["case:ApplicationType"].astype(str).str.strip()
                for (g, a), gg in tmp.groupby(["case:LoanGoal", "case:ApplicationType"]):
                    x = gg["case:RequestedAmount"].to_numpy(dtype=float)
                    if len(x) >= 80:
                        self.amounts_by_goal_app[(g, a)] = x
                        self.bounds_by_goal_app[(g, a)] = self._compute_bounds(x, 0.05, 0.95)

    def _round_amount(self, x: float) -> float:
        """Rounds amount to nearest 100 for realism."""
        return float(int(round(x / 100.0)) * 100)

    def _sample_amount_logical(self, loan_goal: str, app_type: str) -> float:
        """Hierarchical sampling logic: Specific -> Semi-Specific -> Global."""
        # Try most specific
        pool = self.amounts_by_goal_app.get((loan_goal, app_type))
        bounds = self.bounds_by_goal_app.get((loan_goal, app_type))

        # Fallback to Goal only
        if pool is None or bounds is None:
            pool = self.amounts_by_goal.get(loan_goal)
            bounds = self.bounds_by_goal.get(loan_goal)

        # Fallback to Global
        if pool is None or bounds is None:
            pool = self.amounts_global
            bounds = (self.global_lo, self.global_hi)

        lo, hi = bounds

        # Attempt to sample within bounds
        for _ in range(50):
            x = float(self.rng.choice(pool))
            if lo <= x <= hi:
                return self._round_amount(x)

        # Last resort: random integer within bounds
        x = float(self.rng.integers(int(lo), int(hi)))
        return self._round_amount(x)

    def sample(self) -> Dict[str, Any]:
        """Returns a dictionary of sampled attributes for a new case."""
        loan_goal = str(self.rng.choice(self.loan_goals, p=self.loan_goal_p))
        app_type = str(self.rng.choice(self.app_types, p=self.app_type_p))
        requested_amount = self._sample_amount_logical(loan_goal, app_type)

        out: Dict[str, Any] = {
            "case:LoanGoal": loan_goal,
            "case:ApplicationType": app_type,
            "case:RequestedAmount": requested_amount,
        }

        # Generate secondary derived fields (safe defaults)
        out.setdefault("CreditScore", int(self.rng.integers(300, 850)))
        out.setdefault("OfferedAmount", float(out["case:RequestedAmount"]))
        out.setdefault("NumberOfTerms", int(self.rng.choice([12, 24, 36, 48, 60])))
        out.setdefault("MonthlyCost", float(self.rng.integers(100, 1000)))
        out.setdefault("FirstWithdrawalAmount", float(self.rng.integers(500, 10000)))

        return out


# ----------------------------
# Simulation Engine Core
# ----------------------------
class SimulationEngine:
    def __init__(
        self,
        *,
        bpmn: Any,
        arrival_process: Any,
        duration_model: Any,
        next_activity_model: Any,
        availability_model: Any,
        permissions_model: Any,
        selector: Any,
        mode: str = "basic",
        start_time: str | pd.Timestamp = "2016-01-01 00:00:00+00:00",
        end_time: str | pd.Timestamp = "2027-01-01 00:00:00+00:00",
        out_csv_path: str = "simulated_log.csv",
        out_xes_path: Optional[str] = None,
        seed: int = 42,
        batch_size: int = 1,
        allocation_method: str = "standard",
        park_song_discount: float = 0.7,
    ):
        """
        Initialize the Discrete Event Simulation Engine.
        
        Args:
            bpmn: Adapter for BPMN control flow.
            arrival_process: Model for case arrival times (Module 1.2).
            duration_model: Model for activity duration (Module 1.3).
            next_activity_model: Model for XOR routing (Module 1.4).
            availability_model: Model for resource shifts (Module 1.5).
            permissions_model: Model for resource-activity mapping (Module 1.6).
            selector: Logic for picking a specific resource (Module 1.7).
            mode: 'basic' (1h buckets) or 'advanced' (2h buckets + sophisticated rules).
            batch_size: K for K-Batching.  1 = standard one-by-one allocation
                (default, backward-compatible).  >1 = accumulate that many
                tasks before running a batch cost-matrix assignment.
            allocation_method: High-level allocation strategy.
                "standard"  — existing heuristics / K-batching (default).
                "park_song" — prediction-based with strategic idling (Park & Song 2019).
                "rl"        — deep RL policy (Middelhuis et al. 2025).
            park_song_discount: Cost multiplier for predicted tasks in the Park & Song
                cost matrix (default 0.7).  Lower values make reservations more
                attractive; higher values make them more conservative.
        """
        self.mode = mode.lower().strip()
        assert self.mode in {"basic", "advanced"}

        self.bpmn = bpmn
        self.arrivals = arrival_process
        self.duration = duration_model
        self.next_act = next_activity_model
        self.availability = availability_model
        self.permissions = permissions_model
        self.selector = selector

        self.start_time = pd.Timestamp(start_time).tz_convert("UTC")
        self.end_time = pd.Timestamp(end_time).tz_convert("UTC")

        # Ensure output directory exists
        project_root = Path(__file__).resolve().parents[1]
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Config CSV output
        out_csv_p = Path(str(out_csv_path))
        if not out_csv_p.is_absolute():
            out_csv_p = outputs_dir / out_csv_p.name
        self.out_csv_path = str(out_csv_p)

        # Config XES output (optional)
        self.out_xes_path = None
        if out_xes_path:
            out_xes_p = Path(str(out_xes_path))
            if not out_xes_p.is_absolute():
                out_xes_p = outputs_dir / out_xes_p.name
            self.out_xes_path = str(out_xes_p)

        self.rng = np.random.default_rng(seed)

        # --- Queue Initialization ---
        # EQ: Priority Queue for events
        self.eventq: List[Tuple[pd.Timestamp, int, int, Event]] = []
        self.seq = itertools.count()

        # EP: Dictionary of currently executing tasks
        self.executing: Dict[str, Execution] = {}
        self.busy_resources: Set[str] = set()

        # SA: Suspended tasks (High Priority - Resuming)
        self.suspended: Deque[Task] = deque()
        # WA: Waiting tasks (Normal Priority - New)
        self.waiting: Deque[Task] = deque()

        # Counters and State
        self.case_counter = 0
        self.exec_counter = 0
        self.wait_uid_counter = 0

        self.case_attrs: Dict[str, Dict[str, Any]] = {}
        self.case_last2: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Load Attribute Sampler
        csv_path = getattr(arrival_process, "csv_path", "bpi2017.csv")
        self.attr_sampler = CaseAttributeSampler(csv_path, seed=seed)

        self.log_rows: List[Dict[str, Any]] = []

        # Time bucket size for resource checks
        self.bucket_hours = 1 if self.mode == "basic" else 2

        # K-Batching state
        # batch_size == 1  →  standard one-by-one dispatch (no change in behaviour)
        # batch_size >  1  →  accumulate tasks then do cost-matrix assignment
        self.batch_size: int = max(1, int(batch_size))
        # Tracks when the last task entered waiting/suspended; used by the
        # batch timeout so a partial batch is eventually flushed even when
        # no new arrivals fill it to batch_size.
        self._last_enqueue_ts: pd.Timestamp = self.start_time

        # ── Allocation method ──────────────────────────────────────────────────
        # "standard"  — existing heuristics / K-batching (unchanged behaviour)
        # "park_song" — prediction-based strategic idling (Park & Song 2019)
        # "rl"        — deep RL policy (Middelhuis et al. 2025)
        _valid_alloc = {"standard", "park_song", "rl"}
        self.allocation_method: str = allocation_method.lower().strip()
        assert self.allocation_method in _valid_alloc, \
            f"Unknown allocation_method '{allocation_method}'. Choose from {_valid_alloc}"

        # Park & Song — strategic-idling reservation store
        # Reservations: (case_id, activity) → reserved resource ID
        self._park_song_discount: float = float(park_song_discount)
        self._reservations: Dict[Tuple[str, str], str] = {}
        self._reservation_ts: Dict[Tuple[str, str], pd.Timestamp] = {}

        # RL — populated by rl_environment / run_simulation after construction
        # rl_allocator.select_action(state, mask) is called inside _allocate_rl()
        self.rl_allocator: Optional[Any] = None
        self._all_resources_list: List[str] = []   # fixed-order resource index
        self._all_activities_list: List[str] = []  # fixed-order activity index

        # RL — per-case cycle-time tracking for reward computation
        self._rl_case_start: Dict[str, pd.Timestamp] = {}
        # Completed-case cycle times (hours); harvested by rl_environment per step
        self._rl_completed_cases: Dict[str, float] = {}
        self._rl_max_cases: Optional[int] = None

    # ---------- Event Queue Management ----------
    def _push_event(self, ts: pd.Timestamp, etype: str, payload: Optional[dict] = None):
        """Adds an event to the priority queue."""
        ts = pd.Timestamp(ts).tz_convert("UTC")
        ev = Event(ts=ts, type=etype, payload=payload or {})
        prio = _EVENT_PRIORITY.get(etype, 99)
        # Use (ts, priority, sequence_id, event) for stable sorting
        heapq.heappush(self.eventq, (ts, prio, next(self.seq), ev))

    def _pop_event(self) -> Optional[Event]:
        """Retrieves the next event from the queue."""
        if not self.eventq:
            return None
        return heapq.heappop(self.eventq)[-1]

    # ---------- ID Generation ----------
    def _new_case_id(self) -> str:
        self.case_counter += 1
        return f"Case_{self.case_counter}"

    def _new_exec_id(self) -> str:
        self.exec_counter += 1
        return f"T{self.exec_counter}"

    def _new_wait_uid(self) -> str:
        self.wait_uid_counter += 1
        return f"W{self.wait_uid_counter}"

    # ---------- Logging Logic ----------
    def _log(self, ts: pd.Timestamp, case_id: str, activity: str, transition: str, resource: Optional[str]):
        """
        Records a simulation event to the log list.
        Merges static case attributes into every row for data completeness.
        """
        row = {
            "time:timestamp": pd.Timestamp(ts).tz_convert("UTC").isoformat(),
            "case:concept:name": case_id,
            "concept:name": activity,
            "lifecycle:transition": transition,
            "org:resource": resource if resource else "",
        }

        # Merge Case Attributes
        attrs = self.case_attrs.get(case_id, {})
        for k, v in (attrs or {}).items():
            if k not in row:
                row[k] = v

        self.log_rows.append(row)

    # ---------- Resource Bucket Logic ----------
    def _next_bucket_boundary(self, t: pd.Timestamp) -> pd.Timestamp:
        """Calculates the timestamp of the next shift/resource check boundary."""
        t = pd.Timestamp(t).tz_convert("UTC")
        t0 = t.floor("h")

        if (t - t0).total_seconds() == 0:
            return t0 + pd.Timedelta(hours=self.bucket_hours)

        nxt = t0 + pd.Timedelta(hours=1)
        if self.bucket_hours == 2:
            if int(nxt.hour) % 2 == 1:
                nxt = nxt + pd.Timedelta(hours=1)
        return nxt

    def _bucket_end(self, t: pd.Timestamp) -> pd.Timestamp:
        return self._next_bucket_boundary(t)

    # ---------- Queue Management ----------
    def _enqueue_waiting_no_log(self, t: pd.Timestamp, case_id: str, activity: str) -> Task:
        """
        Adds a task to the Waiting Queue (WA).
        Note: We do NOT log 'schedule' here. It is only logged later if the task 
        fails to start immediately (contention).
        """
        t = pd.Timestamp(t).tz_convert("UTC")
        task = Task(
            uid=self._new_wait_uid(),
            case_id=case_id,
            activity=activity,
            ready_ts=t,
            remaining_s=None,
            pending_schedule=True,
        )
        self.waiting.append(task)
        self._last_enqueue_ts = t  # used by K-batch timeout
        return task

    def _flush_schedules(self, t: pd.Timestamp):
        """
        Checks for tasks that are still waiting after a resource allocation phase.
        If they are waiting, we log the 'schedule' event now.
        """
        t = pd.Timestamp(t).tz_convert("UTC")
        for task in self.waiting:
            if task.pending_schedule and task.ready_ts == t:
                self._log(t, task.case_id, task.activity, "schedule", "")
                task.pending_schedule = False

    # ---------- Execution Logic ----------
    def _start_or_resume_task(self, t: pd.Timestamp, task: Task, resource: str, *, resumed: bool):
        """
        Transitions a task from Waiting/Suspended -> Executing.
        Determines if the task can finish within the current resource bucket.
        """
        t = pd.Timestamp(t).tz_convert("UTC")
        bucket_end = self._bucket_end(t)
        bucket_remaining_s = max(0.0, float((bucket_end - t).total_seconds()))

        # Determine Duration
        if resumed and task.remaining_s is not None:
            dur_s = float(task.remaining_s)
        else:
            attrs = self.case_attrs[task.case_id]
            if hasattr(self.duration, "sample_duration"):
                dur_s = float(
                    self.duration.sample_duration(
                        activity=task.activity,
                        case_attributes=attrs,
                        resource=resource,
                        current_time=t.to_pydatetime(),
                        method="sample" if self.mode == "advanced" else "median",
                    )
                )
            elif hasattr(self.duration, "predict_duration"):
                dur_s = float(
                    self.duration.predict_duration(
                        activity=task.activity,
                        case_attributes=attrs,
                        resource=resource,
                        current_time=t.to_pydatetime(),
                        quantile=0.5,
                    )
                )
            else:
                dur_s = 60.0

        dur_s = max(1.0, float(dur_s))

        # Create Execution Object
        exec_id = self._new_exec_id()
        ex = Execution(
            task_id=exec_id,
            case_id=task.case_id,
            activity=task.activity,
            resource=resource,
            slice_start=t,
            remaining_s=dur_s,
            complete_ts=None,
        )

        self.executing[exec_id] = ex
        self.busy_resources.add(resource)

        self._log(t, task.case_id, task.activity, "resume" if resumed else "start", resource)

        # Check: Can we finish in this bucket?
        if dur_s <= bucket_remaining_s + 1e-9:
            complete_ts = t + pd.to_timedelta(dur_s, unit="s")
            ex.complete_ts = complete_ts
            self._push_event(complete_ts, EventType.ACTIVITY_COMPLETE, {"task_id": exec_id})
        # If not, the RESOURCE_CHECK event at bucket boundary will handle suspension.

    # ---------- Completion Logic ----------
    def _complete_task(self, t: pd.Timestamp, task_id: str):
        """
        Handles the completion of a task.
        Frees the resource and triggers the next activity in the process.
        """
        t = pd.Timestamp(t).tz_convert("UTC")
        ex = self.executing.get(task_id)
        
        # Validation checks
        if ex is None:
            return
        if ex.complete_ts is None:
            return
        if abs((ex.complete_ts - t).total_seconds()) > 1e-6:
            return

        # Free resources
        del self.executing[task_id]
        self.busy_resources.discard(ex.resource)

        self._log(t, ex.case_id, ex.activity, "complete", ex.resource)

        # Update N-Gram History
        prev2, prev1 = self.case_last2.get(ex.case_id, (None, None))
        self.case_last2[ex.case_id] = (prev1, ex.activity)

        # Check for process end
        if hasattr(self.bpmn, "is_final") and self.bpmn.is_final(ex.activity):
            # RL: record cycle time so the environment can compute the reward.
            if self.allocation_method == "rl":
                start_ts = self._rl_case_start.get(ex.case_id)
                if start_ts is not None:
                    self._rl_completed_cases[ex.case_id] = float(
                        (t - start_ts).total_seconds()
                    ) / 3600.0
            return

        # Determine next activity via BPMN + Predictor
        allowed_next = []
        if hasattr(self.bpmn, "allowed_next"):
            allowed_next = list(self.bpmn.allowed_next(ex.activity, self.case_attrs[ex.case_id]) or [])

        if not allowed_next:
            return

        if len(allowed_next) == 1:
            nxt = allowed_next[0]
        else:
            prev2, prev1 = self.case_last2.get(ex.case_id, (None, None))
            if hasattr(self.next_act, "sample_next"):
                nxt = self.next_act.sample_next(prev2, prev1, allowed_next=allowed_next)
            else:
                nxt = str(self.rng.choice(allowed_next))

        # Add next task to queue.
        # Park & Song fast-path: if a reservation exists for (case_id, nxt),
        # skip the queue and start on the reserved resource immediately.
        if self.allocation_method == "park_song":
            key = (ex.case_id, nxt)
            if key in self._reservations:
                reserved_res = self._reservations.pop(key)
                self._reservation_ts.pop(key, None)
                # Release the strategic hold so the resource is not double-counted.
                self.busy_resources.discard(reserved_res)
                task = Task(
                    uid=self._new_wait_uid(),
                    case_id=ex.case_id,
                    activity=nxt,
                    ready_ts=t,
                    remaining_s=None,
                    pending_schedule=False,
                )
                self._start_or_resume_task(t, task, reserved_res, resumed=False)
                self._push_event(t, EventType.RECHECK, {})
                return

        self._enqueue_waiting_no_log(t, ex.case_id, nxt)

        # Trigger re-allocation immediately
        self._push_event(t, EventType.RECHECK, {})

    # ---------- Bucket Boundary Check ----------
    def _resource_check(self, t: pd.Timestamp):
        """
        Runs at the end of every time bucket.
        1. Identifies tasks that must be suspended (resource leaving).
        2. Identifies tasks that can now finish in the new bucket.
        """
        t = pd.Timestamp(t).tz_convert("UTC")

        available_now = set(self.availability.get_available_resources(t))
        bucket_end = self._bucket_end(t)
        bucket_remaining_s = max(0.0, float((bucket_end - t).total_seconds()))

        to_suspend: List[str] = []
        to_schedule_complete: List[str] = []

        for task_id, ex in list(self.executing.items()):
            # Update remaining duration based on time elapsed in previous bucket
            elapsed = max(0.0, float((t - ex.slice_start).total_seconds()))
            ex.remaining_s = max(0.0, ex.remaining_s - elapsed)
            ex.slice_start = t

            # Case A: Resource is no longer available -> Suspend
            if ex.resource not in available_now:
                to_suspend.append(task_id)
                continue

            # Case B: Resource available, check if task finishes in new bucket
            if ex.complete_ts is None and ex.remaining_s <= bucket_remaining_s + 1e-9:
                to_schedule_complete.append(task_id)

        # Execute Suspensions
        for task_id in to_suspend:
            ex = self.executing.pop(task_id, None)
            if ex is None:
                continue

            self.busy_resources.discard(ex.resource)
            self._log(t, ex.case_id, ex.activity, "suspend", ex.resource)

            # Move to Suspended Queue (High Priority)
            self.suspended.append(
                Task(
                    uid=self._new_wait_uid(),
                    case_id=ex.case_id,
                    activity=ex.activity,
                    ready_ts=t,
                    remaining_s=float(ex.remaining_s),
                    pending_schedule=False,
                )
            )

        if to_suspend:
            self._last_enqueue_ts = t  # suspended tasks re-entered the queue

        # Execute New Completions
        for task_id in to_schedule_complete:
            ex = self.executing.get(task_id)
            if ex is None:
                continue
            complete_ts = t + pd.to_timedelta(ex.remaining_s, unit="s")
            ex.complete_ts = complete_ts
            self._push_event(complete_ts, EventType.ACTIVITY_COMPLETE, {"task_id": task_id})

        # Trigger Dispatch
        self._push_event(t, EventType.RECHECK, {})

        # Schedule next bucket check
        nxt_boundary = t + pd.Timedelta(hours=self.bucket_hours)
        if nxt_boundary <= self.end_time:
            self._push_event(nxt_boundary, EventType.RESOURCE_CHECK, {})

    # ---------- Resource Allocation Logic ----------
    def _allocate(self, t: pd.Timestamp):
        """
        Matches free resources to waiting/suspended tasks.
        Priority: Suspended Tasks > Waiting Tasks.

        When batch_size == 1 (default) the original one-by-one loop runs
        unchanged.  When batch_size > 1 the call is forwarded to
        _allocate_batch() which accumulates K tasks before dispatching.
        """
        t = pd.Timestamp(t).tz_convert("UTC")

        # ── K-Batch path ───────────────────────────────────────────────────
        if self.batch_size > 1:
            self._allocate_batch(t)
            return

        # ── Park & Song prediction-based allocation ─────────────────────────
        if self.allocation_method == "park_song":
            self._allocate_park_song(t)
            return

        # ── RL policy allocation ────────────────────────────────────────────
        if self.allocation_method == "rl" and self.rl_allocator is not None:
            self._allocate_rl(t)
            return

        # ── Standard one-by-one path (batch_size == 1) ────────────────────

        # Seconds remaining in the current time bucket — the same for every
        # resource because bucket boundaries depend only on simulation time t.
        # Used by shift-aware candidate filtering inside try_queue().
        bucket_end = self._bucket_end(t)
        bucket_remaining_s = max(0.0, float((bucket_end - t).total_seconds()))

        while True:
            # Refresh available pool
            available = set(self.availability.get_available_resources(t))
            free_resources = list(available - self.busy_resources)
            if not free_resources:
                return

            # Helper to process a specific queue
            def try_queue(queue: Deque[Task], resumed: bool) -> bool:
                nonlocal free_resources
                if not queue or not free_resources:
                    return False

                n = len(queue)
                for _ in range(n):
                    if not free_resources:
                        break

                    task = queue.popleft()

                    # Filter candidates by permission
                    candidates = [r for r in free_resources if self.permissions.can_execute(r, task.activity)]
                    if not candidates:
                        queue.append(task)
                        continue

                    # ── Shift-aware candidate filtering ───────────────────
                    # Predict task duration to decide whether any candidate
                    # can finish within the current bucket.
                    # Suspended tasks already carry their remaining_s; for
                    # new tasks we call the duration model with the first
                    # candidate as a resource proxy (cheap median estimate).
                    task_dur_s: Optional[float] = task.remaining_s
                    if task_dur_s is None:
                        attrs = self.case_attrs.get(task.case_id, {})
                        try:
                            if hasattr(self.duration, "sample_duration"):
                                task_dur_s = float(
                                    self.duration.sample_duration(
                                        activity=task.activity,
                                        case_attributes=attrs,
                                        resource=candidates[0],
                                        current_time=t.to_pydatetime(),
                                        method="sample" if self.mode == "advanced" else "median",
                                    )
                                )
                            elif hasattr(self.duration, "predict_duration"):
                                task_dur_s = float(
                                    self.duration.predict_duration(
                                        activity=task.activity,
                                        case_attributes=attrs,
                                        resource=candidates[0],
                                        current_time=t.to_pydatetime(),
                                        quantile=0.5,
                                    )
                                )
                        except Exception:
                            pass  # duration unknown — skip filtering

                    # Prefer candidates that can finish without suspension.
                    # If none qualify (task will span into next bucket), fall
                    # back to all candidates (suspension is unavoidable).
                    if task_dur_s is not None:
                        preferred = [r for r in candidates
                                     if bucket_remaining_s >= task_dur_s]
                        filtered = preferred if preferred else candidates
                    else:
                        filtered = candidates

                    # Select resource; pass executing so R-SHQ can read loads.
                    chosen = self.selector.select(
                        filtered,
                        executing=self.executing,
                    )
                    if chosen is None:
                        queue.append(task)
                        continue

                    # Allocation Successful
                    free_resources.remove(chosen)
                    self._start_or_resume_task(t, task, chosen, resumed=resumed)
                    return True

                return False

            # Strict Priority Check
            did_something = False
            if try_queue(self.suspended, resumed=True):
                did_something = True
            elif try_queue(self.waiting, resumed=False):
                did_something = True

            if not did_something:
                return

    # ---------- K-Batch Allocation Logic ----------
    def _allocate_batch(self, t: pd.Timestamp):
        """
        K-Batch allocation: accumulate batch_size tasks then assign them all
        at once via selector.batch_assign() (cost-matrix approach).

        Flush conditions (either triggers dispatch):
          1. Total pending tasks >= batch_size  — batch is full.
          2. Time since last enqueue >= 1 bucket — timeout; flush partial
             batch to prevent tasks waiting indefinitely when arrivals are
             sparse.

        After each dispatch attempt the enqueue timer resets so the next
        batch window starts fresh.

        Suspended tasks are collected before waiting tasks to preserve the
        engine's existing suspension priority.
        """
        total_pending = len(self.suspended) + len(self.waiting)
        if total_pending == 0:
            return

        # Timeout = 1 time bucket (1 h in basic mode, 2 h in advanced).
        timeout_s = self.bucket_hours * 3600.0
        elapsed_s = (t - self._last_enqueue_ts).total_seconds()

        batch_full = total_pending >= self.batch_size
        timed_out  = elapsed_s >= timeout_s

        if not batch_full and not timed_out:
            return  # Wait — neither condition met yet.

        # Collect batch: suspended first (priority), then waiting.
        batch: List[Task] = list(self.suspended) + list(self.waiting)

        # Get currently free resources.
        available = set(self.availability.get_available_resources(t))
        free_resources = sorted(available - self.busy_resources)
        if not free_resources:
            # All resources busy right now.  Do NOT reset the timer — the
            # flush condition (batch_full or timed_out) is still true and
            # should fire again as soon as a resource becomes free on the
            # next RECHECK, rather than making tasks wait another full
            # timeout period.
            return

        # Reset timer only now that we are actually proceeding with an
        # assignment attempt.  This starts a fresh batch-accumulation window
        # regardless of whether every task gets a real resource or a dummy.
        self._last_enqueue_ts = t

        # Delegate to selector for cost-matrix assignment.
        assignments = self.selector.batch_assign(
            tasks=batch,
            candidates=free_resources,
            duration_model=self.duration,
            case_attrs=self.case_attrs,
            current_time=t,
            mode=self.mode,
            permissions_model=self.permissions,
        )

        if not assignments:
            return

        # Execute assignments and collect UIDs of assigned tasks.
        assigned_uids: Set[str] = set()
        for task, resource in assignments:
            assigned_uids.add(task.uid)
            resumed = task.remaining_s is not None
            self._start_or_resume_task(t, task, resource, resumed=resumed)

        # Remove assigned tasks from their queues (rebuild deques in-place).
        self.suspended = deque(
            tsk for tsk in self.suspended if tsk.uid not in assigned_uids
        )
        self.waiting = deque(
            tsk for tsk in self.waiting if tsk.uid not in assigned_uids
        )

    # ---------- Park & Song (Method A) ----------

    def _predict_upcoming_tasks(self, t: pd.Timestamp) -> List[Dict]:
        """
        For every case currently being executed, predict the *next* activity and
        estimate when it will arrive (= now + remaining processing time).

        Returns a list of dicts with keys:
            case_id, activity, arrival_ts, is_predicted (always True)
        """
        predictions: List[Dict] = []
        for ex in list(self.executing.values()):
            # Skip final activities — no next task will arrive.
            if hasattr(self.bpmn, "is_final") and self.bpmn.is_final(ex.activity):
                continue

            allowed: List[str] = []
            if hasattr(self.bpmn, "allowed_next"):
                allowed = list(
                    self.bpmn.allowed_next(ex.activity, self.case_attrs.get(ex.case_id, {}))
                    or []
                )
            if not allowed:
                continue

            prev2, prev1 = self.case_last2.get(ex.case_id, (None, None))
            if hasattr(self.next_act, "sample_next"):
                predicted_activity = self.next_act.sample_next(
                    prev2, prev1, allowed_next=allowed
                )
            else:
                predicted_activity = allowed[0]

            # Use the correctly adjusted remaining time for the arrival estimate.
            # ex.remaining_s was last updated at ex.slice_start; correct for elapsed.
            elapsed = max(0.0, float((t - ex.slice_start).total_seconds()))
            current_remaining = max(0.0, ex.remaining_s - elapsed)
            estimated_arrival = t + pd.to_timedelta(current_remaining, unit="s")

            predictions.append({
                "case_id": ex.case_id,
                "activity": predicted_activity,
                "arrival_ts": estimated_arrival,
                "is_predicted": True,
            })
        return predictions

    def _allocate_park_song(self, t: pd.Timestamp):
        """
        Park & Song (2019) prediction-based allocation with strategic idling.

        Algorithm:
          1. Release stale reservations (older than 2 h).
          2. Collect real waiting/suspended tasks + predicted upcoming tasks.
          3. Build a combined cost matrix (task × free_resource).
             Real tasks:      c(t,r) = sample_duration(activity, attrs, r)
             Predicted tasks: c(t,r) = sample_duration(predicted, attrs, r) × discount
          4. Add dummy columns when n_tasks > n_free_resources (same formula as K-batch).
          5. Solve with Hungarian algorithm (scipy) or greedy fallback.
          6. Real assignments → _start_or_resume_task().
             Predicted assignments → store reservation in self._reservations;
             add resource to self.busy_resources (strategic idle hold).
        """
        horizon_s = 2.0 * 3600.0  # 2-hour reservation horizon

        # ── 1. Release stale reservations ─────────────────────────────────────
        stale_keys = [
            k for k, ts in self._reservation_ts.items()
            if (t - ts).total_seconds() > horizon_s
        ]
        for k in stale_keys:
            res = self._reservations.pop(k, None)
            self._reservation_ts.pop(k, None)
            if res:
                self.busy_resources.discard(res)

        # ── 2. Collect tasks ───────────────────────────────────────────────────
        real_tasks: List[Task] = list(self.suspended) + list(self.waiting)
        available = set(self.availability.get_available_resources(t))
        free_resources: List[str] = sorted(available - self.busy_resources)

        if not free_resources:
            return

        upcoming = self._predict_upcoming_tasks(t)
        predicted_tasks = [
            p for p in upcoming
            if (p["arrival_ts"] - t).total_seconds() <= horizon_s
            # Avoid double-reserving an already-reserved (case_id, activity) pair.
            and (p["case_id"], p["activity"]) not in self._reservations
        ]

        all_tasks: List[Any] = list(real_tasks) + list(predicted_tasks)
        if not all_tasks:
            return

        # ── scipy unavailable → greedy real-task-only fallback ───────────────
        if not _HAS_SCIPY:
            assignments = self.selector.batch_assign(
                tasks=real_tasks,
                candidates=free_resources,
                duration_model=self.duration,
                case_attrs=self.case_attrs,
                current_time=t,
                mode=self.mode,
                permissions_model=self.permissions,
            )
            if assignments:
                assigned_uids = {task.uid for task, _ in assignments}
                for task, resource in assignments:
                    self._start_or_resume_task(
                        t, task, resource, resumed=(task.remaining_s is not None)
                    )
                self.suspended = deque(
                    tk for tk in self.suspended if tk.uid not in assigned_uids
                )
                self.waiting = deque(
                    tk for tk in self.waiting if tk.uid not in assigned_uids
                )
            return

        # ── 3. Build cost matrix ───────────────────────────────────────────────
        n_tasks = len(all_tasks)
        n_real_res = len(free_resources)
        res_idx = {r: j for j, r in enumerate(free_resources)}
        discount = self._park_song_discount

        cost = np.full((n_tasks, n_real_res), np.inf)

        for i, item in enumerate(all_tasks):
            is_pred = isinstance(item, dict)
            activity = item["activity"] if is_pred else item.activity
            case_id  = item["case_id"]  if is_pred else item.case_id
            attrs    = self.case_attrs.get(case_id, {})

            for r in free_resources:
                if not self.permissions.can_execute(r, activity):
                    continue
                j = res_idx[r]
                try:
                    if hasattr(self.duration, "sample_duration"):
                        dur = float(self.duration.sample_duration(
                            activity=activity,
                            case_attributes=attrs,
                            resource=r,
                            current_time=t.to_pydatetime(),
                            method="sample" if self.mode == "advanced" else "median",
                        ))
                    elif hasattr(self.duration, "predict_duration"):
                        dur = float(self.duration.predict_duration(
                            activity=activity,
                            case_attributes=attrs,
                            resource=r,
                            current_time=t.to_pydatetime(),
                            quantile=0.5,
                        ))
                    else:
                        dur = 60.0
                except Exception:
                    dur = 60.0

                dur = max(1.0, dur)
                cost[i][j] = dur * discount if is_pred else dur

        # ── 4. Dummy columns (n_tasks > n_resources) ──────────────────────────
        all_resources_ext: List[str] = list(free_resources)
        if n_tasks > n_real_res:
            n_dummies = n_tasks - n_real_res
            row_dummy = np.zeros(n_tasks)
            for i in range(n_tasks):
                finite_vals = cost[i][np.isfinite(cost[i])]
                if finite_vals.size > 0:
                    row_dummy[i] = finite_vals.sum() / finite_vals.size
                else:
                    row_dummy[i] = 3600.0
            cost = np.hstack([cost, np.outer(row_dummy, np.ones(n_dummies))])
            for k in range(n_dummies):
                all_resources_ext.append(f"__dummy_{k}__")

        original_cost = cost.copy()

        # ── 5. Solve Hungarian ────────────────────────────────────────────────
        finite_flat = original_cost[np.isfinite(original_cost)]
        M = float(finite_flat.max()) * 1000.0 if finite_flat.size > 0 else 1e9
        cost_hp = np.where(np.isfinite(original_cost), original_cost, M)
        row_ind, col_ind = _ps_hungarian(cost_hp)

        # ── 6. Execute assignments ────────────────────────────────────────────
        assigned_uids: Set[str] = set()
        for i, j in zip(row_ind, col_ind):
            if not np.isfinite(original_cost[i, j]):
                continue  # impermissible pair
            r = all_resources_ext[j]
            if r.startswith("__dummy_"):
                continue  # deferred

            item = all_tasks[i]
            is_pred = isinstance(item, dict)

            if is_pred:
                # Strategic idle: reserve resource for predicted task.
                key = (item["case_id"], item["activity"])
                if key not in self._reservations:
                    self._reservations[key] = r
                    self._reservation_ts[key] = t
                    self.busy_resources.add(r)  # hold — not available for others
            else:
                # Real task: start immediately.
                assigned_uids.add(item.uid)
                self._start_or_resume_task(
                    t, item, r, resumed=(item.remaining_s is not None)
                )

        self.suspended = deque(
            tk for tk in self.suspended if tk.uid not in assigned_uids
        )
        self.waiting = deque(
            tk for tk in self.waiting if tk.uid not in assigned_uids
        )

    # ---------- RL Helpers (Method C) ----------

    def _rl_get_state(self, t: pd.Timestamp) -> np.ndarray:
        """
        Build the RL observation vector at time t.

        Layout (length = 2*|R| + |A|):
          [0 … |R|-1]         δ_r  — 1 if resource r is busy, else 0
          [|R| … 2|R|-1]      η_r  — remaining work (hours, normalised)
          [2|R| … 2|R|+|A|-1] κ_a  — count of waiting tasks of type a
        """
        nR = len(self._all_resources_list)
        nA = len(self._all_activities_list)

        busy_vec      = np.zeros(nR, dtype=np.float32)
        remaining_vec = np.zeros(nR, dtype=np.float32)
        task_vec      = np.zeros(nA, dtype=np.float32)

        r_to_i = {r: i for i, r in enumerate(self._all_resources_list)}
        a_to_i = {a: i for i, a in enumerate(self._all_activities_list)}

        for ex in self.executing.values():
            if ex.resource in r_to_i:
                i = r_to_i[ex.resource]
                busy_vec[i] = 1.0
                elapsed = max(0.0, float((t - ex.slice_start).total_seconds()))
                remaining_vec[i] = max(0.0, ex.remaining_s - elapsed) / 3600.0

        for task in list(self.suspended) + list(self.waiting):
            if task.activity in a_to_i:
                task_vec[a_to_i[task.activity]] += 1.0

        return np.concatenate([busy_vec, remaining_vec, task_vec]).astype(np.float32)

    def _rl_get_action_mask(self, t: pd.Timestamp) -> np.ndarray:
        """
        Binary mask of length |R|*|A|+1 for MaskablePPO.

        mask[r_idx * |A| + a_idx] = 1 iff resource r is free AND authorised
            for activity a AND at least one task of type a is waiting.
        mask[-1] (Postpone) is always 1.
        """
        nR = len(self._all_resources_list)
        nA = len(self._all_activities_list)

        available = set(self.availability.get_available_resources(t))
        free = available - self.busy_resources

        waiting_counts: Dict[str, int] = {}
        for task in list(self.suspended) + list(self.waiting):
            waiting_counts[task.activity] = waiting_counts.get(task.activity, 0) + 1

        mask = np.zeros(nR * nA + 1, dtype=bool)
        mask[-1] = True  # Postpone always valid

        for ri, r in enumerate(self._all_resources_list):
            if r not in free:
                continue
            for ai, a in enumerate(self._all_activities_list):
                if waiting_counts.get(a, 0) > 0 and self.permissions.can_execute(r, a):
                    mask[ri * nA + ai] = True

        return mask

    def _allocate_rl(self, t: pd.Timestamp):
        """
        Execute one assignment decision from the RL policy (inference path).

        Called by _allocate() when allocation_method == "rl" and an
        rl_allocator has been attached to the engine.
        """
        if not self._all_resources_list or not self._all_activities_list:
            return

        nA = len(self._all_activities_list)
        nR = len(self._all_resources_list)

        state = self._rl_get_state(t)
        mask  = self._rl_get_action_mask(t)
        action = int(self.rl_allocator.select_action(state, mask))

        postpone = nR * nA
        if action == postpone:
            return  # strategic postpone

        r_idx = action // nA
        a_idx = action % nA
        resource = self._all_resources_list[r_idx]
        activity = self._all_activities_list[a_idx]

        # Find first matching task (suspended priority).
        task: Optional[Task] = None
        is_resumed = False
        for tk in list(self.suspended):
            if tk.activity == activity:
                task = tk
                is_resumed = True
                break
        if task is None:
            for tk in list(self.waiting):
                if tk.activity == activity:
                    task = tk
                    break

        if task is None or resource in self.busy_resources:
            return
        if not self.permissions.can_execute(resource, activity):
            return

        # Remove from queue.
        if is_resumed:
            self.suspended = deque(tk for tk in self.suspended if tk.uid != task.uid)
        else:
            self.waiting = deque(tk for tk in self.waiting if tk.uid != task.uid)

        self._start_or_resume_task(t, task, resource, resumed=is_resumed)

    # ---------- Step-by-Step Execution (RL training interface) ----------

    def _init_run(self, max_cases: Optional[int] = None) -> None:
        """
        Seed the event queue for step-by-step execution.
        Call once before looping over _step_one_event() in the RL environment.
        """
        self._rl_max_cases = max_cases
        t0 = self.start_time
        self._push_event(t0, EventType.CASE_ARRIVAL, {})
        self._push_event(self._next_bucket_boundary(t0), EventType.RESOURCE_CHECK, {})

    def _step_one_event(self) -> Optional[Tuple[str, pd.Timestamp]]:
        """
        Dequeue and process exactly one event.

        Returns (event_type, timestamp) on success, or None when the
        simulation has ended (empty queue or past end_time).

        In RL mode (allocation_method == "rl"), RECHECK events do NOT
        trigger _allocate() — the external RL environment handles assignment.
        All other modes behave identically to the main run() loop.
        """
        ev = self._pop_event()
        if ev is None:
            return None

        t = ev.ts
        if t > self.end_time:
            return None

        if ev.type == EventType.CASE_ARRIVAL:
            max_c = self._rl_max_cases
            if max_c is None or self.case_counter < max_c:
                case_id = self._new_case_id()
                attrs   = self.attr_sampler.sample()
                self.case_attrs[case_id]  = attrs
                self.case_last2[case_id]  = (None, None)

                if self.allocation_method == "rl":
                    self._rl_case_start[case_id] = t

                first_act = (
                    self.bpmn.start_activity(attrs)
                    if hasattr(self.bpmn, "start_activity")
                    else "A_Create Application"
                )
                self._enqueue_waiting_no_log(t, case_id, first_act)

                t_next = self.arrivals.next_arrival_time(t)
                if pd.Timestamp(t_next).tz_convert("UTC") <= self.end_time:
                    self._push_event(t_next, EventType.CASE_ARRIVAL, {})
                self._push_event(t, EventType.RECHECK, {})

        elif ev.type == EventType.ACTIVITY_COMPLETE:
            self._complete_task(t, ev.payload.get("task_id"))

        elif ev.type == EventType.RESOURCE_CHECK:
            self._resource_check(t)

        elif ev.type == EventType.RECHECK:
            # RL mode: caller decides allocation; non-RL: auto-allocate.
            if self.allocation_method != "rl":
                self._allocate(t)
            self._flush_schedules(t)

        return (ev.type, t)

    # ---------- Export Helper ----------
    def _export_xes(self, df: pd.DataFrame, xes_path: str):
        if pm4py is None:
            print("⚠️ pm4py not installed -> skipping XES export.")
            return

        df2 = df.copy()
        df2["time:timestamp"] = pd.to_datetime(df2["time:timestamp"], utc=True, errors="coerce")
        df2 = df2.dropna(subset=["time:timestamp"]).copy()

        df2 = pm4py.format_dataframe(
            df2,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )

        log = pm4py.convert_to_event_log(df2)
        pm4py.write_xes(log, xes_path)
        print("✅ wrote XES:", xes_path)

    # ---------- Main Run Loop ----------
    def run(self, max_cases: Optional[int] = None) -> str:
        """
        Executes the simulation loop from start_time to end_time.
        Returns the path to the generated CSV log.
        """
        t0 = self.start_time

        # Initialize schedule
        self._push_event(t0, EventType.CASE_ARRIVAL, {})
        self._push_event(self._next_bucket_boundary(t0), EventType.RESOURCE_CHECK, {})

        while True:
            ev = self._pop_event()
            if ev is None:
                break

            t = ev.ts
            if t > self.end_time:
                break

            # 1. New Case Arrival
            if ev.type == EventType.CASE_ARRIVAL:
                if max_cases is not None and self.case_counter >= max_cases:
                    continue

                case_id = self._new_case_id()
                attrs = self.attr_sampler.sample()

                self.case_attrs[case_id] = attrs
                self.case_last2[case_id] = (None, None)

                if hasattr(self.bpmn, "start_activity"):
                    first_act = self.bpmn.start_activity(attrs)
                else:
                    first_act = "A_Create Application"

                self._enqueue_waiting_no_log(t, case_id, first_act)

                # Schedule next arrival
                t_next = self.arrivals.next_arrival_time(t)
                if pd.Timestamp(t_next).tz_convert("UTC") <= self.end_time:
                    self._push_event(t_next, EventType.CASE_ARRIVAL, {})

                # Trigger dispatch
                self._push_event(t, EventType.RECHECK, {})

            # 2. Activity Completion
            elif ev.type == EventType.ACTIVITY_COMPLETE:
                self._complete_task(t, ev.payload.get("task_id"))

            # 3. Resource Shift Check
            elif ev.type == EventType.RESOURCE_CHECK:
                self._resource_check(t)

            # 4. Re-Allocation
            elif ev.type == EventType.RECHECK:
                self._allocate(t)
                # Now log schedule only for tasks that are truly waiting
                self._flush_schedules(t)

        # Final Export
        out_csv = Path(self.out_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        df_out = pd.DataFrame(self.log_rows)
        df_out.to_csv(out_csv, index=False)
        print("✅ wrote CSV:", str(out_csv))

        if self.out_xes_path:
            out_xes = Path(self.out_xes_path)
            out_xes.parent.mkdir(parents=True, exist_ok=True)
            self._export_xes(df_out, str(out_xes))

        return str(out_csv)


# Simple BPMN Mock for testing without XML
class SimpleBPMN:
    def __init__(self, edges: Dict[str, List[str]], start: str, finals: Optional[Set[str]] = None):
        self.edges = edges
        self.start = start
        self.finals = finals or set()

    def start_activity(self, case_attrs: Dict[str, Any]) -> str:
        return self.start

    def allowed_next(self, current_activity: str, case_attrs: Dict[str, Any]) -> List[str]:
        return list(self.edges.get(current_activity, []))

    def is_final(self, activity: str) -> bool:
        return activity in self.finals