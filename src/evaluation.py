"""
src/evaluation.py

Task 1.2 — Evaluation Metrics for Resource Allocation.

Computes six metrics from a simulated event log CSV produced by
SimulationEngine.run().  The module is fully standalone — it only
reads the output CSV and requires no simulation components.

Basic Metrics (required by the assignment):
    1. Average Cycle Time (hours)
       Mean end-to-end duration per completed case, measured from the
       case's first log event to its last 'complete' lifecycle transition.
    2. Average Resource Occupation (0–1)
       For each resource: busy_time / simulation_window.  Averaged
       across all resources that appear in the log.
    3. (Weighted) Resource Fairness
       Mean absolute deviation of per-resource occupation from the
       group mean.  Weighted by each resource's share of total busy time
       so that heavily-used resources are penalised more for imbalance.
       Lower = fairer (0 = perfectly equal load).

Advanced Metrics (additional contributions):
    4. P90 Cycle Time (hours)
       90th-percentile cycle time — captures tail / worst-case behaviour
       that the mean obscures.
    5. Average Waiting Time (hours)
       Mean queue wait per task: time from 'schedule' to 'start'/'resume'.
       Only tasks that actually experienced a wait are included.
    6. Throughput (cases / day)
       Number of fully completed cases divided by the simulation window
       in days.  Higher = more productive.

Usage:
    from evaluation import compute_all_metrics, print_comparison_table

    m = compute_all_metrics("outputs/simulated_log.csv")
    print_comparison_table({"r_rma": m})
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Log Loading
# ----------------------------

def load_log(csv_path: str) -> pd.DataFrame:
    """
    Load a simulated event log CSV and normalise key columns.

    Expected columns (minimum):
        time:timestamp, case:concept:name, concept:name,
        lifecycle:transition, org:resource
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Parse timestamps — simulation engine writes UTC ISO strings.
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["time:timestamp"]).copy()

    # Normalise text columns to lower-case, stripped strings.
    df["lifecycle:transition"] = (
        df["lifecycle:transition"].astype(str).str.strip().str.lower()
    )
    if "org:resource" in df.columns:
        df["org:resource"] = df["org:resource"].astype(str).str.strip()

    return df


# ----------------------------
# Internal — Resource Busy Time
# ----------------------------

def _compute_resource_busy_seconds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute total execution time (seconds) per resource.

    Algorithm (one pass per resource, events sorted by timestamp):
        ┌─────────────────────────────────────────────────────────────────┐
        │  'start' or 'resume' → open a new work segment (seg_start = t) │
        │  If a segment was already open (e.g. from a suspension that     │
        │  was immediately followed by new/resumed work) close it first:  │
        │      busy += t_current − t_seg_open                             │
        │  'complete' → close the current segment:                        │
        │      busy += t_complete − t_seg_open; seg_start = None          │
        └─────────────────────────────────────────────────────────────────┘

    This correctly handles tasks that span bucket boundaries.  In the
    simulation, a suspended task is re-queued and resumed at the very next
    RESOURCE_CHECK event (effectively zero gap), so treating the suspension
    point as the close of one segment and the resume point as the open of
    the next produces accurate totals.
    """
    work_df = df[
        df["lifecycle:transition"].isin(["start", "resume", "complete"])
    ].sort_values("time:timestamp")

    busy: Dict[str, float] = {}

    for resource, grp in work_df.groupby("org:resource"):
        grp = grp.sort_values("time:timestamp")
        total_s = 0.0
        seg_start: Optional[pd.Timestamp] = None

        for _, row in grp.iterrows():
            trans = row["lifecycle:transition"]
            ts    = row["time:timestamp"]

            if trans in ("start", "resume"):
                if seg_start is not None:
                    # Close previous segment before opening a new one.
                    total_s += (ts - seg_start).total_seconds()
                seg_start = ts
            elif trans == "complete" and seg_start is not None:
                total_s += (ts - seg_start).total_seconds()
                seg_start = None

        busy[resource] = max(0.0, total_s)

    return busy


# ----------------------------
# Metric 1 & 4 — Cycle Time
# ----------------------------

def compute_cycle_times(df: pd.DataFrame) -> pd.Series:
    """
    Compute end-to-end cycle time (hours) for every completed case.

    A case is 'complete' when it has at least one 'complete' lifecycle
    event.  Cycle time = max(complete_ts) − min(any_ts in case).

    Returns a Series indexed by case ID, values in hours.
    """
    case_col = "case:concept:name"

    complete_mask = df["lifecycle:transition"] == "complete"
    if not complete_mask.any():
        return pd.Series(dtype=float, name="cycle_time_h")

    # Latest completion per case → case end.
    case_end   = df[complete_mask].groupby(case_col)["time:timestamp"].max()
    # Earliest event per case → case start (includes schedule events).
    case_start = df.groupby(case_col)["time:timestamp"].min()

    # Only include cases observed from start to finish.
    common = case_end.index.intersection(case_start.index)
    cycle_s = (case_end[common] - case_start[common]).dt.total_seconds()
    return (cycle_s / 3600.0).rename("cycle_time_h")


# ----------------------------
# Metric 2 — Resource Occupation
# ----------------------------

def compute_resource_occupation(
    df: pd.DataFrame,
    sim_start: Optional[pd.Timestamp] = None,
    sim_end:   Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute per-resource occupation fraction over the simulation window.

    occupation_r = busy_time_r / window_seconds

    where window = sim_end − sim_start (total simulation wall-clock length).
    If sim_start / sim_end are not supplied they are derived from the log.

    Returns a DataFrame with columns:
        resource   — resource identifier
        busy_h     — total hours the resource spent executing work
        occupation — fraction 0–1 (capped at 1.0 for robustness)
    """
    if sim_start is None:
        sim_start = df["time:timestamp"].min()
    if sim_end is None:
        sim_end = df["time:timestamp"].max()

    window_s = max(1.0, (sim_end - sim_start).total_seconds())

    busy_s = _compute_resource_busy_seconds(df)

    records = [
        {
            "resource":   r,
            "busy_h":     bs / 3600.0,
            "occupation": min(1.0, bs / window_s),
        }
        for r, bs in busy_s.items()
        if r and r.lower() != "nan"
    ]
    return (
        pd.DataFrame(records)
        .sort_values("resource")
        .reset_index(drop=True)
    )


# ----------------------------
# Metric 3 — Resource Fairness
# ----------------------------

def compute_fairness(occ_df: pd.DataFrame, weighted: bool = True) -> float:
    """
    Compute resource fairness as the mean absolute deviation of occupation.

    Unweighted: mean(|occ_r − mean_occ|) for all resources r.
    Weighted  : weighted mean of deviations, where each resource is
                weighted by its share of total busy time.

    Lower is fairer (0 = perfectly equal load distribution).
    """
    if occ_df.empty or len(occ_df) < 2:
        return 0.0

    occ  = occ_df["occupation"].values
    mu   = float(np.mean(occ))
    devs = np.abs(occ - mu)

    if weighted and "busy_h" in occ_df.columns:
        weights = occ_df["busy_h"].values
        total_w = float(weights.sum())
        if total_w > 0:
            return float(np.average(devs, weights=weights))

    return float(np.mean(devs))


# ----------------------------
# Metric 5 — Waiting Time
# ----------------------------

def compute_waiting_times(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-task waiting time (hours): schedule → start/resume gap.

    Only tasks that experienced a wait (had a 'schedule' event logged
    before their 'start'/'resume') are included in the result.

    Logic:
        1. Collect all 'schedule' events (case, activity, sched_ts).
        2. Collect all 'start'/'resume' events (case, activity, start_ts).
        3. Join on (case, activity); keep pairs where start_ts ≥ sched_ts.
        4. For each schedule event, pick the earliest subsequent start.
    """
    case_col = "case:concept:name"
    act_col  = "concept:name"

    sched = df[df["lifecycle:transition"] == "schedule"].copy()
    start = df[df["lifecycle:transition"].isin(["start", "resume"])].copy()

    if sched.empty or start.empty:
        return pd.Series(dtype=float, name="waiting_time_h")

    sched = sched.rename(columns={"time:timestamp": "sched_ts"})
    start = start.rename(columns={"time:timestamp": "start_ts"})

    merged = sched[[case_col, act_col, "sched_ts"]].merge(
        start[[case_col, act_col, "start_ts"]],
        on=[case_col, act_col],
        how="inner",
    )

    # Retain only valid pairs (start must follow schedule).
    merged = merged[merged["start_ts"] >= merged["sched_ts"]]

    # For each unique schedule event, pick the earliest matching start.
    merged = merged.sort_values("start_ts")
    merged = merged.drop_duplicates(
        subset=[case_col, act_col, "sched_ts"], keep="first"
    )

    wait_h = (merged["start_ts"] - merged["sched_ts"]).dt.total_seconds() / 3600.0
    return wait_h.rename("waiting_time_h").reset_index(drop=True)


# ----------------------------
# Aggregated Metric Bundle
# ----------------------------

def compute_all_metrics(
    csv_path:  str,
    sim_start: Optional[pd.Timestamp] = None,
    sim_end:   Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    Compute all six evaluation metrics from a simulated event log CSV.

    Parameters
    ----------
    csv_path  : path to SimulationEngine output CSV.
    sim_start : simulation start timestamp (derived from log if None).
    sim_end   : simulation end timestamp   (derived from log if None).

    Returns
    -------
    dict with keys:
        avg_cycle_time_h    — Metric 1: mean cycle time (hours)
        p90_cycle_time_h    — Metric 4: 90th-percentile cycle time (hours)
        avg_resource_occ    — Metric 2: mean resource occupation (0–1)
        fairness            — Metric 3: weighted resource fairness (lower = fairer)
        avg_waiting_time_h  — Metric 5: mean task queue wait (hours)
        throughput_per_day  — Metric 6: completed cases per day
        n_cases             — number of completed cases in the log
        n_resources         — number of distinct resources in the log
    """
    df = load_log(csv_path)

    if sim_start is not None:
        sim_start = pd.Timestamp(sim_start).tz_convert("UTC")
    if sim_end is not None:
        sim_end = pd.Timestamp(sim_end).tz_convert("UTC")

    # ── Metric 1 & 4: Cycle Time ──────────────────────────────────────
    ct = compute_cycle_times(df)
    avg_ct = float(ct.mean())    if len(ct) > 0 else float("nan")
    p90_ct = float(ct.quantile(0.90)) if len(ct) > 0 else float("nan")

    # ── Metric 2 & 3: Occupation + Fairness ──────────────────────────
    occ_df  = compute_resource_occupation(df, sim_start, sim_end)
    avg_occ = float(occ_df["occupation"].mean()) if len(occ_df) > 0 else float("nan")
    fair    = compute_fairness(occ_df, weighted=True)

    # ── Metric 5: Waiting Time ────────────────────────────────────────
    wt     = compute_waiting_times(df)
    avg_wt = float(wt.mean()) if len(wt) > 0 else float("nan")

    # ── Metric 6: Throughput ──────────────────────────────────────────
    n_cases  = int(len(ct))
    t_start  = sim_start or df["time:timestamp"].min()
    t_end    = sim_end   or df["time:timestamp"].max()
    days     = max(1.0, (t_end - t_start).total_seconds() / 86400.0)
    throughput = n_cases / days

    return {
        "avg_cycle_time_h":   avg_ct,
        "p90_cycle_time_h":   p90_ct,
        "avg_resource_occ":   avg_occ,
        "fairness":           fair,
        "avg_waiting_time_h": avg_wt,
        "throughput_per_day": throughput,
        "n_cases":            n_cases,
        "n_resources":        int(len(occ_df)),
    }


# ----------------------------
# Display Helpers
# ----------------------------

# Human-readable labels for the metric keys.
_METRIC_LABELS: Dict[str, str] = {
    "avg_cycle_time_h":   "Avg Cycle Time (h)        [↓ better]",
    "p90_cycle_time_h":   "P90 Cycle Time (h)        [↓ better]",
    "avg_resource_occ":   "Avg Resource Occ (%)      [context]",
    "fairness":           "Resource Fairness         [↓ fairer]",
    "avg_waiting_time_h": "Avg Waiting Time (h)      [↓ better]",
    "throughput_per_day": "Throughput (cases/day)    [↑ better]",
    "n_cases":            "Completed Cases",
    "n_resources":        "Active Resources",
}


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted side-by-side comparison table of metrics.

    Parameters
    ----------
    results : mapping config_name → metric_dict (from compute_all_metrics).
    """
    configs = list(results.keys())
    metrics = list(_METRIC_LABELS.keys())

    col_w  = 22
    lbl_w  = 38
    header = f"{'Metric':<{lbl_w}}" + "".join(f"{c:>{col_w}}" for c in configs)
    sep    = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for m in metrics:
        label = _METRIC_LABELS[m]
        row   = f"{label:<{lbl_w}}"
        for c in configs:
            val = results[c].get(m, float("nan"))
            if m == "avg_resource_occ":
                # Show as percentage
                row += f"{val * 100:>{col_w}.2f}"
            elif m in ("n_cases", "n_resources"):
                row += f"{int(val):>{col_w}}"
            else:
                row += f"{val:>{col_w}.3f}"
        print(row)

    print(sep + "\n")
