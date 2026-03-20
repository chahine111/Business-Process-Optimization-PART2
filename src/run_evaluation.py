"""
src/run_evaluation.py

Part A: runs all 5 allocation configs for 2016 and compares metrics.
Part B: identifies and removes the 2 least-contributing employees, re-runs baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Project imports
from arrival_model_1_2 import ArrivalProcess
from resource_availability_1_5 import ResourceAvailabilityModel
from permissions_model_1_6 import PermissionsModel
from resource_selector_1_7 import ResourceSelector
from processing_time_predictor import ProcessingTimePredictor
from simulation_engine_1_1 import SimulationEngine
from bpmn_adapter import BPMNAdapter
from run_simulation import NextActivityPredictor, train_if_missing
from evaluation import (
    compute_all_metrics,
    compute_resource_occupation,
    load_log,
    print_comparison_table,
)


# ══════════════════════════════════════════════════════════════════════════
# Availability Wrappers (Workforce Scenarios)
# ══════════════════════════════════════════════════════════════════════════

class ExcludeResourcesAvailability:
    """
    Thin wrapper around ResourceAvailabilityModel that removes specific
    resources from every get_available_resources() response.

    Used for the "fire employees" scenario: wrapping the base availability
    model with this class effectively removes the excluded resources from
    the simulation without touching any core module.
    """

    def __init__(self, base_model: Any, exclude: Set[str]) -> None:
        self._base    = base_model
        self._exclude = frozenset(exclude)

    def get_available_resources(self, t: pd.Timestamp):
        """Return available resources, minus the excluded set."""
        return [
            r for r in self._base.get_available_resources(t)
            if r not in self._exclude
        ]


class NineToFiveAvailability:
    """
    Thin wrapper that restricts all resources to work only between
    09:00 and 17:00 local time on weekdays (Monday–Friday).

    Used for the "reduce to nine-to-five" schedule scenario.
    Hours outside this window return an empty list.
    """

    def __init__(self, base_model: Any, tz: str = "Europe/Amsterdam") -> None:
        self._base = base_model
        self._tz   = tz

    def get_available_resources(self, t: pd.Timestamp):
        """Return available resources only during 09:00–17:00 Mon–Fri."""
        t_local = pd.Timestamp(t).tz_convert(self._tz)
        if t_local.weekday() < 5 and 9 <= t_local.hour < 17:
            return self._base.get_available_resources(t)
        return []


# ══════════════════════════════════════════════════════════════════════════
# Engine Factory
# ══════════════════════════════════════════════════════════════════════════

# Simulation configs — mirrors run_simulation.py but skips "rl" (needs
# a pre-trained policy that may not exist yet).
_CONFIGS = {
    "r_rma":     dict(strategy="random",         batch_size=1, allocation_method="standard"),
    "r_rra":     dict(strategy="round_robin",    batch_size=1, allocation_method="standard"),
    "r_shq":     dict(strategy="shortest_queue", batch_size=1, allocation_method="standard"),
    "kbatch":    dict(strategy="random",         batch_size=5, allocation_method="standard"),
    "park_song": dict(strategy="random",         batch_size=1, allocation_method="park_song"),
}

SIM_START = "2016-01-01 00:00:00+00:00"
SIM_END   = "2017-01-01 00:00:00+00:00"


def _build_engine(
    *,
    project_root: Path,
    config: str,
    out_csv: str,
    mode:    str = "advanced",
    availability_override: Any = None,
) -> SimulationEngine:
    """
    Construct a SimulationEngine for *config*, ready to call .run().

    Parameters
    ----------
    availability_override : Optional availability model wrapper.
        If provided it replaces the standard ResourceAvailabilityModel.
        Used for workforce scenarios (ExcludeResourcesAvailability,
        NineToFiveAvailability, etc.).
    """
    data_dir   = project_root / "data"
    models_dir = project_root / "models"

    _csv = data_dir / "bpi2017.csv"
    if not _csv.exists():
        _csv = project_root / "bpi2017.csv"
    csv_path  = str(_csv)
    bpmn_path = str(data_dir / "Signavio_Model.bpmn")

    # Cached artifact paths — identical to run_simulation.py.
    arrivals_cache     = str(models_dir / "arrival_model_1_2.pkl")
    availability_cache = str(models_dir / f"availability_{mode}_1_5.pkl")
    permissions_cache  = str(models_dir / f"permissions_{mode}_1_6.pkl")
    model_13_path      = str(models_dir / "processing_model_advanced.pkl")
    model_14_path      = str(models_dir / "next_activity_bigram_model.pkl")

    cfg = _CONFIGS[config]

    # Availability: use the override if provided, otherwise the standard model.
    availability = (
        availability_override
        if availability_override is not None
        else ResourceAvailabilityModel(
            csv_path=csv_path, mode=mode, cache_path=availability_cache
        )
    )

    return SimulationEngine(
        bpmn               = BPMNAdapter(bpmn_path),
        arrival_process    = ArrivalProcess(
                                 csv_path=csv_path, seed=42,
                                 cache_path=arrivals_cache,
                             ),
        duration_model     = ProcessingTimePredictor(model_path=model_13_path),
        next_activity_model= NextActivityPredictor(model_path=model_14_path, seed=42),
        availability_model = availability,
        permissions_model  = PermissionsModel(
                                 csv_path=csv_path, mode=mode,
                                 cache_path=permissions_cache,
                             ),
        selector           = ResourceSelector(strategy=cfg["strategy"], seed=42),
        mode               = mode,
        start_time         = SIM_START,
        end_time           = SIM_END,
        out_csv_path       = out_csv,
        seed               = 42,
        batch_size         = cfg["batch_size"],
        allocation_method  = cfg["allocation_method"],
    )


# ══════════════════════════════════════════════════════════════════════════
# Part A — Method Comparison
# ══════════════════════════════════════════════════════════════════════════

def run_part_a(project_root: Path, outputs_dir: Path) -> Dict[str, Dict]:
    """
    Run simulation for each configuration and compute evaluation metrics.

    Skips re-running if the output CSV already exists (fast re-evaluation
    after code changes to evaluation.py without re-simulating).

    Returns
    -------
    dict mapping config_name → metrics_dict (from compute_all_metrics).
    """
    results: Dict[str, Dict] = {}
    t_start = pd.Timestamp(SIM_START)
    t_end   = pd.Timestamp(SIM_END)

    for config in _CONFIGS:
        out_csv = str(outputs_dir / f"eval_{config}.csv")

        if Path(out_csv).exists():
            print(f"  [SKIP] {config:<12} — log exists, loading metrics only.")
        else:
            print(f"  [RUN ] {config:<12} — running simulation …")
            engine = _build_engine(
                project_root=project_root,
                config=config,
                out_csv=out_csv,
            )
            engine.run()
            print(f"         → saved to {out_csv}")

        results[config] = compute_all_metrics(
            out_csv, sim_start=t_start, sim_end=t_end
        )

    return results


# ══════════════════════════════════════════════════════════════════════════
# Part B — Workforce Optimization: Fire Two Employees
# ══════════════════════════════════════════════════════════════════════════

def _identify_fire_candidates(
    baseline_csv: str,
    n: int = 2,
    sim_start: Optional[pd.Timestamp] = None,
    sim_end:   Optional[pd.Timestamp] = None,
) -> list:
    """
    Identify the n resources that contribute least to the process.

    Scoring: busy_h × (n_tasks + 1).
    Resources with few hours worked AND few tasks handled are the
    safest to remove — their absence will have the smallest impact
    on cycle time and throughput.

    Returns a list of (resource_name, busy_h, occupation, n_tasks) tuples
    sorted from most- to least-expendable (lowest score first).
    """
    df = load_log(baseline_csv)

    if sim_start is None:
        sim_start = df["time:timestamp"].min()
    if sim_end is None:
        sim_end = df["time:timestamp"].max()

    occ_df = compute_resource_occupation(df, sim_start, sim_end)

    # Count how many tasks each resource completed.
    task_counts = (
        df[df["lifecycle:transition"] == "complete"]
        .groupby("org:resource")
        .size()
        .rename("n_tasks")
        .reset_index()
    )
    occ_df = occ_df.merge(
        task_counts, left_on="resource", right_on="org:resource", how="left"
    )
    occ_df["n_tasks"] = occ_df["n_tasks"].fillna(0).astype(int)

    # Composite score: lower = more expendable.
    occ_df["score"] = occ_df["busy_h"] * (occ_df["n_tasks"] + 1)
    candidates = occ_df.nsmallest(n, "score")

    return [
        (row["resource"], row["busy_h"], row["occupation"], row["n_tasks"])
        for _, row in candidates.iterrows()
    ]


def run_part_b(project_root: Path, outputs_dir: Path) -> None:
    """
    Identify and remove the two least-contributing employees and measure
    the impact on all six evaluation metrics.

    Requires Part A to have run first (uses the r_rma baseline log).
    """
    baseline_csv = str(outputs_dir / "eval_r_rma.csv")
    if not Path(baseline_csv).exists():
        print("  [WARN] Baseline log not found. Run Part A first.")
        return

    t_start = pd.Timestamp(SIM_START)
    t_end   = pd.Timestamp(SIM_END)

    # ── Identify candidates ──────────────────────────────────────────
    candidates = _identify_fire_candidates(
        baseline_csv, n=2, sim_start=t_start, sim_end=t_end
    )

    fire_set: Set[str] = set()
    print("\n  Candidates identified for removal (ranked, lowest contribution first):")
    print(f"  {'Resource':<42} {'Busy (h)':>9} {'Occ %':>7} {'Tasks':>7}")
    print(f"  {'─'*42} {'─'*9} {'─'*7} {'─'*7}")
    for rank, (name, busy_h, occ, n_tasks) in enumerate(candidates, 1):
        print(
            f"  [{rank}] {name:<40} {busy_h:>9.1f}"
            f" {occ * 100:>6.1f}%  {n_tasks:>6}"
        )
        fire_set.add(name)

    # ── Re-run baseline without the two removed employees ────────────
    out_csv_fired = str(outputs_dir / "eval_r_rma_fired2.csv")
    if Path(out_csv_fired).exists():
        print(f"\n  [SKIP] Fired-employees log already exists.")
    else:
        print(f"\n  [RUN ] Re-running r_rma without {fire_set} …")
        _csv_b = project_root / "data" / "bpi2017.csv"
        if not _csv_b.exists():
            _csv_b = project_root / "bpi2017.csv"
        base_avail = ResourceAvailabilityModel(
            csv_path=str(_csv_b),
            mode="advanced",
            cache_path=str(project_root / "models" / "availability_advanced_1_5.pkl"),
        )
        fired_avail = ExcludeResourcesAvailability(base_avail, fire_set)
        engine = _build_engine(
            project_root=project_root,
            config="r_rma",
            out_csv=out_csv_fired,
            availability_override=fired_avail,
        )
        engine.run()

    # ── Compare metrics ───────────────────────────────────────────────
    metrics_before = compute_all_metrics(baseline_csv,    sim_start=t_start, sim_end=t_end)
    metrics_after  = compute_all_metrics(out_csv_fired,   sim_start=t_start, sim_end=t_end)

    print_comparison_table({
        "Baseline (r_rma, all staff)": metrics_before,
        f"Fired {len(fire_set)} resources":  metrics_after,
    })

    # ── Metric delta summary ──────────────────────────────────────────
    ct_delta   = metrics_after["avg_cycle_time_h"]   - metrics_before["avg_cycle_time_h"]
    occ_delta  = metrics_after["avg_resource_occ"]   - metrics_before["avg_resource_occ"]
    tput_delta = metrics_after["throughput_per_day"] - metrics_before["throughput_per_day"]

    print("  Impact summary after removing two employees:")
    print(f"    Avg Cycle Time :  {ct_delta:+.2f} h"
          f"  ({'worsened ↑' if ct_delta > 0 else 'improved ↓'})")
    print(f"    Avg Occupation :  {occ_delta * 100:+.1f} pp"
          f"  ({'higher utilization' if occ_delta > 0 else 'lower utilization'})")
    print(f"    Throughput     :  {tput_delta:+.2f} cases/day"
          f"  ({'lower ↓' if tput_delta < 0 else 'unchanged/higher'})")
    print()


# ══════════════════════════════════════════════════════════════════════════
# Optional Part B' — Nine-to-Five Schedule Scenario
# ══════════════════════════════════════════════════════════════════════════

def run_nine_to_five(project_root: Path, outputs_dir: Path) -> None:
    """
    Evaluate the impact of restricting all resource availability to a
    strict 09:00–17:00 Monday–Friday schedule.

    Answers the management question:
        "Management wants you to reduce the working hours to nine to five!"
        "How would you choose the working schedule?"
        "What is the impact on your metrics?"

    Call this function from main() to run the scenario.  Not executed
    by default because it runs a full-year simulation.
    """
    baseline_csv = str(outputs_dir / "eval_r_rma.csv")
    if not Path(baseline_csv).exists():
        print("  [WARN] Baseline log not found. Run Part A first.")
        return

    t_start = pd.Timestamp(SIM_START)
    t_end   = pd.Timestamp(SIM_END)

    out_csv_95 = str(outputs_dir / "eval_r_rma_nine_to_five.csv")
    if Path(out_csv_95).exists():
        print("  [SKIP] Nine-to-five log already exists.")
    else:
        print("  [RUN ] Re-running r_rma with 09:00–17:00 Mon–Fri schedule …")
        base_avail = ResourceAvailabilityModel(
            csv_path=str(project_root / "data" / "bpi2017.csv"),
            mode="advanced",
            cache_path=str(project_root / "models" / "availability_advanced_1_5.pkl"),
        )
        avail_95 = NineToFiveAvailability(base_avail)
        engine = _build_engine(
            project_root=project_root,
            config="r_rma",
            out_csv=out_csv_95,
            availability_override=avail_95,
        )
        engine.run()

    metrics_before = compute_all_metrics(baseline_csv, sim_start=t_start, sim_end=t_end)
    metrics_after  = compute_all_metrics(out_csv_95,   sim_start=t_start, sim_end=t_end)

    print_comparison_table({
        "Baseline (r_rma, real schedule)": metrics_before,
        "Nine-to-five schedule":           metrics_after,
    })


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / "data"
    outputs_dir  = project_root / "outputs"
    models_dir   = project_root / "models"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # bpi2017.csv may live in data/ or at project root (both locations checked).
    csv_path = data_dir / "bpi2017.csv"
    if not csv_path.exists():
        csv_path = project_root / "bpi2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing data file: expected at data/bpi2017.csv or project root."
        )

    # Ensure all sub-models are trained before simulating.
    train_if_missing(project_root, csv_path)

    # ── Part A: run all configs and compare ─────────────────────────────
    print("\n" + "=" * 70)
    print("PART A — Resource Allocation Method Comparison (2016 full year)")
    print("=" * 70)

    results = run_part_a(project_root, outputs_dir)
    print_comparison_table(results)

    # ── Part B: workforce optimization — fire two employees ─────────────
    print("=" * 70)
    print("PART B — Workforce Optimization: 'Fire Two Employees'")
    print("=" * 70)

    run_part_b(project_root, outputs_dir)

    # ── Optional Part B': nine-to-five schedule analysis ────────────────
    # Uncomment the lines below to also run the nine-to-five scenario.
    # print("=" * 70)
    # print("PART B' — Schedule Scenario: Nine-to-Five")
    # print("=" * 70)
    # run_nine_to_five(project_root, outputs_dir)


if __name__ == "__main__":
    main()
