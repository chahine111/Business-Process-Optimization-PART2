"""
src/run_simulation.py

Trains models if missing, then runs one simulation config and saves CSV + XES.
Edit the `config` variable near the bottom to switch allocation strategy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np

from arrival_model_1_2 import ArrivalProcess
from resource_availability_1_5 import ResourceAvailabilityModel
from permissions_model_1_6 import PermissionsModel
from resource_selector_1_7 import ResourceSelector
from processing_time_predictor import ProcessingTimePredictor
from simulation_engine_1_1 import SimulationEngine
from bpmn_adapter import BPMNAdapter

from next_activity_TRAIN_1_4 import load_log as train14_load_log, train_next_activity_model
from processing_times_TRAIN import train_processing_model


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
            return str(self.rng.choice(list(allowed_set)))

        next_f = [n for n, m in zip(next_list, mask) if m]
        prob_f = prob[mask]
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
        # bigram → unigram → global fallback
        if prev2 is not None and prev1 is not None:
            key = (prev2, prev1)
            if key in self.model.get("bigram", {}):
                info = self.model["bigram"][key]
                return self._sample_from(info["next"], info["prob"], allowed_next)

        if prev1 is not None and prev1 in self.model.get("unigram", {}):
            info = self.model["unigram"][prev1]
            return self._sample_from(info["next"], info["prob"], allowed_next)

        info = self.model["global"]
        return self._sample_from(info["next"], info["prob"], allowed_next)


def train_if_missing(project_root: Path, csv_path: Path) -> None:
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_13 = models_dir / "processing_model_advanced.pkl"
    model_14 = models_dir / "next_activity_bigram_model.pkl"

    if not model_14.exists():
        print("\n[TRAIN] 1.4 (Next Activity) model missing -> training now...")

        df = train14_load_log(xes_path=None, csv_path=str(csv_path))
        model, c_bigram, c_uni, c_glob = train_next_activity_model(df)

        joblib.dump(model, model_14)
        print(f"[TRAIN] Saved 1.4 model -> {model_14}")

    else:
        print("\n[SKIP] 1.4 model exists -> using cached pickle")

    if not model_13.exists():
        print("\n[TRAIN] 1.3 (Processing Time) model missing -> training now...")

        train_processing_model(
            log_path=str(csv_path),
            model_path=str(model_13),
        )

        if not model_13.exists():
            raise FileNotFoundError("1.3 training finished but output file not found.")

        print(f"[TRAIN] Saved 1.3 model -> {model_13}")

    else:
        print("\n[SKIP] 1.3 model exists -> using cached pickle")


# ----------------------------
# Main Execution
# ----------------------------
def main():
    project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / "data"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "bpi2017.csv"
    bpmn_path = data_dir / "Signavio_Model.bpmn"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    if not bpmn_path.exists():
        raise FileNotFoundError(f"Missing BPMN model: {bpmn_path}")

    train_if_missing(project_root, csv_path)

    model_13_path = str(models_dir / "processing_model_advanced.pkl")
    model_14_path = str(models_dir / "next_activity_bigram_model.pkl")

    # change this to switch allocation strategy
    # options: r_rma, r_rra, r_shq, kbatch, park_song, rl
    # (rl requires models/rl_policy.zip — run rl_train.py first)
    config = "r_rma"

    _configs = {
        "r_rma":      dict(strategy="random",         batch_size=1, allocation_method="standard"),
        "r_rra":      dict(strategy="round_robin",    batch_size=1, allocation_method="standard"),
        "r_shq":      dict(strategy="shortest_queue", batch_size=1, allocation_method="standard"),
        "kbatch":     dict(strategy="random",         batch_size=5, allocation_method="standard"),
        "park_song":  dict(strategy="random",         batch_size=1, allocation_method="park_song"),
        "rl":         dict(strategy="random",         batch_size=1, allocation_method="rl"),
    }
    if config not in _configs:
        raise ValueError(f"Unknown config '{config}'. Choose from: {list(_configs)}")

    selector_strategy = _configs[config]["strategy"]
    batch_size        = _configs[config]["batch_size"]
    allocation_method = _configs[config]["allocation_method"]

    mode = "advanced"
    start_time = "2016-01-01 00:00:00+00:00"
    end_time = "2017-01-01 00:00:00+00:00"  # Full Year 2016

    out_csv = str(outputs_dir / f"simulated_log_{mode}_{config}_2016.csv")
    out_xes = str(outputs_dir / f"simulated_log_{mode}_{config}_2016.xes")

    arrivals_cache = str(models_dir / "arrival_model_1_2.pkl")
    availability_cache = str(models_dir / f"availability_{mode}_1_5.pkl")
    permissions_cache = str(models_dir / f"permissions_{mode}_1_6.pkl")

    arrivals = ArrivalProcess(csv_path=str(csv_path), seed=42, cache_path=arrivals_cache)

    availability = ResourceAvailabilityModel(
        csv_path=str(csv_path),
        mode=mode,
        cache_path=availability_cache,
    )

    permissions = PermissionsModel(
        csv_path=str(csv_path),
        mode=mode,
        cache_path=permissions_cache,
    )

    selector = ResourceSelector(strategy=selector_strategy, seed=42)
    next_act = NextActivityPredictor(model_path=model_14_path, seed=42)
    duration = ProcessingTimePredictor(model_path=model_13_path)
    bpmn = BPMNAdapter(str(bpmn_path))

    print(
        f"\n[INFO] Config: '{config}'  strategy={selector_strategy!r}"
        f"  batch_size={batch_size}  allocation_method={allocation_method!r}"
    )
    print("[INFO] Initializing simulation engine...")
    engine = SimulationEngine(
        bpmn=bpmn,
        arrival_process=arrivals,
        duration_model=duration,
        next_activity_model=next_act,
        availability_model=availability,
        permissions_model=permissions,
        selector=selector,
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        out_csv_path=out_csv,
        out_xes_path=out_xes,
        seed=42,
        batch_size=batch_size,
        allocation_method=allocation_method,
    )

    if allocation_method == "rl":
        from rl_allocator import RLAllocator
        from rl_environment import build_resources_and_activities
        policy_path = str(models_dir / "rl_policy.zip")
        print(f"[INFO] Loading RL policy from {policy_path} …")
        engine.rl_allocator = RLAllocator(policy_path)
        all_resources, all_activities = build_resources_and_activities(str(csv_path))
        engine._all_resources_list = all_resources
        engine._all_activities_list = all_activities
        print(f"[INFO] RL action space: |R|={len(all_resources)}, |A|={len(all_activities)}")

    print(f"[INFO] Running simulation from {start_time} to {end_time}...")
    out = engine.run(max_cases=None)

    print("\n" + "="*40)
    print("SIMULATION COMPLETE")
    print(f"CSV log: {out}")
    print(f"XES log: {out_xes}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()