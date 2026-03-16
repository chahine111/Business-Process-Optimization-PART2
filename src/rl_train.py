"""
src/rl_train.py

Training script for the RL resource-allocation policy (Method C).
Reference: Middelhuis et al. 2025 — "Learning policies for resource allocation
           in business processes".

How to run
----------
    cd <project_root>
    python src/rl_train.py

Output
------
    models/rl_policy.zip        — trained MaskablePPO model
    models/rl_learning_curve.csv — episode rewards over training

Configuration
-------------
Adjust TOTAL_TIMESTEPS, N_STEPS, etc. below to trade off training time vs
policy quality.  The default 100,000 timesteps is a quick smoke-test;
increase to 500,000+ for a well-trained policy.

Dependencies
------------
    pip install stable-baselines3 sb3-contrib gymnasium
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make src/ importable when running from the project root.
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np

# ── Check optional dependencies ────────────────────────────────────────────
try:
    import gymnasium
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:
    print(
        "Missing RL dependencies. Install them with:\n"
        "    pip install stable-baselines3 sb3-contrib gymnasium\n"
    )
    raise

# ── Project imports ────────────────────────────────────────────────────────
from arrival_model_1_2 import ArrivalProcess
from resource_availability_1_5 import ResourceAvailabilityModel
from permissions_model_1_6 import PermissionsModel
from resource_selector_1_7 import ResourceSelector
from processing_time_predictor import ProcessingTimePredictor
from run_simulation import NextActivityPredictor
from bpmn_adapter import BPMNAdapter
from rl_environment import BPSimEnv, build_resources_and_activities

# ══════════════════════════════════════════════════════════════════════════
# Training configuration
# ══════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS = 100_000   # increase for better policy quality
N_STEPS         = 2_048     # PPO rollout length
BATCH_SIZE      = 64
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
SEED            = 42

# Use a shorter window during training to keep episodes manageable.
TRAIN_START = "2016-01-01 00:00:00+00:00"
TRAIN_END   = "2016-04-01 00:00:00+00:00"   # 3 months


# ══════════════════════════════════════════════════════════════════════════
# Reward-logging callback
# ══════════════════════════════════════════════════════════════════════════
class EpisodeLogger(BaseCallback):
    """Logs episode rewards to CSV for learning-curve analysis."""

    def __init__(self, csv_path: str, log_every: int = 10) -> None:
        super().__init__(verbose=0)
        self._csv_path = csv_path
        self._log_every = log_every
        self._ep_count = 0
        self._ep_rewards: list = []
        self._fh = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["episode", "timestep", "mean_reward"])
        self._fh.flush()

    def _on_step(self) -> bool:
        # SB3 stores episode info in the info dict when an episode ends.
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self._ep_rewards.append(ep_info["r"])
                self._ep_count += 1
                if self._ep_count % self._log_every == 0:
                    mean_r = float(np.mean(self._ep_rewards[-self._log_every:]))
                    self._writer.writerow([self._ep_count, self.num_timesteps, mean_r])
                    self._fh.flush()
                    print(
                        f"  Episode {self._ep_count:>5}  "
                        f"step {self.num_timesteps:>8}  "
                        f"mean_reward={mean_r:.3f}"
                    )
        return True  # continue training

    def _on_training_end(self) -> None:
        self._fh.close()


# ══════════════════════════════════════════════════════════════════════════
# Action-masking wrapper expected by sb3_contrib.ActionMasker
# ══════════════════════════════════════════════════════════════════════════
def _get_action_mask(env: BPSimEnv) -> np.ndarray:
    return env.action_masks()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir   = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = str(data_dir / "bpi2017.csv")
    bpmn_path = str(data_dir / "Signavio_Model.bpmn")

    policy_path = str(models_dir / "rl_policy.zip")
    curve_path  = str(models_dir / "rl_learning_curve.csv")

    print("[RL-TRAIN] Building resource/activity lists from CSV …")
    all_resources, all_activities = build_resources_and_activities(csv_path)
    print(f"  |R| = {len(all_resources)},  |A| = {len(all_activities)}")
    print(f"  action_space size = {len(all_resources) * len(all_activities) + 1}")

    # ── Build engine kwargs (same as run_simulation, but shorter training window) ─
    # Using "advanced" mode (2h buckets) to match the full simulation configuration.
    # The training window is trimmed to 3 months so each episode completes quickly.
    mode = "advanced"
    arrivals_cache     = str(models_dir / "arrival_model_1_2.pkl")
    availability_cache = str(models_dir / f"availability_{mode}_1_5.pkl")
    permissions_cache  = str(models_dir / f"permissions_{mode}_1_6.pkl")
    model_13_path      = str(models_dir / "processing_model_advanced.pkl")
    model_14_path      = str(models_dir / "next_activity_bigram_model.pkl")

    engine_kwargs = dict(
        bpmn               = BPMNAdapter(bpmn_path),
        arrival_process    = ArrivalProcess(csv_path=csv_path, seed=SEED,
                                            cache_path=arrivals_cache),
        duration_model     = ProcessingTimePredictor(model_path=model_13_path),
        next_activity_model= NextActivityPredictor(model_path=model_14_path, seed=SEED),
        availability_model = ResourceAvailabilityModel(csv_path=csv_path, mode=mode,
                                                       cache_path=availability_cache),
        permissions_model  = PermissionsModel(csv_path=csv_path, mode=mode,
                                              cache_path=permissions_cache),
        selector           = ResourceSelector(strategy="random", seed=SEED),
        mode               = mode,
        start_time         = TRAIN_START,
        end_time           = TRAIN_END,
        out_csv_path       = str(models_dir / "_rl_train_tmp.csv"),
        seed               = SEED,
        batch_size         = 1,
        # allocation_method is forced to "rl" inside BPSimEnv.reset()
    )

    print("[RL-TRAIN] Constructing environment …")
    env = BPSimEnv(
        engine_kwargs  = engine_kwargs,
        all_resources  = all_resources,
        all_activities = all_activities,
    )

    # Wrap with ActionMasker so MaskablePPO queries action_masks() automatically.
    env = ActionMasker(env, _get_action_mask)

    # Wrap with Monitor so SB3 writes episode statistics into the info dict.
    # EpisodeLogger reads the "episode" key to log rewards to CSV.
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env)

    print(f"[RL-TRAIN] Training MaskablePPO for {TOTAL_TIMESTEPS:,} timesteps …")
    model = MaskablePPO(
        policy          = "MlpPolicy",
        env             = env,
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        learning_rate   = LEARNING_RATE,
        verbose         = 0,
        seed            = SEED,
    )

    logger = EpisodeLogger(curve_path, log_every=10)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger)

    model.save(policy_path)
    print(f"\n[RL-TRAIN] Saved policy  → {policy_path}")
    print(f"[RL-TRAIN] Learning curve → {curve_path}")


if __name__ == "__main__":
    main()
