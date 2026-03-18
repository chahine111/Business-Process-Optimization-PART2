# Business Process Simulation & Optimization
### BPI 2017 — Loan Application Process · TU Munich BPSO Assignment

A data-driven discrete-event simulation (DES) of the BPI 2017 loan
application process, extended with six resource allocation methods and a
full evaluation framework.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Setup](#data-setup)
3. [Folder Structure](#folder-structure)
4. [Quick Start](#quick-start)
5. [All Scripts](#all-scripts)
6. [Allocation Methods](#allocation-methods)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Workforce Optimization](#workforce-optimization)
9. [Deep RL (Optional)](#deep-rl-optional)

---

## Prerequisites

Python **3.10+** is required. Install all dependencies:

```bash
pip install pandas numpy scipy scikit-learn joblib matplotlib lxml pm4py
```

For the Deep RL method (optional):
```bash
pip install gymnasium stable-baselines3 sb3-contrib
```

---

## Data Setup

The raw event log is **not** included in the repository (too large for git).

1. Download **BPI Challenge 2017** (`bpi2017.csv`) from the 4TU repository
   or your course materials.
2. Place it at the **project root** (same level as `src/` and `data/`):

```
Business-Process-Simulation-Model-main/
├── bpi2017.csv          ← place here
├── data/
│   └── Signavio_Model.bpmn
├── src/
│   └── ...
```

> The BPMN process model (`data/Signavio_Model.bpmn`) is already included.

---

## Folder Structure

```
.
├── bpi2017.csv                        # Raw event log (download separately)
├── data/
│   └── Signavio_Model.bpmn            # BPMN process model
├── models/                            # Auto-generated .pkl artifacts (git-ignored)
├── outputs/
│   └── task_1_2_analysis_report.pdf   # Final evaluation report (real results)
├── src/
│   ├── simulation_engine_1_1.py       # Core DES engine
│   ├── arrival_model_1_2.py           # Case arrival process
│   ├── processing_times_TRAIN.py      # Processing time model training
│   ├── processing_time_predictor.py   # Processing time predictor (runtime)
│   ├── next_activity_TRAIN_1_4.py     # Next-activity model training
│   ├── next_activity_predictor_1_4.py # Next-activity predictor (runtime)
│   ├── resource_availability_1_5.py   # Resource availability model
│   ├── permissions_model_1_6.py       # Resource-activity authorization
│   ├── resource_selector_1_7.py       # Heuristic resource selectors
│   ├── bpmn_adapter.py                # BPMN parser / XOR gateway logic
│   ├── rl_environment.py              # Gymnasium env for Deep RL
│   ├── rl_allocator.py                # RL policy inference wrapper
│   ├── rl_train.py                    # MaskablePPO training script
│   ├── run_simulation.py              # Run a single allocation config
│   ├── evaluation.py                  # Evaluation metrics module
│   ├── run_evaluation.py              # Run all configs + optimization
│   └── generate_analysis_report.py   # PDF report generator (real data)
└── README.md
```

---

## Quick Start

> **Windows note:** Always prefix Python commands with
> `PYTHONIOENCODING=utf-8` to avoid encoding errors with special characters
> printed by some training scripts.

### Option A — Run everything at once (recommended)

```bash
cd src
PYTHONIOENCODING=utf-8 python run_evaluation.py
```

This single command:
1. Trains all sub-models automatically if they don't exist yet (~2–5 min first run)
2. Runs the simulation for **all 5 allocation methods** for the full year 2016
3. Identifies the 2 least-contributing employees and re-runs the baseline without them
4. Prints a metric comparison table to the console
5. Saves all output CSVs to `outputs/`

Then generate the PDF report with real results:

```bash
PYTHONIOENCODING=utf-8 python generate_analysis_report.py
```

Output: `outputs/task_1_2_analysis_report.pdf`

---

### Option B — Run a single config

Edit the `config` variable at the bottom of `run_simulation.py`, then:

```bash
cd src
PYTHONIOENCODING=utf-8 python run_simulation.py
```

Available configs: `r_rma`, `r_rra`, `r_shq`, `kbatch`, `park_song`, `rl`

---

### Option C — Compute metrics on an existing log

```python
from evaluation import compute_all_metrics, print_comparison_table

metrics = compute_all_metrics("outputs/eval_r_rma.csv")
print_comparison_table({"r_rma": metrics})
```

---

## All Scripts

| Script | Purpose |
|---|---|
| `run_simulation.py` | Run one allocation config, save log CSV |
| `run_evaluation.py` | Run all configs, compare metrics, fire-2-employees scenario |
| `generate_analysis_report.py` | Generate 7-page PDF from real simulation results |
| `evaluation.py` | Standalone metrics module (importable) |
| `rl_train.py` | Train the Deep RL policy (optional, needs `gymnasium`) |

---

## Allocation Methods

Six methods are implemented in `simulation_engine_1_1.py`:

| Config key | Method | Description |
|---|---|---|
| `r_rma` | **Random (R-RMA)** | Pick a random authorized resource for each task |
| `r_rra` | **Round-Robin (R-RRA)** | Rotate through resources in fixed order |
| `r_shq` | **Shortest Queue (R-SHQ)** | Assign to the resource with fewest waiting tasks |
| `kbatch` | **K-Batching (k=5)** | Wait for 5 tasks, then solve a cost-matrix assignment with the Hungarian algorithm |
| `park_song` | **Park & Song (2019)** | Predict upcoming tasks using a bigram model; include predictions in the cost matrix to enable strategic idling |
| `rl` | **Deep RL (Middelhuis et al. 2025)** | MaskablePPO policy trained via a Gymnasium wrapper — requires `python rl_train.py` first |

### Simulation results (2016 full year)

| Metric | R-RMA | R-RRA | R-SHQ | K-Batch | Park & Song |
|---|---|---|---|---|---|
| Avg Cycle Time (h) | 1.295 | 1.503 | 1.507 | **1.198** | 1.439 |
| P90 Cycle Time (h) | **2.000** | 4.000 | 4.000 | 4.000 | 4.001 |
| Avg Resource Occ (%) | 28.6% | 28.5% | 34.9% | 23.1% | 37.1% |
| Resource Fairness | 0.304 | 0.293 | 0.256 | **0.248** | 0.292 |
| Avg Waiting Time (h) | **1.489** | 1.470 | 1.631 | 1.696 | 1.716 |
| Throughput (cases/day) | 6.96 | 6.15 | 6.72 | **10.70** | 7.61 |
| Completed Cases | 2,547 | 2,251 | 2,461 | **3,915** | 2,786 |

**K-Batch** wins on throughput and average cycle time.
**R-RMA** wins on P90 cycle time (avoids long-tail stacking).
**K-Batch** is also the fairest in terms of load distribution.

---

## Evaluation Metrics

All metrics are computed by `evaluation.py` from a simulated log CSV.

### Basic (required by assignment)

| # | Metric | Definition |
|---|---|---|
| 1 | **Avg Cycle Time** | Mean of `(last complete event - first event)` per case, in hours |
| 2 | **Avg Resource Occupation** | Mean of `busy_time / simulation_window` per resource |
| 3 | **Resource Fairness** | Weighted mean absolute deviation of per-resource occupation from the group mean (0 = perfectly equal) |

### Advanced (own contributions)

| # | Metric | Why it is useful |
|---|---|---|
| 4 | **P90 Cycle Time** | Captures tail / worst-case behaviour; the mean alone hides long outliers |
| 5 | **Avg Waiting Time** | Queue wait only (schedule to start); isolates allocation delay from processing time |
| 6 | **Throughput (cases/day)** | Productivity measure — how many cases complete per calendar day |

---

## Workforce Optimization

`run_evaluation.py` automatically answers the management question:

> *"Which two employees should we fire?"*

Resources are ranked by `busy_hours x (tasks_completed + 1)`.
The two with the lowest score contribute least and are safest to remove.

**Results (BPI 2017):**

The two candidates (User_114, User_122) had **0 tasks and 0% occupation** —
effectively inactive accounts. Impact of removing them:

| | Baseline | After firing 2 |
|---|---|---|
| Avg Cycle Time | 1.295 h | 1.337 h (+0.04 h) |
| Avg Occupation | 28.6% | 31.4% (+2.9 pp) |
| Throughput | 6.96 c/day | 6.43 c/day (-0.53) |
| Fairness | 0.304 | 0.289 (improved) |

To also run the **nine-to-five schedule** scenario (restrict all resources to
09:00–17:00 Mon–Fri), uncomment `run_nine_to_five()` at the bottom of
`run_evaluation.py`.

---

## Deep RL (Optional)

Requires `gymnasium`, `stable-baselines3`, and `sb3-contrib`.

```bash
# Step 1 — train the policy (~100k timesteps on 3 months of data)
cd src
PYTHONIOENCODING=utf-8 python rl_train.py
# Saves: models/rl_policy.zip  and  models/rl_learning_curve.csv

# Step 2 — run simulation with the trained policy
# Edit run_simulation.py: set config = "rl"  at the bottom
PYTHONIOENCODING=utf-8 python run_simulation.py
```

The RL method uses **MaskablePPO** with:
- State: `[busy x |R|, remaining_norm x |R|, task_count x |A|]`
- Actions: `Discrete(|R| x |A| + 1)` where the last action means Postpone
- Reward: `-cycle_time_hours` per completed case, `-0.001` per step

---

## References

[1] Park, G., & Song, M. (2019). Prediction-based resource allocation using
LSTM and minimum cost and maximum flow algorithm. ICPM 2019, pp. 121-128.

[2] Kunkler, M., & Rinderle-Ma, S. (2024). Online resource allocation to
process tasks under uncertain resource availabilities. ICPM 2024, pp. 137-144.

[3] Middelhuis, J., et al. (2025). Learning policies for resource allocation
in business processes. Information Systems, 128, 102492.
