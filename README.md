# Business Process Simulation & Optimization
### BPI 2017 — Loan Application Process

A data-driven discrete-event simulation (DES) of the BPI 2017 loan
application process, extended with six resource allocation methods and a
full evaluation framework.

---

## Prerequisites

Python **3.10+** is required. Install all dependencies:

```bash
pip install pandas numpy scipy scikit-learn joblib matplotlib lxml pm4py lightgbm
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
2. Place it inside the `data/` folder:

```
Business-Process-Simulation-Model-main/
├── data/
│   ├── bpi2017.csv          ← place here
│   └── Signavio_Model.bpmn
├── src/
│   └── ...
```

> The BPMN process model (`data/Signavio_Model.bpmn`) is already included.

---

## Folder Structure

```
.
├── data/
│   ├── bpi2017.csv                    # Raw event log (download separately)
│   └── Signavio_Model.bpmn            # BPMN process model
├── models/                            # Auto-generated .pkl artifacts (git-ignored)
├── outputs/
│   └── task_1_2_analysis_report.pdf   # Final evaluation report
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
│   └── generate_analysis_report.py    # PDF report generator
└── README.md
```

---

## Quick Start

### Option A — Run everything at once (recommended)

```bash
cd src
python run_evaluation.py
```

This single command:
1. Trains all sub-models automatically if they don't exist yet (~2-5 min first run)
2. Runs the simulation for **all 5 allocation methods** for the full year 2016
3. Identifies the 2 least-contributing employees and re-runs the baseline without them
4. Prints a metric comparison table to the console
5. Saves all output CSVs to `outputs/`

Then generate the PDF report:

```bash
python generate_analysis_report.py
```

Output: `outputs/task_1_2_analysis_report.pdf`

---

### Option B — Run a single config

Edit the `config` variable in `run_simulation.py`, then:

```bash
cd src
python run_simulation.py
```

Available configs: `r_rma`, `r_rra`, `r_shq`, `kbatch`, `park_song`, `rl`

---

## Allocation Methods

| Config key | Method | Description |
|---|---|---|
| `r_rma` | **Random (R-RMA)** | Pick a random authorized resource for each task |
| `r_rra` | **Round-Robin (R-RRA)** | Rotate through resources in fixed order |
| `r_shq` | **Shortest Queue (R-SHQ)** | Assign to the resource with fewest historical tasks |
| `kbatch` | **K-Batching (k=5)** | Wait for 5 tasks, then solve a cost-matrix assignment with the Hungarian algorithm |
| `park_song` | **Park & Song (2019)** | Predict upcoming tasks using a bigram model; include predictions in the cost matrix to enable strategic idling |
| `rl` | **Deep RL (Middelhuis et al. 2025)** | MaskablePPO policy trained via a Gymnasium wrapper — requires `python rl_train.py` first |

### Simulation results (2016 full year)

| Metric | R-RMA | R-RRA | R-SHQ | K-Batch | Park & Song |
|---|---|---|---|---|---|
| Avg Cycle Time (h) | 1.283 | **1.180** | 1.372 | 1.339 | 1.404 |
| P90 Cycle Time (h) | **2.001** | **2.001** | 4.000 | 4.002 | 4.000 |
| Avg Resource Occ (%) | 30.1% | 29.4% | 37.4% | 21.4% | **39.3%** |
| Resource Fairness | 0.262 | 0.250 | 0.293 | **0.225** | 0.268 |
| Avg Waiting Time (h) | 1.569 | **1.329** | 1.481 | 1.814 | 1.951 |
| Throughput (cases/day) | 6.76 | 6.33 | 6.22 | **10.56** | 7.90 |
| Completed Cases | 2,473 | 2,318 | 2,276 | **3,866** | 2,893 |

**K-Batch** wins on throughput and fairness.
**R-RRA** wins on average cycle time, P90, and waiting time.
**Park & Song** achieves the highest resource occupation.

---

## Evaluation Metrics

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

The two candidates (User_11, User_101) were the least-contributing resources.
Impact of removing them:

| | Baseline | After firing 2 | Delta |
|---|---|---|---|
| Avg Cycle Time | 1.283 h | 1.610 h | +0.33 h |
| P90 Cycle Time | 2.001 h | 3.994 h | +1.99 h |
| Avg Occupation | 30.1% | 30.2% | +0.2 pp |
| Throughput | 6.76 c/day | 6.90 c/day | +0.15 |

---

## Deep RL (Optional)

Requires `gymnasium`, `stable-baselines3`, and `sb3-contrib`.

```bash
# Step 1 — train the policy (~100k timesteps on 3 months of data)
cd src
python rl_train.py

# Step 2 — run simulation with the trained policy
# Edit run_simulation.py: set config = "rl"
python run_simulation.py
```
