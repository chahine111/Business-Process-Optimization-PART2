Data-Driven Business Process Simulation (BPIC17)
================================================

Hi,
this repository contains our data-driven simulator for the BPIC17 Loan Application process.
The simulation follows the BPMN control-flow and uses the BPIC17 historical log to learn the
sub-models (arrivals, durations, next activity, availability, permissions, etc.).


What you need
-------------
1) BPMN model
   - data/Signavio_Model.bpmn

2) Historical event log (CSV)
   - data/bpi2017.csv

IMPORTANT:
The simulator expects the CSV to be named exactly "bpi2017.csv" and placed inside the data/ folder.


BPIC17 dataset preparation (XES → CSV)
--------------------------------------
The original BPIC17 log is usually provided as a .xes file.
Before running, please convert the .xes into CSV and name it:

  bpi2017.csv

Then place it here:

  data/bpi2017.csv


Dependencies
------------
Core (always required):

  pip install numpy pandas scipy joblib pm4py

RL training / inference only (Method C):

  pip install gymnasium stable-baselines3 sb3-contrib

scipy is used by the Park & Song cost-matrix solver (Method A).
gymnasium / stable-baselines3 / sb3-contrib are only needed if you use the "rl" config.


Resource Allocation Methods
----------------------------
The simulation supports six allocation configurations selectable in run_simulation.py:

  r_rma       — Random assignment, one task at a time (baseline).
  r_rra       — Round-robin assignment, one task at a time.
  r_shq       — Shortest-queue heuristic, one task at a time.
  kbatch      — K-Batch: accumulate K=5 tasks then solve a cost matrix (Hungarian).
  park_song   — Method A: prediction-based allocation with strategic idling
                (Park & Song 2019).  Predicts each running case's next activity
                and reserves a resource for it using a discounted cost matrix.
  rl          — Method C: MaskablePPO deep RL policy trained on the simulation
                (Middelhuis et al. 2025).  Must be trained before use — see below.

To change the active configuration, edit the `config` variable at the top of
src/run_simulation.py.


How to run
----------
From the project root, run:

  python src/run_simulation.py

This will run the simulation and create an output event log.


Training the RL policy (Method C only)
----------------------------------------
Before using the "rl" config you must train the policy:

  python src/rl_train.py

This trains a MaskablePPO model on a 3-month simulation window (Jan–Apr 2016)
for 100,000 timesteps and saves the result to:

  models/rl_policy.zip          — trained policy (loaded at inference time)
  models/rl_learning_curve.csv  — episode rewards over training

After training, switch to the "rl" config in run_simulation.py and run normally.
Training time depends on hardware but typically takes 20–60 minutes.


Outputs
-------
After running, the generated log will be written to:

  outputs/

Example:
  outputs/simulated_log.csv

(The exact filename depends on the active run configuration.)


Model caching (pkl files)
-------------------------
On the first run, the simulator trains the sub-models from the BPIC17 CSV and
stores them as .pkl files in:

  models/

This is done once to avoid retraining every time. On later runs, the simulator
loads the cached models directly, which makes execution faster.


Notes
-----
- The first run can take longer since the sub-models are trained and then cached.
- Please keep the filenames and paths as described above (especially inside data/).
- The Park & Song method (Method A) requires scipy; it falls back to greedy
  assignment if scipy is not installed.
- The RL method (Method C) requires gymnasium, stable-baselines3, and sb3-contrib.
  Run rl_train.py first to generate models/rl_policy.zip before using the "rl" config.
