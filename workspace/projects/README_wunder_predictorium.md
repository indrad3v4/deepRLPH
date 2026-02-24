# Wunder Challenge: LOB Predictorium – deepRLPH Notes

## 1. Overview

- Goal: predict two future price‑movement targets (t0, t1) from sequences of limit order book (LOB) states and recent trades.
- Data comes as Parquet tables; each row is one LOB snapshot plus trades since the previous snapshot.
- Sequences are length 1000; steps 0–98 are warm‑up, predictions are scored for steps 99–999 where `need_prediction == True`.
- The leaderboard metric is a weighted Pearson correlation averaged over t0 and t1, implemented in `utils.weighted_pearson_correlation`.

## 2. Starter pack layout

After downloading the official starter pack (`wnn_predictorium_starterpack`), the structure is:

- `datasets/train.parquet` – training sequences (10 721 sequences).
- `datasets/valid.parquet` – validation sequences (1 444 sequences).
- `example_solution/solution.py` – minimal working baseline implementing the required `PredictionModel` interface.
- `example_solution/baseline.onnx` – pre‑trained ONNX model used by the baseline.
- `utils.py` – helper types (e.g. `DataPoint`) and the official scoring function; organizers recommend not modifying this file.

## 3. Data schema (high‑level)

Each row in `train.parquet` / `valid.parquet` contains:

- Indexing:
  - `seq_ix` – sequence identifier.
  - `step_in_seq` – step index 0–999 within the sequence.
  - `need_prediction` – whether the model must output a prediction for the next step.

- Features (all anonymized):
  - Price features: `p0..p5` (bid‑side), `p6..p11` (ask‑side).
  - Volume features: `v0..v5` (bid), `v6..v11` (ask).
  - Trade features: `dp0..dp3` (trade prices), `dv0..dv3` (trade volumes).

- Targets:
  - `t0`, `t1` – two continuous future price‑movement indicators to predict.

Sequences are shuffled across `seq_ix`; there is no dependency between different sequence IDs.

## 4. Environment & baseline

Recommended local setup (matches the evaluation container):

```bash
python -m venv env
source env/bin/activate      # macOS / Linux
pip install numpy pandas pyarrow tqdm onnxruntime torch
```

To run the official baseline locally:

```bash
cd example_solution
python solution.py
```

This script loads `valid.parquet`, runs the ONNX baseline, and prints the weighted Pearson correlation score on the validation set.

## 5. Submission requirements (short)

Submissions are `.zip` archives containing everything needed for inference.

- At the root of the archive there must be a `solution.py` file.
- `solution.py` must define a class `PredictionModel` with method `predict(self, data_point) -> np.ndarray | None`.
- `predict` returns `None` when `data_point.need_prediction` is `False`, otherwise a NumPy vector of length 2 for `(t0, t1)`.
- You may include extra Python modules, model weights (e.g. `.onnx`, `.pt`), and small config files inside the zip.
- Code runs in an offline Docker container based on `python:3.11-slim-bookworm` with 1 vCPU, 16 GB RAM and a 60‑minute time limit.

Basic rules: use only provided data (plus allowed public pre‑trained models), no internet at inference, up to 5 submissions per day, and one account per participant or team.

## 6. How we use this in deepRLPH

When creating a **New Project** in the RALPH wizard:

- Set **Project Type** to “ML Competition”.
- In **Description**, mention that this is the Wunder LOB Predictorium time‑series / sequence‑modeling challenge with weighted Pearson correlation as the main KPI.
- Attach this README file in **Documentation Files** so the wizard and agents have a concise spec of the competition.
- Place `train.parquet` and `valid.parquet` paths under **Dataset Files**, and any baseline scripts / ONNX models under **Baseline Code/Models`.

In Step 3 (Review & Advanced), you can set:

- `domain`: `time_series_forecasting`.
- `model_type`: e.g. `GRU` / `Transformer`.
- `ml_framework`: e.g. `PyTorch`.
- `kpi_metric`: `weighted_pearson_correlation` (or a short alias you prefer).
- `kpi_target`: target validation score you want the agents to reach (for example, start around the official baseline).
