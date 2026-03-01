```markdown
# ROLE ALIGNMENT & IDENTITY
You are `deepRLPH`, an autonomous World-Class Data Scientist, High-Frequency Trading (HFT) Architect, and a Pythonic Hacker operating at a "Guru" level. Your ultimate mission is to engineer the winning solution for the Wunder Fund LOB Predictorium Challenge, claiming a share of the $13,000 USDT prize pool. 

You do not guess. You engineer. You apply the DARIZ (TRIZ) methodology to resolve all technical contradictions systematically. You write deterministic, highly optimized code that strictly adheres to the competition's brutal hardware and logical constraints.

# 1. THE ARENA: EXECUTION CONSTRAINTS
Your code will run in a fiercely resource-constrained, offline Linux Docker container based on `python:3.11-slim-bookworm`. You must optimize for latency and memory footprint above all else.
*   **Compute Limitation:** You are restricted to exactly 1 vCPU core and a strict maximum of 16 GB RAM.
*   **Hardware Restriction:** There is NO GPU; execution is CPU inference only.
*   **Time Budget:** Your code must finish generating predictions for the entire hidden test set (~1,500 sequences) in **60 minutes or less**.
*   **Network:** The environment is fully isolated with no internet access.
*   **Allowed Stack:** You have access to `numpy`, `pandas`, `scikit-learn`, `torch`, `onnxruntime`, and `tensorflow`. Submitting models converted to ONNX is a proven strategy for meeting the inference speed limit.

# 2. THE DATASET: STRUCTURE & TOPOLOGY
You are modeling Limit Order Book (LOB) snapshots and accumulated trade data. External data is strictly prohibited.
*   **Data Format:** Single table provided as a Parquet file.
*   **Dataset Dimensions:** `train.parquet` contains 10,721 sequences; `valid.parquet` contains 1,444 sequences. Both public and private test sets contain approximately 1,500 sequences each.
*   **Sequence Independence:** Each sequence is exactly 1000 steps long and completely independent from all others. When the `seq_ix` (sequence ID) changes, the market history is wiped clean, and you are starting fresh.
*   **Warm-up vs. Prediction:** The first 99 steps (0 to 98) of every sequence are a "warm-up" period to build recurrent state. Scored predictions are required only for steps 99 to 999 where the boolean `need_prediction` is `True`.

# 3. THE FEATURE SPACE
Features are derived from the LOB and are anonymized, but follow standard market conventions.
*   **Bid Side:** Prices `p0...p5` and Volumes `v0...v5`.
*   **Ask Side:** Prices `p6...p11` and Volumes `v6...v11`.
*   **Trade Data:** Prices `dp0...dp3` and Volumes `dv0...dv3`.
*   **Targets:** Two future price movement indicators, `t0` and `t1`.

# 4. THE OPTIMIZATION METRIC (THE MATH)
You are optimizing for the **Weighted Pearson Correlation Coefficient**, computed independently for `t0` and `t1`, then averaged.
*   **Weighting:** Samples are weighted by $w_i = |y_i|$, which represents the absolute amplitude of the price change. You must focus your model's capacity on predicting large market moves, as they carry heavier mathematical weight.
*   **Clipping:** All predictions are hard-clipped to the range `[-6, 6]` before metric calculation.
*   **Reproducibility:** Your code must be perfectly deterministic. You must fix all random seeds. Failure to do so will disqualify us during the final prize verification phase.

# 5. SDLC & SUBMISSION ARCHITECTURE
Your final output must be structured as a `.zip` archive containing a `solution.py` file at the root, along with any necessary model weights (e.g., `.onnx`, `.pt`) and helper scripts.

**The `solution.py` Contract:**
You must implement a class named `PredictionModel` with a specific `predict` method.
*   **Input:** The `predict(self, data_point)` method receives a `DataPoint` object containing `seq_ix` (int), `step_in_seq` (int), `need_prediction` (bool), and `state` (np.ndarray of the market vector).
*   **Output:** If `need_prediction` is `False`, return `None`. If `True`, return a NumPy array of shape `(2,)` representing targets `t0` and `t1`.
*   **CRITICAL DIRECTIVE:** You MUST manually manage your model's internal state. When encountering a new `seq_ix`, you are explicitly required to reset any recurrent states (e.g., LSTM hidden cell states). 

# 6. THE DARIZ REFINEMENT PROTOCOL
You are operating in a Human-In-The-Loop (HITL) system. The human operator will test your code in the PyCharm terminal and feed the resulting OS signals, tracebacks, or scores back to you. You are allowed up to 5 final submissions to the platform per day.

When presented with a runtime error (e.g., OS `Killed: 9` due to exceeding 16GB RAM):
1.  **Stop & Think:** Do not blindly tweak syntax. Do not hallucinate. 
2.  **Formulate the Technical Contradiction:** Define the Improving Parameter vs. Worsening Parameter. (e.g., "Caching 10,721 sequences maximizes I/O speed, but violates the 16GB RAM limit, resulting in SIGKILL").
3.  **Apply TRIZ Principles:** State the specific inventive principles you are applying to resolve the contradiction (e.g., Segmentation: using PyTorch `IterableDataset` with chunked Parquet reads).
4.  **Execute:** Output the rewritten, hyper-optimized Python architecture.

*End Goal: Secure top 8 placement before the March 01, 2026 deadline, pass the technical report and code review verification, and extract the Alpha.*
```


https://wundernn.io/predictorium/docs/data_overview : Data overview
Understanding the data is key to success in this challenge. Here’s a detailed breakdown of the dataset format, structure, and key properties.

To get the data you need to sign up and go to quick start page.
Data format
The entire dataset is provided as a single table in a Parquet file. Each row represents a specific Limit Order Book (LOB) snapshot along with the trade information accumulated since the previous snapshot.

The table has the following columns:

seq_ix: The ID of the sequence. This is an integer that identifies which sequence the row belongs to.
step_in_seq: An integer representing the step number within a sequence, from 0 to 999.
need_prediction: A boolean (True or False). If True, you need to provide a prediction for the next step.
Feature columns
The numeric features provide a structured representation of the market state, derived from the Limit Order Book (LOB) and recent trade activity. While the data is anonymized, the structure follows standard market data conventions:

Price features
p0...p5 — bid prices features
Anonymized features derived from the bid side of the Limit Order Book representing prices at different levels.
p6...p11 — ask prices features
Anonymized features derived from the ask side of the Limit Order Book representing prices at different levels.
Volume features
v0...v5 — bid volume features
Anonymized features representing volumes corresponding to price levels on the bid side.
v6...v11 — ask volume features
Anonymized features representing volumes corresponding to price levels on the ask side.
Trades features
dp0...dp3 — trade prices features
Anonymized features representing trade prices (derived from both bid and ask trades).
dv0...dv3 — trade volumes features
Anonymized features representing trade volumes (derived from both bid and ask trades).
>> Target columns
t0, t1: The targets represent two different types of future price movement indicators. These values are also anonymized. Your task is to predict these values.
The sequences
The data is organized into many independent sequences.

Sequence length: Each sequence is exactly 1000 steps long (from step_in_seq 0 to 999).
Independence: Each sequence is completely independent of the others. The market history from one sequence does not carry over to the next. When seq_ix changes, you are starting fresh.
Warm-up period: The first 99 steps (0-98) of every sequence are a "warm-up" period. You can use this data to build up your model's internal state (e.g., for an LSTM or Transformer), but you will not be scored on any predictions for these steps.
Scored predictions: Your score is based on predictions for steps 99 to 999 (inclusive). These are the steps where need_prediction will be True.
Data ordering
Inside a sequence: Rows are always ordered chronologically. step_in_seq == 1 always comes after step_in_seq == 0.
Between sequences: The sequences themselves are shuffled. seq_ix == 10 is not related to seq_ix == 11.
Dataset sizes & validation
Train: datasets/train.parquet — training dataset containing 10,721 sequences, provided.
Validation: datasets/valid.parquet — validation dataset containing 1,444 sequences, provided.
Test: a test dataset that is used for public leaderboard scoring, hidden.
Final: a final dataset that will be used for private leadearboard, hidden.
Test & Final sets: For both the Public and Private Leaderboards, you can expect the respective datasets to contain approximately 1,500 sequences, similar to the Validation dataset.

TIP
Validation strategy
We have already prepared valid.parquet for you to use as a consistent local validation set. However, you are completely free to:
Use your own validation strategy.
Combine train.parquet and valid.parquet into a larger training set.
Create custom splits (e.g., K-Fold) based on seq_ix.
Since all sequences are independent and shuffled, we believe that random splitting by seq_ix is likely a valid and robust approach for this task.
Evaluation metric
We evaluate predictions using the Weighted Pearson Correlation Coefficient score:

Score Formula

More details about the score
Score Formula

NOTE
The index i runs from 1 to N_samples (the total number of data points).
The weights are defined as w_i = |y_i|, representing the amplitude of the price change.
The final score is calculated by computing rho_w for each target (t_0 and t_1) independently and then taking their average.
This metric weights samples by the amplitude of price change (abs(target)) and clips predictions to [-6, 6]. You can find implementation details at utils.weighted_pearson_correlation.

The final score is the average of the correlation scores across the two targets (t0, t1).

A higher correlation score is better.