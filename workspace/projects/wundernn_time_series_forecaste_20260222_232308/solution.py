import os
import sys
import torch
import numpy as np
from typing import Optional

# Ensure local imports work correctly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import DataPoint

# Iron Shield Constants - MUST MATCH TRAINING EXACTLY
EPS = 1e-5
FEAT_CLIP = 20.0
WINDOW_SIZE = 1000


class PredictionModel:
    """
    Guru Predictor: temporal Fusion DeepLOB.
    Uses a 40-feature Alpha Stack and a rolling 1000-step buffer.
    """

    def __init__(self):
        # 1. State Management
        self.current_seq_ix = None
        self.history = []  # Buffer for raw 32 features

        # 2. Load the Beast (TorchScript JIT)
        model_path = os.path.join(CURRENT_DIR, "model_jit.pt")
        self.device = torch.device("cpu")

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"✅ Success: Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Critical Error loading model: {e}")
            self.model = None

    def _engineer_features(self, raw_buffer: np.ndarray) -> torch.Tensor:
        """
        The 'Iron Shield' Pipeline: Converts (1000, 32) -> (1, 1000, 40)
        Matches loader.py logic exactly to preserve the 0.51 Pearson.
        """
        # Split 32 raw features
        prices = raw_buffer[:, :12]
        volumes = np.clip(raw_buffer[:, 12:24], 0, None)
        deal_prices = raw_buffer[:, 24:28]
        deal_volumes = np.clip(raw_buffer[:, 28:32], 0, None)

        # 1. Mid-Price (Calculated on the fly)
        best_bid, best_ask = prices[:, 0], prices[:, 6]
        mid_prices = (best_bid + best_ask) / 2.0
        mid_prices[mid_prices == 0] = 1.0
        mid_exp = mid_prices[:, np.newaxis]

        # 2. Normalize Prices & Deals
        norm_prices = (prices - mid_exp) / (mid_exp + EPS)

        norm_deal_prices = (deal_prices - mid_exp) / (mid_exp + EPS)
        trade_mask = deal_prices > 0
        norm_deal_prices[~trade_mask] = 0.0

        # 3. Log-Scale Volumes
        log_volumes = np.log1p(volumes)
        log_deal_volumes = np.log1p(deal_volumes)

        # 4. OFI & Imbalances
        ofi = log_volumes[:, :6] - log_volumes[:, 6:12]
        is_buy = (deal_prices >= best_ask[:, np.newaxis]) & trade_mask
        is_sell = (deal_prices <= best_bid[:, np.newaxis]) & trade_mask
        buy_imb = np.sum(log_deal_volumes * is_buy, axis=1, keepdims=True)
        sell_imb = np.sum(log_deal_volumes * is_sell, axis=1, keepdims=True)

        # 5. Concatenate 40 Features
        X = np.concatenate([
            norm_prices, log_volumes, norm_deal_prices,
            log_deal_volumes, ofi, buy_imb, sell_imb
        ], axis=1).astype(np.float32)

        # Final Sanitization
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -FEAT_CLIP, FEAT_CLIP)

        # Add batch dimension
        return torch.from_numpy(X).unsqueeze(0).to(self.device)

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        # A. Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []

        # B. Collect current state
        self.history.append(data_point.state.astype(np.float32))

        # C. Return None if evaluator doesn't want a prediction yet
        if not data_point.need_prediction:
            return None

        if self.model is None:
            return np.zeros(2)

        # D. Sequence Preparation (Temporal Context)
        # Take the last 1000 steps of history
        raw_window = self.history[-WINDOW_SIZE:]

        # E. Padding (if history is too short for the model)
        if len(raw_window) < WINDOW_SIZE:
            padding = [np.zeros(32, dtype=np.float32)] * (WINDOW_SIZE - len(raw_window))
            raw_window = padding + raw_window

        raw_array = np.array(raw_window)

        # F. Feature Engineering & Inference
        with torch.no_grad():
            input_tensor = self._engineer_features(raw_array)
            output = self.model(input_tensor)  # Expected: (1, 901, 2)

        # G. Slice the Prediction
        # The evaluator asks for prediction STEP BY STEP.
        # Our model predicts the whole sequence, so we return the last step
        # corresponding to the current microsecond.
        prediction = output[0, -1, :].cpu().numpy()

        return prediction.astype(np.float64)  # Parquet/Scorer expects double