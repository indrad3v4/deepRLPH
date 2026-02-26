import os
import sys

# ==========================================
# CPU THREAD LOCK (CRITICAL FOR 1-vCPU DOCKER)
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
from typing import Optional

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import DataPoint

# Force PyTorch into pure inference mode globally
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Iron Shield Constants
EPS = 1e-5
FEAT_CLIP = 20.0
WINDOW_SIZE = 1000


class PredictionModel:
    """
    Strategic Predictor: temporal Fusion DeepLOB.
    Uses O(1) step-wise feature engineering and a contiguous rolling buffer.
    """

    def __init__(self):
        # 1. Load Model
        model_path = os.path.join(CURRENT_DIR, "model_jit.pt")
        self.device = torch.device("cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 2. State Management (Pre-allocated for extreme speed)
        self.current_seq_ix = None
        self.feature_buffer = np.zeros((WINDOW_SIZE, 40), dtype=np.float32)

    def _engineer_single_tick(self, state: np.ndarray) -> np.ndarray:
        """
        O(1) Feature Engineering.
        Instead of re-calculating the 1000-step window, we calculate
        only the new incoming microsecond and return a 40-feature vector.
        """
        # Slicing
        prices = state[:12]
        volumes = np.clip(state[12:24], 0, None)
        deal_prices = state[24:28]
        deal_volumes = np.clip(state[28:32], 0, None)

        # Mid-Price
        best_bid = prices[0]
        best_ask = prices[6]
        mid_price = (best_bid + best_ask) / 2.0
        if mid_price == 0:
            mid_price = 1.0

        # Normalize Prices
        norm_prices = (prices - mid_price) / (mid_price + EPS)

        # Normalize Deals
        trade_mask = deal_prices > 0
        norm_deal_prices = np.zeros_like(deal_prices)
        norm_deal_prices[trade_mask] = (deal_prices[trade_mask] - mid_price) / (mid_price + EPS)

        # Log Volumes
        log_volumes = np.log1p(volumes)
        log_deal_volumes = np.log1p(deal_volumes)

        # OFI & Imbalances
        ofi = log_volumes[:6] - log_volumes[6:12]

        is_buy = (deal_prices >= best_ask) & trade_mask
        is_sell = (deal_prices <= best_bid) & trade_mask

        # Using [np.sum()] to keep them as 1D arrays for concatenation
        buy_imb = np.array([np.sum(log_deal_volumes * is_buy)], dtype=np.float32)
        sell_imb = np.array([np.sum(log_deal_volumes * is_sell)], dtype=np.float32)

        # Concatenate 40 Features
        feat_vec = np.concatenate([
            norm_prices, log_volumes, norm_deal_prices,
            log_deal_volumes, ofi, buy_imb, sell_imb
        ]).astype(np.float32)

        # Shield
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(feat_vec, -FEAT_CLIP, FEAT_CLIP)

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        # A. Sequence Reset
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.feature_buffer.fill(0.0)  # Lightning-fast C-level zeroing

        # B. Engineer the single tick
        new_features = self._engineer_single_tick(data_point.state)

        # C. Rolling Buffer Update (Shift left, append to end)
        self.feature_buffer[:-1] = self.feature_buffer[1:]
        self.feature_buffer[-1] = new_features

        # D. Skip inference if evaluator doesn't need it
        if not data_point.need_prediction:
            return None

        # E. Inference
        # feature_buffer is already (1000, 40)
        tensor_x = torch.from_numpy(self.feature_buffer).unsqueeze(0).to(self.device)

        output = self.model(tensor_x)  # Output: (1, 901, 2)

        # Return the last prediction step
        prediction = output[0, -1, :].cpu().numpy()
        return prediction.astype(np.float64)