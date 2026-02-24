import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class OFIDataLoader(Dataset):
    """
    Data-Centric Loader for Level-2 LOB & Tape Data.
    Extracts exactly 40 scaled Alpha Features:
    - 12x Normalized LOB Prices (p0-p11)
    - 12x Log-Scaled LOB Volumes (v0-v11)
    - 4x Normalized Deal Prices (dp0-dp3)
    - 4x Log-Scaled Deal Volumes (dv0-dv3)
    - 6x Order Flow Imbalance (Log Bid Vol - Log Ask Vol)
    - 1x Aggressive Buy Imbalance
    - 1x Aggressive Sell Imbalance
    """

    def __init__(self, parquet_path: str, sequence_length: int = 1000,
                 batch_size: int = 16, shuffle: bool = True,
                 normalize_prices: bool = True, device: str = 'cpu',
                 max_sequences: int = 500, test_mode: bool = False):

        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_prices = normalize_prices
        self.device = device
        self.max_sequences = max_sequences
        self.test_mode = test_mode

        self.metadata_cols = ['seq_ix', 'step_in_seq', 'need_prediction']
        self.price_cols = [f'p{i}' for i in range(12)]
        self.volume_cols = [f'v{i}' for i in range(12)]
        self.deal_price_cols = [f'dp{i}' for i in range(4)]
        self.deal_volume_cols = [f'dv{i}' for i in range(4)]
        self.target_cols = ['t0', 't1']

        if test_mode:
            self.sequences = self._create_synthetic_sequences()
        else:
            print(f"Loading metadata from {parquet_path}...")
            metadata_df = pd.read_parquet(parquet_path, columns=['seq_ix'])
            self.unique_seq_ids = np.unique(metadata_df['seq_ix'].values)
            if max_sequences < len(self.unique_seq_ids):
                self.unique_seq_ids = self.unique_seq_ids[:max_sequences]

            print(f"Loading and engineering sequence data...")
            self.sequences = self._load_sequences()

        print(f"Loaded {len(self.sequences)} sequence windows")
        self.dataloader = self._create_dataloader()

    def _create_synthetic_sequences(self):
        return [{'features': np.random.randn(self.sequence_length, 40).astype(np.float32),
                 'targets': np.random.randn(self.sequence_length, 2).astype(np.float32),
                 'seq_id': i} for i in range(min(10, self.max_sequences))]

    def _load_sequences(self):
        sequences = []
        for seq_id in self.unique_seq_ids:
            try:
                df = pd.read_parquet(
                    self.parquet_path, filters=[('seq_ix', '=', seq_id)],
                    columns=self.metadata_cols + self.price_cols + self.volume_cols +
                            self.deal_price_cols + self.deal_volume_cols + self.target_cols
                )
            except Exception:
                continue

            if len(df) < self.sequence_length:
                continue

            max_rows = self.sequence_length * 3
            if len(df) > max_rows: df = df.iloc[:max_rows]

            for start_idx in range(0, len(df) - self.sequence_length + 1, self.sequence_length):
                window_data = df.iloc[start_idx:start_idx + self.sequence_length]
                targets = window_data[self.target_cols].values.astype(np.float32)

                features_df = window_data[self.price_cols + self.volume_cols +
                                          self.deal_price_cols + self.deal_volume_cols]
                features = features_df.values.astype(np.float32)

                # 1. Split raw arrays
                prices = features[:, :12]
                volumes = features[:, 12:24]
                deal_prices = features[:, 24:28]
                deal_volumes = features[:, 28:32]

                # 2. Extract Mid-Price
                best_bid = prices[:, 0]
                best_ask = prices[:, 6]
                mid_prices = (best_bid + best_ask) / 2.0
                mid_exp = mid_prices[:, np.newaxis]
                eps = 1e-8

                # 3. Normalize Prices & Deal Prices
                norm_prices = (prices - mid_exp) / (mid_exp + eps)
                norm_deal_prices = np.where(deal_prices > 0, (deal_prices - mid_exp) / (mid_exp + eps), 0.0).astype(
                    np.float32)

                # 4. Log-Scale Volumes (CRITICAL for Neural Networks)
                log_volumes = np.log1p(volumes)
                log_deal_volumes = np.log1p(deal_volumes)

                # 5. Calculate OFI
                ofi_features = log_volumes[:, :6] - log_volumes[:, 6:12]

                # 6. Calculate Aggressive Trade Flow Imbalance
                # Buy order hits the Ask. Sell order hits the Bid.
                is_buy = (deal_prices >= best_ask[:, np.newaxis]) & (deal_prices > 0)
                is_sell = (deal_prices <= best_bid[:, np.newaxis]) & (deal_prices > 0)

                buy_imbalance = np.sum(log_deal_volumes * is_buy, axis=1, keepdims=True)
                sell_imbalance = np.sum(log_deal_volumes * is_sell, axis=1, keepdims=True)

                # 7. Concatenate exactly 40 features
                combined_features = np.concatenate([
                    norm_prices,  # 12
                    log_volumes,  # 12
                    norm_deal_prices,  # 4
                    log_deal_volumes,  # 4
                    ofi_features,  # 6
                    buy_imbalance,  # 1
                    sell_imbalance  # 1
                ], axis=1).astype(np.float32)

                sequences.append({'features': combined_features, 'targets': targets, 'seq_id': seq_id})
                if len(sequences) >= self.max_sequences: return sequences
        return sequences

    def _create_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0,
                          pin_memory=True if self.device == 'cuda' else False)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {'features': torch.from_numpy(seq['features']), 'targets': torch.from_numpy(seq['targets'])}


def create_data_loaders(train_path: str, valid_path: str, sequence_length: int = 1000, batch_size: int = 16,
                        normalize_prices: bool = True, device: str = 'cpu', max_sequences: int = 500,
                        test_mode: bool = False):
    train_dataset = OFIDataLoader(parquet_path=train_path, sequence_length=sequence_length, batch_size=batch_size,
                                  shuffle=True, normalize_prices=normalize_prices, device=device,
                                  max_sequences=max_sequences, test_mode=test_mode)
    valid_dataset = OFIDataLoader(parquet_path=valid_path, sequence_length=sequence_length, batch_size=batch_size,
                                  shuffle=False, normalize_prices=normalize_prices, device=device,
                                  max_sequences=max_sequences, test_mode=test_mode)
    return train_dataset.dataloader, valid_dataset.dataloader