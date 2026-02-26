import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import warnings
import hashlib
import glob

# Suppress pandas/numpy warnings in production to keep logs clean
warnings.filterwarnings('ignore')


class OFIDataLoader(Dataset):
    """
    Data-Centric Loader for Level-2 LOB & Tape Data.
    Acts as an Iron Shield against data toxicity.
    """

    def __init__(self, parquet_path: str, sequence_length: int = 1000,
                 batch_size: int = 16, shuffle: bool = True,
                 device: str = 'cpu', max_sequences: int = 3000, test_mode: bool = False):

        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.max_sequences = max_sequences
        self.test_mode = test_mode

        # Schema Definitions
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
            # ==========================================
            # TRIZ PRINCIPLE 25 & 34: SELF-AWARE DNA CACHING
            # ==========================================
            split_name = "train" if shuffle else "valid"
            cache_dir = "data/processed"
            os.makedirs(cache_dir, exist_ok=True)

            # Calculate DNA of this file
            current_file_path = os.path.abspath(__file__)
            with open(current_file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]

            cache_file = os.path.join(cache_dir, f"cache_{split_name}_{max_sequences}_{file_hash}.pt")

            if os.path.exists(cache_file):
                print(f"⚡ [CACHE HIT] DNA match '{file_hash}'. Loading {split_name} sequences instantly...")
                self.sequences = torch.load(cache_file)
            else:
                # Cleanup obsolete caches
                old_caches = glob.glob(os.path.join(cache_dir, f"cache_{split_name}_{max_sequences}_*.pt"))
                for old_file in old_caches:
                    os.remove(old_file)
                    print(f"🗑️ [CACHE CLEANUP] Deleted obsolete memory: {os.path.basename(old_file)}")

                print(f"🐢 [CACHE MISS] Loading and engineering sequences. This will take a while...")
                metadata_df = pd.read_parquet(parquet_path, columns=['seq_ix'])
                self.unique_seq_ids = np.unique(metadata_df['seq_ix'].values)
                if max_sequences < len(self.unique_seq_ids):
                    self.unique_seq_ids = self.unique_seq_ids[:max_sequences]

                self.sequences = self._load_sequences()

                print(f"💾 Saving engineered sequences to {cache_file} for instant loading next time...")
                torch.save(self.sequences, cache_file)

        print(f"Successfully loaded {len(self.sequences)} valid sequence windows.")
        self.dataloader = self._create_dataloader()

    def _create_synthetic_sequences(self) -> List[Dict[str, Any]]:
        """Fallback synthetic generator ensuring exactly 40 features."""
        return [{'features': np.zeros((self.sequence_length, 40), dtype=np.float32),
                 'targets': np.zeros((self.sequence_length, 2), dtype=np.float32)}
                for _ in range(min(10, self.max_sequences))]

    def _load_sequences(self) -> List[Dict[str, Any]]:
        sequences = []
        for seq_id in self.unique_seq_ids:
            try:
                df = pd.read_parquet(
                    self.parquet_path, filters=[('seq_ix', '=', seq_id)],
                    columns=self.metadata_cols + self.price_cols + self.volume_cols +
                            self.deal_price_cols + self.deal_volume_cols + self.target_cols
                )
            except Exception as e:
                print(f"Warning: Skipping seq_ix {seq_id} due to read error: {e}")
                continue

            # 🚨 THE FUTURE SHIFT REQUIREMENT: We need len + 1 to shift targets
            if len(df) < self.sequence_length + 1:
                continue

            max_rows = (self.sequence_length * 3) + 1
            if len(df) > max_rows:
                df = df.iloc[:max_rows]

            # 🚨 THE FUTURE SHIFT 🚨
            # We iterate requiring a window of (sequence_length + 1)
            for start_idx in range(0, len(df) - self.sequence_length, self.sequence_length):
                window = df.iloc[start_idx:start_idx + self.sequence_length + 1]

                # Features take row 0 to 999
                features_raw = window.iloc[:-1][
                    self.price_cols + self.volume_cols + self.deal_price_cols + self.deal_volume_cols].values.astype(
                    np.float32)

                # Targets take row 1 to 1000 (The Future)
                targets = window.iloc[1:][self.target_cols].values.astype(np.float32)

                # ==========================================
                # SCRUBBER 1: Kill NaNs from raw data
                # ==========================================
                features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
                targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

                prices = features_raw[:, :12]
                volumes = np.clip(features_raw[:, 12:24], 0, None)
                deal_prices = features_raw[:, 24:28]
                deal_volumes = np.clip(features_raw[:, 28:32], 0, None)

                # ==========================================
                # MID-PRICE CALCULATION
                # ==========================================
                best_bid = prices[:, 0]
                best_ask = prices[:, 6]
                mid_prices = (best_bid + best_ask) / 2.0

                mid_prices[mid_prices == 0] = 1.0
                mid_exp = mid_prices[:, np.newaxis]
                eps = 1e-5

                # ==========================================
                # STABILIZED FEATURE ENGINEERING
                # ==========================================
                norm_prices = (prices - mid_exp) / (mid_exp + eps)

                norm_deal_prices = (deal_prices - mid_exp) / (mid_exp + eps)
                trade_mask = deal_prices > 0
                norm_deal_prices[~trade_mask] = 0.0

                log_volumes = np.log1p(volumes)
                log_deal_volumes = np.log1p(deal_volumes)

                ofi_features = log_volumes[:, :6] - log_volumes[:, 6:12]

                is_buy = (deal_prices >= best_ask[:, np.newaxis]) & trade_mask
                is_sell = (deal_prices <= best_bid[:, np.newaxis]) & trade_mask
                buy_imbalance = np.sum(log_deal_volumes * is_buy, axis=1, keepdims=True)
                sell_imbalance = np.sum(log_deal_volumes * is_sell, axis=1, keepdims=True)

                # ==========================================
                # FINAL ASSEMBLY & IRON SHIELD
                # ==========================================
                combined = np.concatenate([
                    norm_prices, log_volumes, norm_deal_prices,
                    log_deal_volumes, ofi_features, buy_imbalance, sell_imbalance
                ], axis=1).astype(np.float32)

                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                combined = np.clip(combined, -20.0, 20.0)

                sequences.append({'features': combined, 'targets': targets, 'seq_id': seq_id})
                if len(sequences) >= self.max_sequences:
                    return sequences
        return sequences

    def _create_dataloader(self) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            'features': torch.from_numpy(seq['features']),
            'targets': torch.from_numpy(seq['targets'])
        }


def create_data_loaders(train_path: str, valid_path: str, sequence_length: int = 1000,
                        batch_size: int = 16, normalize_prices: bool = True, device: str = 'cpu',
                        max_sequences: int = 3000, test_mode: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_dataset = OFIDataLoader(
        parquet_path=train_path, sequence_length=sequence_length, batch_size=batch_size,
        shuffle=True, device=device, max_sequences=max_sequences, test_mode=test_mode
    )
    valid_dataset = OFIDataLoader(
        parquet_path=valid_path, sequence_length=sequence_length, batch_size=batch_size,
        shuffle=False, device=device, max_sequences=max_sequences, test_mode=test_mode
    )
    return train_dataset.dataloader, valid_dataset.dataloader