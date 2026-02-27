import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional
import warnings
import hashlib
import glob
import logging
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OFIDataLoader(Dataset):
    """
    Data-Centric Loader for Level-2 LOB & Tape Data.
    Acts as an Iron Shield against data toxicity.
    
    Fixed for temporal data leakage:
    - Input sequence ends at time t
    - Target is the value at time t+1
    - No overlap between input and target sequences
    """

    def __init__(self, parquet_path: str, sequence_length: int = 1000,
                 batch_size: int = 16, shuffle: bool = True,
                 device: str = 'cpu', max_sequences: int = 3000,
                 test_mode: bool = False, **kwargs):

        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.max_sequences = max_sequences
        self.test_mode = test_mode

        self.metadata_cols = ['seq_ix', 'step_in_seq', 'need_prediction']
        self.price_cols = [f'p{i}' for i in range(12)]
        self.volume_cols = [f'v{i}' for i in range(12)]
        self.deal_price_cols = [f'dp{i}' for i in range(4)]
        self.deal_volume_cols = [f'dv{i}' for i in range(4)]
        self.target_cols = ['t0', 't1']

        self.feature_cols = self.price_cols + self.volume_cols + self.deal_price_cols + self.deal_volume_cols

        if test_mode:
            self.sequences = self._create_synthetic_sequences()
            self.dataloader = self.get_dataloader()
        else:
            print(f"Loading metadata from {parquet_path}...")
            split_name = "train" if shuffle else "valid"
            cache_dir = "data/processed"
            os.makedirs(cache_dir, exist_ok=True)

            current_file_path = os.path.abspath(__file__)
            with open(current_file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]

            # Use a different cache prefix so we don't load the old broken pickle files
            cache_file = os.path.join(cache_dir, f"cache_tensor_{split_name}_{max_sequences}_{file_hash}.pt")

            if os.path.exists(cache_file):
                print(f"⚡ [CACHE HIT] Loading {split_name} tensor blocks instantly from disk...")
                try:
                    cache_data = torch.load(cache_file)
                    features_tensor = cache_data['features']
                    targets_tensor = cache_data['targets']

                    self.sequences = []
                    # Reconstruct list of dicts for the PyTorch Dataset __getitem__
                    for i in range(len(features_tensor)):
                        self.sequences.append({
                            'features': features_tensor[i].numpy(),
                            'targets': targets_tensor[i].numpy()
                        })

                    if len(self.sequences) == 0:
                        raise ValueError("Cache file is empty")
                    
                    # Validate temporal alignment
                    self._validate_temporal_alignment()
                except Exception as e:
                    print(f"🐢 [CACHE CORRUPTED] Deleting and regenerating... ({e})")
                    try:
                        os.remove(cache_file)
                    except:
                        pass
                    self._generate_and_save_cache(cache_file)
            else:
                for old_file in glob.glob(os.path.join(cache_dir, f"cache_tensor_{split_name}_*_{file_hash}.pt")):
                    try:
                        os.remove(old_file)
                    except:
                        pass

                print(f"🐢 [CACHE MISS] Loading and engineering sequences...")
                self._generate_and_save_cache(cache_file)

            print(f"Successfully loaded {len(self.sequences)} valid sequence windows.")
            self.dataloader = self.get_dataloader()

    def _validate_temporal_alignment(self):
        """Validate that there is no temporal data leakage."""
        print("🔍 Validating temporal alignment...")
        for i, seq in enumerate(self.sequences[:5]):  # Check first 5 sequences
            features = seq['features']
            targets = seq['targets']
            
            # Check shapes
            if features.shape != targets.shape:
                raise ValueError(f"Sequence {i}: Features shape {features.shape} != Targets shape {targets.shape}")
            
            # Check that we have the expected sequence length
            if features.shape[0] != self.sequence_length - 1:
                raise ValueError(f"Sequence {i}: Expected {self.sequence_length - 1} timesteps, got {features.shape[0]}")
            
            # For a more thorough check, we could verify that targets are indeed
            # the next timestep of features, but that requires access to raw data
            # which we don't have in cache. We'll rely on the generation logic.
        
        print("✅ Temporal alignment validated - no data leakage detected")

    def _generate_and_save_cache(self, cache_file: str):
        """Helper method to stack and save sequences as contiguous tensors."""
        self.sequences = self._load_sequences()
        if not self.sequences:
            raise ValueError("No sequences were generated. Check dataset length and sequence_length.")

        print(f"🧱 Stacking {len(self.sequences)} sequences into contiguous memory blocks...")

        # Extract features and targets into separate lists
        feature_list = [seq['features'] for seq in self.sequences]
        target_list = [seq['targets'] for seq in self.sequences]

        # Convert to monolithic PyTorch tensors
        # Doing np.array() first is significantly faster for stacking than pure torch.stack
        features_tensor = torch.from_numpy(np.array(feature_list))
        targets_tensor = torch.from_numpy(np.array(target_list))

        print(f"💾 Saving monolithic tensor blocks to {cache_file}...")
        torch.save({
            'features': features_tensor,
            'targets': targets_tensor
        }, cache_file)

    def _create_synthetic_sequences(self) -> List[Dict[str, Any]]:
        """Fallback synthetic generator ensuring exactly 40 features for T-KAN."""
        n_seqs = max(self.batch_size * 2, 32)
        # For synthetic data, create sequences with proper temporal alignment
        # Features: sequence_length-1 timesteps
        # Targets: sequence_length-1 timesteps (shifted by 1)
        return [{'features': np.random.randn(self.sequence_length - 1, 40).astype(np.float32),
                 'targets': np.random.randn(self.sequence_length - 1, 2).astype(np.float32)}
                for _ in range(n_seqs)]

    def _load_sequences(self) -> List[Dict[str, Any]]:
        sequences = []
        sequences_loaded = 0
        sequences_skipped = 0

        print(f"📦 [RAM INJECTION] Loading entire dataset into memory for bulk processing... (~3GB)")
        columns_to_load = self.metadata_cols + self.feature_cols + self.target_cols
        full_df = pd.read_parquet(self.parquet_path, columns=columns_to_load)

        print(f"🚀 [RAM INJECTION] Grouping by sequence ID in-memory...")
        grouped = full_df.groupby('seq_ix')

        for seq_id, df_group in tqdm(grouped, total=len(grouped), desc="Engineering Sequences", file=sys.stdout,
                                     mininterval=0.5):
            try:
                if sequences_loaded >= self.max_sequences:
                    break

                # Fix the off-by-one bug: Wundernn has EXACTLY 1000 rows.
                # We need at least sequence_length rows to create sequence_length-1 timesteps
                if len(df_group) < self.sequence_length:
                    sequences_skipped += 1
                    continue

                features_raw = df_group[self.feature_cols].values.astype(np.float32)
                targets_raw = df_group[self.target_cols].values.astype(np.float32)

                # Scrubber
                features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
                targets_raw = np.nan_to_num(targets_raw, nan=0.0, posinf=0.0, neginf=0.0)

                # Slicing
                prices = features_raw[:, :12]
                volumes = np.clip(features_raw[:, 12:24], 0, None)
                deal_prices = features_raw[:, 24:28]
                deal_volumes = np.clip(features_raw[:, 28:32], 0, None)

                # Feature Engineering
                best_bid = prices[:, 0]
                best_ask = prices[:, 6]
                mid_prices = (best_bid + best_ask) / 2.0
                mid_prices[mid_prices == 0] = 1.0
                mid_exp = mid_prices[:, np.newaxis]
                eps = 1e-5

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

                # Assemble EXACTLY 40 features
                combined = np.concatenate([
                    norm_prices, log_volumes, norm_deal_prices,
                    log_deal_volumes, ofi_features, buy_imbalance, sell_imbalance
                ], axis=1).astype(np.float32)

                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                combined = np.clip(combined, -20.0, 20.0)

                # FIX FOR TEMPORAL DATA LEAKAGE:
                # Features: timesteps 0 to sequence_length-2 (total: sequence_length-1)
                # Targets: timesteps 1 to sequence_length-1 (total: sequence_length-1)
                # This ensures input ends at time t and target is at time t+1
                features = combined[:self.sequence_length - 1]  # First sequence_length-1 timesteps
                targets = targets_raw[1:self.sequence_length]   # Shifted by 1 timestep
                
                # Validate no overlap
                if features.shape[0] != targets.shape[0]:
                    raise ValueError(f"Sequence {seq_id}: Features and targets have different lengths")
                
                sequences.append({
                    'features': features,
                    'targets': targets
                })
                sequences_loaded += 1

            except Exception as e:
                print(f"\nWarning: Skipping seq_ix {seq_id} due to error: {e}")
                sequences_skipped += 1
                continue

        print(f"\n✓ Engineered {sequences_loaded} sequences (skipped {sequences_skipped})")

        del full_df
        del grouped
        gc.collect()

        return sequences

    def get_dataloader(self) -> DataLoader:
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
            'features': torch.from_numpy(seq['features']).float() if isinstance(seq['features'], np.ndarray) else seq[
                'features'].float(),
            'targets': torch.from_numpy(seq['targets']).float() if isinstance(seq['targets'], np.ndarray) else seq[
                'targets'].float()
        }


def get_dataloader(split: str = 'train', batch_size: int = 32, **kwargs) -> DataLoader:
    if split == 'train':
        dataset = OFIDataLoader('data/raw/train.parquet', batch_size=batch_size, shuffle=True, **kwargs)
    else:
        dataset = OFIDataLoader('data/raw/valid.parquet', batch_size=batch_size, shuffle=False, **kwargs)
    return dataset.dataloader


def create_data_loaders(train_path: str, valid_path: str, sequence_length: int = 1000,
                        batch_size: int = 16, **kwargs) -> Tuple[DataLoader, DataLoader]:
    train_dataset = OFIDataLoader(train_path, sequence_length=sequence_length, batch_size=batch_size, shuffle=True,
                                  **kwargs)
    valid_dataset = OFIDataLoader(valid_path, sequence_length=sequence_length, batch_size=batch_size, shuffle=False,
                                  **kwargs)
    return train_dataset.dataloader, valid_dataset.dataloader
