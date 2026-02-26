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

# Import the memory-mapped reader
from .memory_mapped_reader import MemoryMappedParquetReader, create_memory_mapped_reader
# Import vectorized feature functions
from .features import (
    calculate_mid_price,
    normalize_prices,
    process_volume_features,
    calculate_order_flow_features,
    assemble_features
)
# Import vectorized sequence generation
from .sequences import (
    generate_overlapping_sequences,
    generate_batched_sequences,
    SequenceGenerator,
    create_sequence_dataset
)

# Suppress pandas/numpy warnings in production to keep logs clean
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OFIDataLoader(Dataset):
    """
    Data-Centric Loader for Level-2 LOB & Tape Data.
    Acts as an Iron Shield against data toxicity.
    
    Enhanced with memory-mapped Parquet reading for efficient large-scale data access.
    Now with vectorized overlapping sequence generation.
    """

    def __init__(self, parquet_path: str, sequence_length: int = 1000,
                 batch_size: int = 16, shuffle: bool = True,
                 device: str = 'cpu', max_sequences: int = 3000, 
                 test_mode: bool = False, use_memory_mapping: bool = True,
                 stride: int = 1, use_vectorized_generation: bool = True):

        self.parquet_path = parquet_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.max_sequences = max_sequences
        self.test_mode = test_mode
        self.use_memory_mapping = use_memory_mapping
        self.stride = stride
        self.use_vectorized_generation = use_vectorized_generation

        # Schema Definitions
        self.metadata_cols = ['seq_ix', 'step_in_seq', 'need_prediction']
        self.price_cols = [f'p{i}' for i in range(12)]
        self.volume_cols = [f'v{i}' for i in range(12)]
        self.deal_price_cols = [f'dp{i}' for i in range(4)]
        self.deal_volume_cols = [f'dv{i}' for i in range(4)]
        self.target_cols = ['t0', 't1']
        
        # All feature columns
        self.feature_cols = self.price_cols + self.volume_cols + self.deal_price_cols + self.deal_volume_cols
        
        # Memory-mapped reader (initialized lazily)
        self._mm_reader = None

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
                try:
                    self.sequences = torch.load(cache_file)
                    # Check if cache is not empty
                    if len(self.sequences) == 0:
                        raise ValueError("Cache file is empty")
                except (EOFError, RuntimeError, ValueError, Exception) as e:
                    logger.warning(f"Cache file {cache_file} is corrupted or empty (error: {e}). Deleting and regenerating...")
                    try:
                        os.remove(cache_file)
                        logger.warning(f"Successfully deleted corrupted cache file: {cache_file}")
                    except Exception as delete_error:
                        logger.warning(f"Failed to delete corrupted cache file {cache_file}: {delete_error}")
                    
                    # Fall back to regenerating sequences from raw data
                    print(f"🐢 [CACHE CORRUPTED] Loading and engineering sequences from raw data. This will take a while...")
                    self.unique_seq_ids = self._get_unique_sequence_ids()
                    if max_sequences < len(self.unique_seq_ids):
                        self.unique_seq_ids = self.unique_seq_ids[:max_sequences]

                    self.sequences = self._load_sequences()

                    print(f"💾 Saving engineered sequences to {cache_file} for instant loading next time...")
                    torch.save(self.sequences, cache_file)
            else:
                # Cleanup obsolete caches
                old_caches = glob.glob(os.path.join(cache_dir, f"cache_{split_name}_{max_sequences}_*.pt"))
                for old_file in old_caches:
                    try:
                        os.remove(old_file)
                        print(f"🗑️ [CACHE CLEANUP] Deleted obsolete memory: {os.path.basename(old_file)}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old cache file {old_file}: {e}")

                print(f"🐢 [CACHE MISS] Loading and engineering sequences. This will take a while...")
                self.unique_seq_ids = self._get_unique_sequence_ids()
                if max_sequences < len(self.unique_seq_ids):
                    self.unique_seq_ids = self.unique_seq_ids[:max_sequences]

                self.sequences = self._load_sequences()

                print(f"💾 Saving engineered sequences to {cache_file} for instant loading next time...")
                torch.save(self.sequences, cache_file)

        print(f"Successfully loaded {len(self.sequences)} valid sequence windows.")
        self.dataloader = self._create_dataloader()
    
    def _get_memory_mapped_reader(self) -> MemoryMappedParquetReader:
        """Get or create the memory-mapped reader."""
        if self._mm_reader is None:
            self._mm_reader = create_memory_mapped_reader(self.parquet_path)
        return self._mm_reader
    
    def _get_unique_sequence_ids(self) -> np.ndarray:
        """Get unique sequence IDs using memory-mapped reader or pandas."""
        if self.use_memory_mapping:
            try:
                reader = self._get_memory_mapped_reader()
                unique_seq_ids = reader.get_unique_sequences('seq_ix')
                print(f"✓ Retrieved {len(unique_seq_ids)} unique sequences using memory-mapped reader")
                return unique_seq_ids
            except Exception as e:
                logger.warning(f"Memory-mapped reader failed, falling back to pandas: {e}")
                self.use_memory_mapping = False
        
        # Fallback to pandas
        metadata_df = pd.read_parquet(self.parquet_path, columns=['seq_ix'])
        unique_seq_ids = np.unique(metadata_df['seq_ix'].values)
        print(f"✓ Retrieved {len(unique_seq_ids)} unique sequences using pandas")
        return unique_seq_ids
    
    def _load_sequence_data_memory_mapped(self, seq_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Load sequence data using memory-mapped reader."""
        try:
            reader = self._get_memory_mapped_reader()
            
            # Get all columns for this sequence
            all_columns = self.metadata_cols + self.feature_cols + self.target_cols
            seq_data = reader.get_sequence_data(seq_id, seq_id_column='seq_ix')
            
            # Filter to only the columns we need
            filtered_data = {}
            for col in all_columns:
                if col in seq_data and len(seq_data[col]) > 0:
                    filtered_data[col] = seq_data[col]
            
            if not filtered_data or len(filtered_data.get('seq_ix', [])) == 0:
                return None
            
            return filtered_data
            
        except Exception as e:
            logger.warning(f"Failed to load sequence {seq_id} with memory-mapped reader: {e}")
            return None
    
    def _load_sequence_data_pandas(self, seq_id: int) -> Optional[pd.DataFrame]:
        """Load sequence data using pandas (fallback)."""
        try:
            df = pd.read_parquet(
                self.parquet_path, filters=[('seq_ix', '=', seq_id)],
                columns=self.metadata_cols + self.feature_cols + self.target_cols
            )
            return df if len(df) > 0 else None
        except Exception as e:
            logger.warning(f"Failed to load sequence {seq_id} with pandas: {e}")
            return None

    def _create_synthetic_sequences(self) -> List[Dict[str, Any]]:
        """Fallback synthetic generator ensuring exactly 40 features."""
        return [{'features': np.zeros((self.sequence_length, 40), dtype=np.float32),
                 'targets': np.zeros((self.sequence_length, 2), dtype=np.float32)}
                for _ in range(min(10, self.max_sequences))]

    def _generate_sequences_vectorized(self, df_dict: Dict[str, np.ndarray], df_length: int) -> List[Dict[str, Any]]:
        """
        Generate sequences using vectorized overlapping sequence generation.
        
        This method uses the new vectorized sequence generation functions
        for improved performance and memory efficiency.
        """
        sequences = []
        
        # Extract raw features and targets
        features_raw_list = []
        for col in self.feature_cols:
            if col in df_dict:
                features_raw_list.append(df_dict[col])
        
        if not features_raw_list:
            return sequences
        
        # Stack features
        features_raw = np.column_stack(features_raw_list).astype(np.float32)
        
        # Extract targets
        targets_list = []
        for col in self.target_cols:
            if col in df_dict:
                targets_list.append(df_dict[col])
        
        if not targets_list:
            return sequences
        
        targets_raw = np.column_stack(targets_list).astype(np.float32)
        
        # Apply NaN cleaning
        features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
        targets_raw = np.nan_to_num(targets_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if we have enough data
        if df_length < self.sequence_length + 1:
            return sequences
        
        # Limit data size if too large
        max_rows = (self.sequence_length * 3) + 1
        if df_length > max_rows:
            features_raw = features_raw[:max_rows]
            targets_raw = targets_raw[:max_rows]
            df_length = max_rows
        
        # Generate sequences using vectorized function
        try:
            features_windows, target_windows = generate_overlapping_sequences(
                data=features_raw,
                window_size=self.sequence_length,
                stride=self.stride,
                target_offset=1,
                target_data=targets_raw
            )
            
            # Process each window
            for i in range(len(features_windows)):
                features_window = features_windows[i]
                targets_window = target_windows[i]
                
                # Extract individual feature groups
                prices = features_window[:, :12]
                volumes = features_window[:, 12:24]
                deal_prices = features_window[:, 24:28]
                deal_volumes = features_window[:, 28:32]

                # ==========================================
                # VECTORIZED FEATURE ENGINEERING
                # ==========================================
                # Calculate best bid and ask
                best_bid = prices[:, 0]
                best_ask = prices[:, 6]
                
                # Calculate mid price
                mid_prices = calculate_mid_price(best_bid, best_ask)
                
                # Normalize prices
                norm_prices, norm_deal_prices = normalize_prices(
                    prices, deal_prices, mid_prices
                )
                
                # Process volume features
                log_volumes, log_deal_volumes = process_volume_features(volumes, deal_volumes)
                
                # Calculate order flow features (OFI and trade imbalances)
                ofi_features, buy_imbalance, sell_imbalance = calculate_order_flow_features(
                    volumes, deal_volumes, deal_prices, best_bid, best_ask
                )
                
                # Assemble all features
                combined = assemble_features(
                    norm_prices, log_volumes, norm_deal_prices,
                    log_deal_volumes, ofi_features, buy_imbalance, sell_imbalance
                )
                
                sequences.append({
                    'features': combined,
                    'targets': targets_window,
                    'seq_id': df_dict.get('seq_ix', [0])[0] if 'seq_ix' in df_dict else 0
                })
                
        except Exception as e:
            logger.warning(f"Vectorized sequence generation failed: {e}")
            # Fall back to traditional method
            return self._generate_sequences_traditional(df_dict, df_length)
        
        return sequences
    
    def _generate_sequences_traditional(self, df_dict: Dict[str, np.ndarray], df_length: int) -> List[Dict[str, Any]]:
        """
        Traditional sequence generation method (fallback).
        """
        sequences = []
        
        # Traditional windowing approach
        for start_idx in range(0, df_length - self.sequence_length, self.sequence_length):
            # Extract window for features (rows 0 to 999) and targets (rows 1 to 1000)
            feature_end_idx = start_idx + self.sequence_length
            target_start_idx = start_idx + 1
            target_end_idx = start_idx + self.sequence_length + 1
            
            # Extract features from all feature columns
            features_raw_list = []
            for col in self.feature_cols:
                if col in df_dict:
                    features_raw_list.append(df_dict[col][start_idx:feature_end_idx])
            
            if not features_raw_list:
                continue
            
            features_raw = np.column_stack(features_raw_list).astype(np.float32)
            
            # Extract targets
            targets_list = []
            for col in self.target_cols:
                if col in df_dict:
                    targets_list.append(df_dict[col][target_start_idx:target_end_idx])
            
            if not targets_list:
                continue
            
            targets = np.column_stack(targets_list).astype(np.float32)

            # ==========================================
            # SCRUBBER 1: Kill NaNs from raw data
            # ==========================================
            features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract individual feature groups
            prices = features_raw[:, :12]
            volumes = features_raw[:, 12:24]
            deal_prices = features_raw[:, 24:28]
            deal_volumes = features_raw[:, 28:32]

            # ==========================================
            # VECTORIZED FEATURE ENGINEERING
            # ==========================================
            # Calculate best bid and ask
            best_bid = prices[:, 0]
            best_ask = prices[:, 6]
            
            # Calculate mid price
            mid_prices = calculate_mid_price(best_bid, best_ask)
            
            # Normalize prices
            norm_prices, norm_deal_prices = normalize_prices(
                prices, deal_prices, mid_prices
            )
            
            # Process volume features
            log_volumes, log_deal_volumes = process_volume_features(volumes, deal_volumes)
            
            # Calculate order flow features (OFI and trade imbalances)
            ofi_features, buy_imbalance, sell_imbalance = calculate_order_flow_features(
                volumes, deal_volumes, deal_prices, best_bid, best_ask
            )
            
            # Assemble all features
            combined = assemble_features(
                norm_prices, log_volumes, norm_deal_prices,
                log_deal_volumes, ofi_features, buy_imbalance, sell_imbalance
            )

            sequences.append({
                'features': combined,
                'targets': targets,
                'seq_id': df_dict.get('seq_ix', [0])[0] if 'seq_ix' in df_dict else 0
            })
        
        return sequences

    def _load_sequences(self) -> List[Dict[str, Any]]:
        sequences = []
        sequences_loaded = 0
        sequences_skipped = 0
        
        for seq_id in self.unique_seq_ids:
            try:
                # Try memory-mapped reader first if enabled
                if self.use_memory_mapping:
                    seq_data = self._load_sequence_data_memory_mapped(seq_id)
                    if seq_data is not None:
                        # Convert dict to DataFrame-like structure for processing
                        df_dict = seq_data
                        df_length = len(seq_data['seq_ix'])
                    else:
                        # Fall back to pandas
                        df = self._load_sequence_data_pandas(seq_id)
                        if df is None:
                            sequences_skipped += 1
                            continue
                        df_dict = {col: df[col].values for col in df.columns}
                        df_length = len(df)
                else:
                    # Use pandas directly
                    df = self._load_sequence_data_pandas(seq_id)
                    if df is None:
                        sequences_skipped += 1
                        continue
                    df_dict = {col: df[col].values for col in df.columns}
                    df_length = len(df)
                
                # Check if we have enough data
                if df_length < self.sequence_length + 1:
                    sequences_skipped += 1
                    continue

                # Limit data size if too large
                max_rows = (self.sequence_length * 3) + 1
                if df_length > max_rows:
                    for col in df_dict:
                        df_dict[col] = df_dict[col][:max_rows]
                    df_length = max_rows

                # Generate sequences using appropriate method
                if self.use_vectorized_generation:
                    seq_batch = self._generate_sequences_vectorized(df_dict, df_length)
                else:
                    seq_batch = self._generate_sequences_traditional(df_dict, df_length)
                
                sequences.extend(seq_batch)
                sequences_loaded += len(seq_batch)
                
                if sequences_loaded >= self.max_sequences:
                    print(f"✓ Loaded {sequences_loaded} sequences (skipped {sequences_skipped})")
                    return sequences[:self.max_sequences]
                    
            except Exception as e:
                print(f"Warning: Skipping seq_ix {seq_id} due to error: {e}")
                sequences_skipped += 1
                continue
        
        print(f"✓ Loaded {sequences_loaded} sequences (skipped {sequences_skipped})")
        return sequences[:self.max_sequences] if self.max_sequences < len(sequences) else sequences

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
    
    def __del__(self):
        """Clean up memory-mapped reader when object is destroyed."""
        if self._mm_reader is not None:
            self._mm_reader.close()
            self._mm_reader = None
        gc.collect()


def create_data_loaders(train_path: str, valid_path: str, sequence_length: int = 1000,
                        batch_size: int = 16, normalize_prices: bool = True, device: str = 'cpu',
                        max_sequences: int = 3000, test_mode: bool = False,
                        use_memory_mapping: bool = True, stride: int = 1,
                        use_vectorized_generation: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_path: Path to training Parquet file
        valid_path: Path to validation Parquet file
        sequence_length: Length of each sequence window
        batch_size: Batch size for DataLoader
        normalize_prices: Whether to normalize price features
        device: Target device ('cpu' or 'cuda')
        max_sequences: Maximum number of sequences to load
        test_mode: Whether to use synthetic test data
        use_memory_mapping: Whether to use memory-mapped reading
        stride: Step size between consecutive sequence windows
        use_vectorized_generation: Whether to use vectorized sequence generation
        
    Returns:
        Tuple of (train_dataloader, valid_dataloader)
    """
    train_dataset = OFIDataLoader(
        parquet_path=train_path, sequence_length=sequence_length, batch_size=batch_size,
        shuffle=True, device=device, max_sequences=max_sequences, 
        test_mode=test_mode, use_memory_mapping=use_memory_mapping,
        stride=stride, use_vectorized_generation=use_vectorized_generation
    )
    valid_dataset = OFIDataLoader(
        parquet_path=valid_path, sequence_length=sequence_length, batch_size=batch_size,
        shuffle=False, device=device, max_sequences=max_sequences, 
        test_mode=test_mode, use_memory_mapping=use_memory_mapping,
        stride=stride, use_vectorized_generation=use_vectorized_generation
    )
    return train_dataset.dataloader, valid_dataset.dataloader


# Backward compatibility function
def create_memory_efficient_loader(parquet_path: str, **kwargs) -> OFIDataLoader:
    """
    Create a memory-efficient data loader using memory-mapped reading.
    
    Args:
        parquet_path: Path to Parquet file
        **kwargs: Additional arguments for OFIDataLoader
        
    Returns:
        OFIDataLoader instance with memory mapping enabled
    """
    kwargs['use_memory_mapping'] = True
    return OFIDataLoader(parquet_path, **kwargs)


# New function for vectorized sequence generation
def create_vectorized_sequence_loader(
    data: np.ndarray,
    targets: np.ndarray,
    window_size: int = 1000,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = 'cpu'
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader using vectorized sequence generation.
    
    Args:
        data: Input features of shape (n_samples, n_features)
        targets: Target values of shape (n_samples, n_targets)
        window_size: Size of each sequence window
        stride: Step size between consecutive windows
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle sequences
        device: Target device for tensors
        
    Returns:
        PyTorch DataLoader
    """
    dataset = create_sequence_dataset(
        data=data,
        window_size=window_size,
        stride=stride,
        target_offset=1,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        target_data=targets
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )