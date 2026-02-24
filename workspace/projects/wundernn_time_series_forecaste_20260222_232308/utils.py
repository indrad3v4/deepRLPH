import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union
import warnings


def normalize_data(data: np.ndarray, method: str = 'zscore', axis: int = 0, 
                   eps: float = 1e-8) -> np.ndarray:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data array
        method: Normalization method ('zscore', 'minmax', 'robust', 'unit')
        axis: Axis along which to compute statistics (0 for features, 1 for samples)
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized data array with same shape as input
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if method == 'zscore':
        # Standardization: (x - mean) / std
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / np.maximum(std, eps)
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        return (data - data_min) / np.maximum(data_max - data_min, eps)
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (data - median) / np.maximum(iqr, eps)
    
    elif method == 'unit':
        # Unit vector normalization (L2 norm)
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return data / np.maximum(norm, eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                         f"Choose from ['zscore', 'minmax', 'robust', 'unit']")


def create_sequences(data: np.ndarray, sequence_length: int, 
                     target_length: int = 1, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    Args:
        data: Input data of shape (timesteps, features)
        sequence_length: Length of input sequences
        target_length: Length of target sequences
        step: Step size between sequences
        
    Returns:
        X: Input sequences of shape (n_sequences, sequence_length, features)
        y: Target sequences of shape (n_sequences, target_length, features)
    """
    X, y = [], []
    for i in range(0, len(data) - sequence_length - target_length + 1, step):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + target_length])
    
    return np.array(X), np.array(y)


def split_sequences(data: np.ndarray, train_ratio: float = 0.7, 
                    val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        train_data, val_data, test_data
    """
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    return data[:train_end], data[train_end:val_end], data[val_end:]


def calculate_statistics(data: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate comprehensive statistics for data.
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'skewness': pd.Series(data.flatten()).skew() if len(data.shape) > 1 else pd.Series(data).skew(),
        'kurtosis': pd.Series(data.flatten()).kurtosis() if len(data.shape) > 1 else pd.Series(data).kurtosis(),
    }
    
    # Add shape information
    stats['shape'] = data.shape
    stats['n_samples'] = len(data)
    
    if len(data.shape) > 1:
        stats['n_features'] = data.shape[1]
    
    return stats


def detect_outliers(data: np.ndarray, method: str = 'iqr', 
                    threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data.
    
    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean mask indicating outliers
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    
    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def handle_missing_values(data: np.ndarray, method: str = 'mean', 
                          axis: int = 0) -> np.ndarray:
    """
    Handle missing values (NaN) in data.
    
    Args:
        data: Input data with possible NaN values
        method: Imputation method ('mean', 'median', 'zero', 'forward', 'backward')
        axis: Axis along which to compute statistics
        
    Returns:
        Data with missing values imputed
    """
    if not np.isnan(data).any():
        return data
    
    data_filled = data.copy()
    
    if method == 'mean':
        fill_value = np.nanmean(data, axis=axis, keepdims=True)
    elif method == 'median':
        fill_value = np.nanmedian(data, axis=axis, keepdims=True)
    elif method == 'zero':
        fill_value = 0
    elif method == 'forward':
        # Forward fill
        mask = np.isnan(data_filled)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        data_filled[mask] = data_filled[idx[mask], np.nonzero(mask)[1]]
        return data_filled
    elif method == 'backward':
        # Backward fill
        mask = np.isnan(data_filled)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], mask.shape[0] - 1)
        idx = np.minimum.accumulate(idx[::-1, :], axis=0)[::-1, :]
        data_filled[mask] = data_filled[idx[mask], np.nonzero(mask)[1]]
        return data_filled
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    # Fill NaN values
    nan_mask = np.isnan(data_filled)
    data_filled[nan_mask] = fill_value
    
    return data_filled


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Weighted Pearson Correlation Coefficient.

    This metric emphasizes performance on data points with larger target amplitudes
    (larger price movements) by using the absolute value of the target as a sample weight.

    Predictions are clipped to the range [-6, 6] before calculation to prevent
    outliers from dominating the metric.

    Args:
        y_true: Ground truth target values (numpy array).
        y_pred: Predicted values (numpy array).

    Returns:
        float: Weighted Pearson correlation coefficient.
    """
    # Clip predictions to valid range [-6, 6]
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)

    # Calculate weights based on target amplitude
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)

    # Calculate weighted means
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0

    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w

    # Calculate weighted deviations
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred

    # Calculate weighted covariance
    cov = np.sum(weights * dev_true * dev_pred) / sum_w

    # Calculate weighted variances
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w

    # Compute correlation
    if var_true <= 0 or var_pred <= 0:
        return 0.0

    corr = cov / (np.sqrt(var_true) * np.sqrt(var_pred))
    
    return float(corr)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Weighted Pearson correlation
    metrics['weighted_pearson'] = weighted_pearson_correlation(y_true, y_pred)
    
    # Regular Pearson correlation
    if len(y_true) > 1:
        corr_matrix = np.corrcoef(y_true, y_pred)
        metrics['pearson'] = corr_matrix[0, 1]
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Percentage Error
    eps = 1e-8
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + eps))
    
    return metrics


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class ScorerStepByStep:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)

        # Calc feature dimension: first 3 columns are seq_ix, step_in_seq & need_prediction
        # Total columns: 3 metadata + 32 features + 2 targets = 37
        # Features are cols [3:35]
        self.dim = 2
        self.features = self.dataset.columns[3:35]
        self.targets = self.dataset.columns[35:]

    def score(self, model) -> dict:
        predictions = []
        targets = []

        prediction = None

        # Iterate over numpy array for speed
        for row in tqdm(self.dataset.values):
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            lob_data = row[3:35]  # 32 features
            labels = row[35:]     # 2 targets
            #
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            prediction = model.predict(data_point)

            self.check_prediction(data_point, prediction)
            if prediction is not None:
                predictions.append(prediction)
                targets.append(labels)

        # report metrics
        return self.calc_metrics(np.array(predictions), np.array(targets))

    def check_prediction(self, data_point: DataPoint, prediction: np.ndarray):
        if not data_point.need_prediction:
            if prediction is not None:
                raise ValueError(f"Prediction is not needed for {data_point}")
            return

        if prediction is None:
            raise ValueError(f"Prediction is required for {data_point}")

        if prediction.shape[0] != self.dim:
            raise ValueError(
                f"Prediction has wrong shape: {prediction.shape[0]} != {self.dim}"
            )

    def calc_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        scores = {}
        for ix_target, target_name in enumerate(self.targets):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix_target], predictions[:, ix_target]
            )
        scores["weighted_pearson"] = np.mean(list(scores.values()))
        return scores


# Test the normalize_data function
if __name__ == "__main__":
    print("Testing utils.py functions...")
    
    # Test normalize_data
    data = np.random.randn(10, 1000)
    normalized = normalize_data(data)
    print(f"Normalization function works: {normalized is not None}")
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {normalized.shape}")
    print(f"Normalized mean: {np.mean(normalized):.6f}")
    print(f"Normalized std: {np.std(normalized):.6f}")
    
    # Test other normalization methods
    normalized_minmax = normalize_data(data, method='minmax')
    print(f"\nMin-max normalized range: [{np.min(normalized_minmax):.6f}, {np.max(normalized_minmax):.6f}]")
    
    # Test create_sequences
    time_series = np.random.randn(100, 5)
    X, y = create_sequences(time_series, sequence_length=10, target_length=2)
    print(f"\nCreated sequences: X shape={X.shape}, y shape={y.shape}")
    
    # Test calculate_statistics
    stats = calculate_statistics(data)
    print(f"\nData statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")