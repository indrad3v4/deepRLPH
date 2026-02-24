import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


class WeightedPearsonEvaluator:
    """
    Evaluation metric for calculating exact weighted Pearson correlation
    on (batch, 901, 2) shaped tensors.
    
    This evaluator computes the weighted Pearson correlation coefficient
    locally for each batch and reports metrics for validation datasets.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize the evaluator.
        
        Args:
            eps: Small value to avoid division by zero
        """
        self.eps = eps
        self.reset()
    
    def reset(self):
        """Reset internal state for new evaluation."""
        self.predictions = []
        self.targets = []
        self.batch_metrics = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update internal state with new batch of predictions and targets.
        
        Args:
            predictions: Predicted values of shape (batch, 901, 2)
            targets: Ground truth values of shape (batch, 901, 2)
        """
        # Validate input shapes
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} doesn't match "
                f"targets shape {targets.shape}"
            )
        
        if len(predictions.shape) != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, seq, targets), got {len(predictions.shape)}D"
            )
        
        if predictions.shape[1] != 901 or predictions.shape[2] != 2:
            raise ValueError(
                f"Expected shape (batch, 901, 2), got {predictions.shape}"
            )
        
        # Store predictions and targets
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        # Compute batch metrics
        batch_metric = self._compute_weighted_pearson_batch(predictions, targets)
        self.batch_metrics.append(batch_metric)
    
    def _compute_weighted_pearson_batch(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute weighted Pearson correlation for a single batch.
        
        Args:
            predictions: Predicted values of shape (batch, 901, 2)
            targets: Ground truth values of shape (batch, 901, 2)
            
        Returns:
            Dictionary with metrics for each target dimension and overall
        """
        batch_size = predictions.shape[0]
        seq_len = predictions.shape[1]
        n_targets = predictions.shape[2]
        
        # Flatten batch and sequence dimensions
        # Shape: (batch * seq_len, n_targets)
        # Use reshape instead of view to handle non-contiguous tensors
        pred_flat = predictions.reshape(-1, n_targets)
        target_flat = targets.reshape(-1, n_targets)
        
        # Compute weighted Pearson for each target dimension
        metrics = {}
        
        for target_idx in range(n_targets):
            pred_slice = pred_flat[:, target_idx]
            target_slice = target_flat[:, target_idx]
            
            # Get absolute values of targets for weighting
            weights = torch.abs(target_slice.detach())
            weights = weights + self.eps
            
            # Compute weighted means
            weighted_mean_pred = torch.sum(weights * pred_slice) / torch.sum(weights)
            weighted_mean_target = torch.sum(weights * target_slice) / torch.sum(weights)
            
            # Compute weighted deviations
            pred_centered = pred_slice - weighted_mean_pred
            target_centered = target_slice - weighted_mean_target
            
            # Apply weights to centered values
            weighted_pred_centered = weights * pred_centered
            weighted_target_centered = weights * target_centered
            
            # Compute weighted covariance and variances
            weighted_cov = torch.sum(weighted_pred_centered * weighted_target_centered)
            weighted_var_pred = torch.sum(weighted_pred_centered * pred_centered)
            weighted_var_target = torch.sum(weighted_target_centered * target_centered)
            
            # Compute weighted Pearson correlation coefficient
            denominator = torch.sqrt(weighted_var_pred * weighted_var_target + self.eps)
            correlation = weighted_cov / denominator
            
            # Clip correlation to [-1, 1] for numerical stability
            correlation = torch.clamp(correlation, -1.0, 1.0)
            
            # Store correlation for this target dimension
            metrics[f'target_{target_idx}_correlation'] = correlation.item()
            metrics[f'target_{target_idx}_loss'] = 1.0 - correlation.item()
        
        # Compute overall metrics (average across targets)
        correlations = [metrics[f'target_{i}_correlation'] for i in range(n_targets)]
        losses = [metrics[f'target_{i}_loss'] for i in range(n_targets)]
        
        metrics['mean_correlation'] = np.mean(correlations)
        metrics['mean_loss'] = np.mean(losses)
        metrics['batch_size'] = batch_size
        
        return metrics
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics across all accumulated batches.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.batch_metrics:
            raise ValueError("No data has been accumulated. Call update() first.")
        
        # Aggregate metrics across all batches
        aggregated = {
            'total_batches': len(self.batch_metrics),
            'total_samples': sum(bm['batch_size'] for bm in self.batch_metrics)
        }
        
        # Get all target dimension names
        n_targets = 2  # Fixed based on shape (batch, 901, 2)
        target_keys = [f'target_{i}_correlation' for i in range(n_targets)]
        
        # Aggregate per-target metrics
        for target_idx in range(n_targets):
            corr_key = f'target_{target_idx}_correlation'
            loss_key = f'target_{target_idx}_loss'
            
            # Weight by batch size for aggregation
            total_weight = 0
            weighted_corr_sum = 0
            weighted_loss_sum = 0
            
            for batch_metric in self.batch_metrics:
                batch_size = batch_metric['batch_size']
                weighted_corr_sum += batch_metric[corr_key] * batch_size
                weighted_loss_sum += batch_metric[loss_key] * batch_size
                total_weight += batch_size
            
            aggregated[corr_key] = weighted_corr_sum / total_weight
            aggregated[loss_key] = weighted_loss_sum / total_weight
        
        # Aggregate overall metrics
        mean_corr_key = 'mean_correlation'
        mean_loss_key = 'mean_loss'
        
        total_weight = 0
        weighted_mean_corr_sum = 0
        weighted_mean_loss_sum = 0
        
        for batch_metric in self.batch_metrics:
            batch_size = batch_metric['batch_size']
            weighted_mean_corr_sum += batch_metric[mean_corr_key] * batch_size
            weighted_mean_loss_sum += batch_metric[mean_loss_key] * batch_size
            total_weight += batch_size
        
        aggregated[mean_corr_key] = weighted_mean_corr_sum / total_weight
        aggregated[mean_loss_key] = weighted_mean_loss_sum / total_weight
        
        # Add standard deviations for uncertainty estimation
        for target_idx in range(n_targets):
            corr_key = f'target_{target_idx}_correlation'
            corr_values = [bm[corr_key] for bm in self.batch_metrics]
            aggregated[f'{corr_key}_std'] = np.std(corr_values)
        
        mean_corr_values = [bm[mean_corr_key] for bm in self.batch_metrics]
        aggregated[f'{mean_corr_key}_std'] = np.std(mean_corr_values)
        
        return aggregated
    
    def evaluate_validation_dataset(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        """
        Evaluate model on validation dataset.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader for validation dataset
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        self.reset()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Assuming batch_data is (inputs, targets) tuple
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    inputs, targets = batch_data, None
                
                # Move to device
                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)
                
                # Get predictions
                predictions = model(inputs)
                
                # If model has predict method, use it
                if hasattr(model, 'predict'):
                    predictions = model.predict(inputs)
                
                # Ensure predictions are float32
                predictions = predictions.float()
                
                # Update metrics
                if targets is not None:
                    self.update(predictions, targets)
                else:
                    warnings.warn(f"No targets provided in batch {batch_idx}")
        
        # Compute final metrics
        metrics = self.compute()
        
        # Add additional information
        metrics['device'] = str(device)
        metrics['model'] = model.__class__.__name__
        
        return metrics
    
    def get_detailed_report(self) -> str:
        """
        Generate a detailed text report of the evaluation metrics.
        
        Returns:
            Formatted report string
        """
        if not self.batch_metrics:
            return "No evaluation data available. Call compute() first."
        
        metrics = self.compute()
        
        report_lines = [
            "=" * 60,
            "WEIGHTED PEARSON EVALUATION REPORT",
            "=" * 60,
            f"Total batches evaluated: {metrics['total_batches']}",
            f"Total samples evaluated: {metrics['total_samples']}",
            "",
            "PER-TARGET METRICS:",
            "-" * 40,
        ]
        
        # Add per-target metrics
        for target_idx in range(2):  # Fixed to 2 targets
            corr = metrics[f'target_{target_idx}_correlation']
            loss = metrics[f'target_{target_idx}_loss']
            corr_std = metrics.get(f'target_{target_idx}_correlation_std', 0.0)
            
            report_lines.append(
                f"Target {target_idx}: "
                f"Correlation = {corr:.6f} ± {corr_std:.6f}, "
                f"Loss = {loss:.6f}"
            )
        
        report_lines.extend([
            "",
            "OVERALL METRICS:",
            "-" * 40,
            f"Mean Correlation: {metrics['mean_correlation']:.6f} ± {metrics.get('mean_correlation_std', 0.0):.6f}",
            f"Mean Loss: {metrics['mean_loss']:.6f}",
            "",
            "INTERPRETATION:",
            "-" * 40,
            "Correlation ranges from -1 to 1:",
            "  1.0: Perfect positive correlation",
            "  0.0: No correlation",
            " -1.0: Perfect negative correlation",
            "Loss = 1 - Correlation (lower is better)",
            "=" * 60,
        ])
        
        return "\n".join(report_lines)


# Convenience function for quick evaluation
def evaluate_weighted_pearson(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Convenience function to compute weighted Pearson metrics for a single batch.
    
    Args:
        predictions: Predicted values of shape (batch, 901, 2)
        targets: Ground truth values of shape (batch, 901, 2)
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = WeightedPearsonEvaluator(eps=eps)
    evaluator.update(predictions, targets)
    return evaluator.compute()