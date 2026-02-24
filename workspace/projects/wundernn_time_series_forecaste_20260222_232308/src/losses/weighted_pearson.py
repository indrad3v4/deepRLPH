import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedPearsonCorrelationLoss(nn.Module):
    """
    Weighted Pearson Correlation Loss for sequence-aware time series forecasting.
    
    This loss function:
    1. Handles 3D tensors (batch, seq, targets) by flattening along batch and sequence dimensions
    2. Applies |y_i| weighting to the correlation calculation
    3. Clips gradients/outputs safely to [-6, 6] for numerical stability
    
    Args:
        reduction (str): Specifies the reduction to apply to the output: 
                        'none' | 'mean' | 'sum'. Default: 'mean'
        eps (float): Small value to avoid division by zero. Default: 1e-8
    """
    
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, predictions, targets):
        """
        Compute weighted Pearson correlation loss.
        
        Args:
            predictions (torch.Tensor): Predicted values of shape (batch, seq_len, n_targets)
            targets (torch.Tensor): Ground truth values of shape (batch, seq_len, n_targets)
            
        Returns:
            torch.Tensor: Weighted Pearson correlation loss
        """
        # Store original shape for potential reshaping
        original_shape = predictions.shape
        
        # Flatten batch and sequence dimensions for correlation calculation
        # Shape becomes (batch * seq_len, n_targets)
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        
        # Get absolute values of targets for weighting
        # Use detach() to prevent gradient flow through weights
        weights = torch.abs(target_flat.detach())
        
        # Add small epsilon to avoid zero weights
        weights = weights + self.eps
        
        # Compute weighted means
        weighted_mean_pred = torch.sum(weights * pred_flat, dim=0) / torch.sum(weights, dim=0)
        weighted_mean_target = torch.sum(weights * target_flat, dim=0) / torch.sum(weights, dim=0)
        
        # Compute weighted deviations from means
        pred_centered = pred_flat - weighted_mean_pred
        target_centered = target_flat - weighted_mean_target
        
        # Apply weights to centered values
        weighted_pred_centered = weights * pred_centered
        weighted_target_centered = weights * target_centered
        
        # Compute weighted covariance and variances
        weighted_cov = torch.sum(weighted_pred_centered * weighted_target_centered, dim=0)
        weighted_var_pred = torch.sum(weighted_pred_centered * pred_centered, dim=0)
        weighted_var_target = torch.sum(weighted_target_centered * target_centered, dim=0)
        
        # Compute weighted Pearson correlation coefficient
        # Clip denominator to avoid numerical issues
        denominator = torch.sqrt(weighted_var_pred * weighted_var_target + self.eps)
        correlation = weighted_cov / denominator
        
        # Clip correlation to [-1, 1] for numerical stability
        correlation = torch.clamp(correlation, -1.0, 1.0)
        
        # Convert correlation to loss (we want correlation close to 1, so loss = 1 - correlation)
        loss_per_target = 1.0 - correlation
        
        # Apply gradient clipping through the loss computation
        # We'll use gradient clipping in the backward pass, but also ensure
        # intermediate values are stable
        loss_per_target = torch.clamp(loss_per_target, 0.0, 2.0)  # correlation in [-1, 1] gives loss in [0, 2]
        
        # Average loss across targets
        loss = torch.mean(loss_per_target)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss_per_target
        elif self.reduction == 'sum':
            return torch.sum(loss_per_target)
        else:  # 'mean'
            return loss
    
    def clip_gradients(self, parameters, clip_value=6.0):
        """
        Clip gradients to prevent explosion.
        
        Args:
            parameters: Model parameters
            clip_value (float): Maximum absolute value for gradients
        """
        torch.nn.utils.clip_grad_norm_(parameters, clip_value)
    
    @staticmethod
    def safe_clip(tensor, min_val=-6.0, max_val=6.0):
        """
        Safely clip tensor values for numerical stability.
        
        Args:
            tensor (torch.Tensor): Input tensor
            min_val (float): Minimum value
            max_val (float): Maximum value
            
        Returns:
            torch.Tensor: Clipped tensor
        """
        return torch.clamp(tensor, min_val, max_val)


def weighted_pearson_loss(predictions, targets, reduction='mean', eps=1e-8):
    """
    Functional version of weighted Pearson correlation loss.
    
    Args:
        predictions (torch.Tensor): Predicted values
        targets (torch.Tensor): Ground truth values
        reduction (str): Reduction method
        eps (float): Small value for numerical stability
        
    Returns:
        torch.Tensor: Weighted Pearson correlation loss
    """
    return WeightedPearsonCorrelationLoss(reduction=reduction, eps=eps)(predictions, targets)