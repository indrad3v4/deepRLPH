import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import pytest

# Import modules to test
from src.train import TrainingPipeline, create_data_loaders
from src.models.deep_lob import create_model
from src.data.loader import OFIDataLoader


class TestTrainingPipeline:
    """Test cases for the training pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.test_dir) / 'checkpoints'
        self.log_dir = Path(self.test_dir) / 'logs'
        
        # Create a simple model for testing
        self.model = create_model()
        
        # Create synthetic data loaders for testing
        batch_size = 4
        sequence_length = 1000
        
        # Create synthetic features and targets
        train_features = torch.randn(batch_size * 2, sequence_length, 32)
        train_targets = torch.randn(batch_size * 2, 901, 2)
        
        val_features = torch.randn(batch_size, sequence_length, 32)
        val_targets = torch.randn(batch_size, 901, 2)
        
        # Create simple dataset class
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    'targets': self.targets[idx]
                }
        
        # Create datasets
        train_dataset = SyntheticDataset(train_features, train_targets)
        val_dataset = SyntheticDataset(val_features, val_targets)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_pipeline_initialization(self):
        """Test that training pipeline initializes correctly."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        assert pipeline.model is not None
        assert pipeline.device == 'cpu'
        assert pipeline.max_epochs == 2
        assert pipeline.checkpoint_dir.exists()
        assert pipeline.log_dir.exists()
    
    def test_train_epoch(self):
        """Test that training epoch runs without errors."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Run one training epoch
        train_loss = pipeline.train_epoch()
        
        assert isinstance(train_loss, float)
        assert train_loss >= 0.0
    
    def test_validation(self):
        """Test that validation runs without errors."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Run validation
        val_loss = pipeline.validate()
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0
    
    def test_dimension_matching(self):
        """Test that predictions and targets dimensions are properly matched."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Get a batch of data
        batch_data = next(iter(self.train_loader))
        features = batch_data['features'].to('cpu')
        targets = batch_data['targets'].to('cpu')
        
        # Forward pass
        pipeline.model.eval()
        with torch.no_grad():
            predictions = pipeline.model(features)
        
        # Check dimensions
        assert predictions.shape == torch.Size([4, 901, 2]), \
            f"Expected shape (4, 901, 2), got {predictions.shape}"
        assert targets.shape == torch.Size([4, 901, 2]), \
            f"Expected shape (4, 901, 2), got {targets.shape}"
        
        # Test dimension matching in loss computation
        loss = pipeline.criterion(predictions, targets)
        assert isinstance(loss, torch.Tensor)
    
    def test_checkpoint_saving(self):
        """Test that model checkpoints are saved correctly."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Save a checkpoint
        pipeline.save_checkpoint(epoch=1, val_loss=0.5, is_best=True)
        
        # Check that files were created
        checkpoint_path = self.checkpoint_dir / 'checkpoint_epoch_001.pt'
        best_path = self.checkpoint_dir / 'best_model.pt'
        
        assert checkpoint_path.exists(), "Regular checkpoint not saved"
        assert best_path.exists(), "Best model checkpoint not saved"
        
        # Load and verify checkpoint
        checkpoint = torch.load(best_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert checkpoint['epoch'] == 1
        assert checkpoint['val_loss'] == 0.5
    
    def test_history_saving(self):
        """Test that training history is saved correctly."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Add some dummy history
        pipeline.history['train_loss'] = [0.8, 0.7]
        pipeline.history['val_loss'] = [0.6, 0.5]
        
        # Save history
        pipeline.save_history()
        
        # Check that file was created
        history_path = self.log_dir / 'training_history.json'
        assert history_path.exists(), "Training history not saved"
    
    def test_short_training(self):
        """Test a short training run."""
        pipeline = TrainingPipeline(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            max_epochs=2
        )
        
        # Run short training
        pipeline.train()
        
        # Check that history was updated
        assert len(pipeline.history['train_loss']) > 0
        assert len(pipeline.history['val_loss']) > 0
        assert 'best_val_loss' in pipeline.history
        assert 'best_epoch' in pipeline.history


def test_create_data_loaders():
    """Test that data loaders are created correctly."""
    # Test with test_mode=True to use synthetic data
    train_loader, val_loader = create_data_loaders(
        batch_size=4,
        sequence_length=1000,
        max_sequences=10,
        test_mode=True
    )
    
    # Check that loaders are created
    assert train_loader is not None
    assert val_loader is not None
    
    # Get a batch from train loader
    batch_data = next(iter(train_loader))
    
    # Check batch structure
    assert 'features' in batch_data
    assert 'targets' in batch_data
    
    # Check dimensions
    features = batch_data['features']
    targets = batch_data['targets']
    
    assert features.shape[1] == 1000, f"Expected sequence length 1000, got {features.shape[1]}"
    assert features.shape[2] == 32, f"Expected 32 features, got {features.shape[2]}"
    assert targets.shape[1] == 901, f"Expected 901 time steps, got {targets.shape[1]}"
    assert targets.shape[2] == 2, f"Expected 2 targets, got {targets.shape[2]}"


def test_model_compatibility():
    """Test that model output matches expected dimensions."""
    model = create_model()
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1000, 32)
    
    # Forward pass
    output = model(dummy_input)
    
    # Check output dimensions
    assert output.shape == torch.Size([batch_size, 901, 2]), \
        f"Expected shape ({batch_size}, 901, 2), got {output.shape}"
    
    # Check that output is float32
    assert output.dtype == torch.float32, \
        f"Expected float32, got {output.dtype}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])