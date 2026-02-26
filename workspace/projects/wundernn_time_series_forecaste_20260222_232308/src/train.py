import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
import warnings
import argparse

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Import project modules
from src.models.deep_lob import create_model as create_deep_lob
from src.models.t_kan import create_model as create_t_kan
from src.data.loader import OFIDataLoader
from src.losses.weighted_pearson import WeightedPearsonCorrelationLoss


class TrainingPipeline:
    """
    Resilient Training Pipeline for Time Series Forecasting Models.

    TRIZ Principle 9: Prior Counteraction.
    It catches infinite/NaN gradients BEFORE they poison the network,
    and accurately aligns temporal sequences to prevent chronological leakage.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cpu',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-2,
            max_epochs: int = 10,
            patience: int = 3,
            gradient_clip: float = 6.0,
            model_name: str = 'deep_lob'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.model_name = model_name

        self.checkpoint_dir = Path('models/checkpoints')
        self.final_dir = Path('models/final')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = WeightedPearsonCorrelationLoss(reduction='mean')

        # Optimizer & Scheduler (Strong L2 Regularization applied here)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1, verbose=True, min_lr=1e-6
        )

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_pearson': [],
            'best_val_loss': float('inf'), 'best_epoch': 0,
            'train_pearson': [], 'val_pearson_history': []
        }

        print(f"✅ Training pipeline initialized for {model_name} on {device.upper()}")
        print(f"✅ Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        for batch_idx, batch_data in enumerate(self.train_loader):
            features = batch_data['features'].to(self.device)
            targets = batch_data['targets'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)

            # ==========================================
            # 🚨 THE TRUE ALIGNMENT FIX (Separation in Space/Time)
            # ==========================================
            # We must align the ends of the sequences (-min_len:), not the beginnings!
            # This prevents the model from using Time 99 features to predict Time 0 targets.
            if predictions.shape[1] != targets.shape[1]:
                min_len = min(predictions.shape[1], targets.shape[1])
                predictions = predictions[:, -min_len:, :]
                targets = targets[:, -min_len:, :]

            loss = self.criterion(predictions, targets)

            # TRIZ Principle 9: The Poison Catcher
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [🛡️ SHIELD ACTIVATED] Batch {batch_idx + 1} yielded NaN/Inf loss. Skipping to protect weights.")
                continue

            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            if (batch_idx + 1) % 50 == 0:
                batch_pearson = 1.0 - loss.item()
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f} | Pearson approx: {batch_pearson:.4f}")

        return total_loss / max(valid_batches, 1)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                features = batch_data['features'].to(self.device)
                targets = batch_data['targets'].to(self.device)

                predictions = self.model(features)

                # ==========================================
                # 🚨 THE TRUE ALIGNMENT FIX (Validation)
                # ==========================================
                if predictions.shape[1] != targets.shape[1]:
                    min_len = min(predictions.shape[1], targets.shape[1])
                    predictions = predictions[:, -min_len:, :]
                    targets = targets[:, -min_len:, :]

                loss = self.criterion(predictions, targets)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1

        return total_loss / max(valid_batches, 1)

    def save_model(self, epoch: int, val_loss: float):
        """Saves the PyTorch state dict securely."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'model_name': self.model_name,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        # Save to checkpoints
        torch.save(checkpoint, self.checkpoint_dir / f'best_{self.model_name}.pt')
        # Save directly to final for easy export
        torch.save(checkpoint, self.final_dir / f'best_{self.model_name}.pt')

    def train(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting Training for {self.model_name.upper()}: Unleashing the Beast (Anti-Overfit Edition)")
        print("=" * 60)

        no_improvement_count = 0

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch}/{self.max_epochs} ---")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Unmasking the Pearson Score (Loss = 1 - Pearson)
            train_pearson = 1.0 - train_loss
            val_pearson = 1.0 - val_loss

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_pearson'].append(train_pearson)
            self.history['val_pearson_history'].append(val_pearson)

            self.scheduler.step(val_loss)

            is_best = val_loss < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                no_improvement_count = 0
                print(f"  🌟 NEW BEST! Val Pearson: {val_pearson:.4f}")
                self.save_model(epoch, val_loss)
            else:
                no_improvement_count += 1

            epoch_time = time.time() - epoch_start
            print(f"  Summary -> Train Pearson: {train_pearson:.4f} | Val Pearson: {val_pearson:.4f} | Time: {epoch_time:.1f}s")

            if no_improvement_count >= self.patience:
                print(f"\n🛑 Early stopping triggered. No improvement for {self.patience} epochs.")
                break

        best_pearson = 1.0 - self.history['best_val_loss']
        print("\n" + "=" * 60)
        print(f"🎯 Training Complete for {self.model_name.upper()}. 🏆 Best Validation Pearson: {best_pearson:.4f} (epoch {self.history['best_epoch']})")
        print("=" * 60)
        
        return best_pearson


def load_config(model_type: str, fast_dev_run: bool = False, test_mode: bool = False):
    """Load configuration based on model type."""
    if model_type == 't_kan':
        # T-KAN specific configuration
        config = {
            'batch_size': 32,
            'sequence_length': 1000,
            'learning_rate': 0.001,  # Higher LR for KAN
            'weight_decay': 0.01,
            'max_epochs': 50,
            'patience': 5,
            'gradient_clip': 5.0,
            'max_sequences': 20000,
            'test_mode': False
        }
    else:
        # Default DeepLOB config
        config = {
            'batch_size': 32,
            'sequence_length': 1000,
            'learning_rate': 0.0005,
            'weight_decay': 1e-2,
            'max_epochs': 10,
            'patience': 3,
            'gradient_clip': 6.0,
            'max_sequences': 20000,
            'test_mode': False
        }
    
    # Override for fast dev run
    if fast_dev_run:
        config['max_epochs'] = 1
        config['max_sequences'] = 100
        config['test_mode'] = True
    
    # Override for test mode
    if test_mode:
        config['test_mode'] = True
    
    return config


def create_model(model_type: str):
    """Create model based on type."""
    if model_type == 't_kan':
        print("🔧 Creating TemporalKAN model...")
        return create_t_kan()
    else:
        print("🔧 Creating deep_lob model...")
        return create_deep_lob()


def main():
    parser = argparse.ArgumentParser(description='Train time series forecasting models')
    parser.add_argument('--model', type=str, default='deep_lob', 
                       choices=['deep_lob', 't_kan'], help='Model type to train')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs')
    parser.add_argument('--epochs', type=int, default=None, help='Maximum number of epochs (alias for --max_epochs)')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a quick development test')
    parser.add_argument('--dry-run', action='store_true', help='Run a quick development test (alias for --fast_dev_run)')
    parser.add_argument('--test_mode', action='store_true', help='Use synthetic data for testing')
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.epochs is not None:
        args.max_epochs = args.epochs
    if args.dry_run:
        args.fast_dev_run = True
    
    # Load configuration
    config = load_config(args.model, args.fast_dev_run, args.test_mode)
    
    # Override config with command line arguments
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    
    print(f"\n📋 Configuration for {args.model.upper()}:")
    for key, value in config.items(): 
        print(f"  {key}: {value}")
    
    print(f"\nInitializing Data Loaders...")
    train_dataset = OFIDataLoader('data/raw/train.parquet', config['sequence_length'], config['batch_size'], True,
                                  'cpu', config['max_sequences'], config['test_mode'])
    val_dataset = OFIDataLoader('data/raw/valid.parquet', config['sequence_length'], config['batch_size'], False, 'cpu',
                                config['max_sequences'], config['test_mode'])
    
    # Create model
    model = create_model(args.model)
    
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_dataset.dataloader,
        val_loader=val_dataset.dataloader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        gradient_clip=config['gradient_clip'],
        model_name=args.model
    )
    
    best_pearson = pipeline.train()
    
    # Check if validation pearson meets acceptance criteria
    if best_pearson >= 0.25:
        print(f"\n✅ SUCCESS: Validation weighted_pearson metric reached {best_pearson:.4f} (>= 0.25)")
    else:
        print(f"\n⚠️ WARNING: Validation weighted_pearson metric is {best_pearson:.4f} (< 0.25)")
    
    return best_pearson


if __name__ == "__main__":
    main()