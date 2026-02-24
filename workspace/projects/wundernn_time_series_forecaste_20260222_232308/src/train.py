import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import time
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import project modules
from src.models.deep_lob import TemporalFusionDeepLOB, create_model
from src.data.loader import OFIDataLoader
from src.losses.weighted_pearson import WeightedPearsonCorrelationLoss


class TrainingPipeline:
    """
    Optimized Training pipeline for DeepLOB model.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str = 'cpu',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            max_epochs: int = 5,
            patience: int = 2,
            checkpoint_dir: str = 'models/checkpoints',
            log_dir: str = 'logs',
            gradient_clip: float = 6.0,
            subsample_ratio: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.subsample_ratio = subsample_ratio

        self._validate_model_dimensions()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = WeightedPearsonCorrelationLoss(reduction='mean')

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1,
                                                              verbose=True, min_lr=1e-6)

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_pearson': [],
            'best_val_loss': float('inf'), 'best_epoch': 0
        }

        print(f"Training pipeline initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _validate_model_dimensions(self):
        print("Validating model dimensions...")
        sample_batch = next(iter(self.train_loader))
        features = sample_batch['features'].to(self.device)
        targets = sample_batch['targets'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)

        if predictions.shape[-1] != targets.shape[-1]:
            raise ValueError(f"Feature dimension mismatch: pred={predictions.shape[-1]}, targets={targets.shape[-1]}")

        self.model.train()
        print("Model dimension validation passed ✓")

    def _create_subsampled_loader(self, loader: DataLoader, ratio: float) -> DataLoader:
        if ratio >= 1.0:
            return loader
        dataset_size = len(loader.dataset)
        subsample_size = int(dataset_size * ratio)
        indices = torch.randperm(dataset_size)[:subsample_size]
        sampler = SubsetRandomSampler(indices)
        return DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler, num_workers=loader.num_workers)

    def train_epoch(self, subsample: bool = True) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        train_loader = self._create_subsampled_loader(self.train_loader, self.subsample_ratio) if (
                    subsample and self.subsample_ratio < 1.0) else self.train_loader

        for batch_idx, batch_data in enumerate(train_loader):
            features = batch_data['features'].to(self.device)
            targets = batch_data['targets'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)

            if predictions.shape != targets.shape:
                min_seq_len = min(predictions.shape[1], targets.shape[1])
                predictions = predictions[:, :min_seq_len, :]
                targets = targets[:, :min_seq_len, :]

            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 20 == 0:
                batch_pearson = 1.0 - loss.item()
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Pearson approx: {batch_pearson:.4f}")

        return total_loss / max(num_batches, 1)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                features = batch_data['features'].to(self.device)
                targets = batch_data['targets'].to(self.device)

                predictions = self.model(features)

                if predictions.shape != targets.shape:
                    min_seq_len = min(predictions.shape[1], targets.shape[1])
                    predictions = predictions[:, :min_seq_len, :]
                    targets = targets[:, :min_seq_len, :]

                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

            final_dir = Path('models/final')
            final_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, final_dir / 'best_model.pt')

    def train(self, use_subsample: bool = True):
        print("\n" + "=" * 60)
        print("🚀 Starting Training: Feeding the Beast")
        print("=" * 60)

        no_improvement_count = 0

        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()

            print(f"\n--- Epoch {epoch}/{self.max_epochs} ---")
            train_loss = self.train_epoch(subsample=use_subsample)
            val_loss = self.validate()

            # CALCULATE REAL PEARSON SCORE
            train_pearson = 1.0 - train_loss
            val_pearson = 1.0 - val_loss

            self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_pearson'].append(val_pearson)

            is_best = val_loss < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                no_improvement_count = 0
                print(f"  🌟 NEW BEST! Val Pearson: {val_pearson:.4f}")
            else:
                no_improvement_count += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            epoch_time = time.time() - epoch_start_time
            print(
                f"  Summary -> Train Pearson: {train_pearson:.4f} | Val Pearson: {val_pearson:.4f} | Time: {epoch_time:.1f}s")

            if no_improvement_count >= self.patience:
                print(f"\n🛑 Early stopping triggered. No improvement for {self.patience} epochs.")
                break

        print("\n" + "=" * 60)
        print("🎯 Training Complete")
        best_pearson = 1.0 - self.history['best_val_loss']
        print(f"🏆 Best Validation Pearson: {best_pearson:.4f} (epoch {self.history['best_epoch']})")
        print("=" * 60)


def main():
    # --- FEED THE BEAST CONFIGURATION ---
    config = {
        'batch_size': 16,
        'sequence_length': 1000,
        'learning_rate': 0.001,  # Smoother convergence for more data
        'weight_decay': 1e-4,
        'max_epochs': 5,  # Allow more time to learn
        'patience': 2,
        'gradient_clip': 6.0,
        'max_sequences': 3000,  # Increased from 500!
        'test_mode': False,
        'subsample_ratio': 1.0  # Use 100% of the loaded sequences
    }

    print("Configuration:")
    for key, value in config.items(): print(f"  {key}: {value}")

    # Create data loaders
    train_dataset = OFIDataLoader('data/raw/train.parquet', config['sequence_length'], config['batch_size'], True, True,
                                  'cpu', config['max_sequences'], config['test_mode'])
    val_dataset = OFIDataLoader('data/raw/valid.parquet', config['sequence_length'], config['batch_size'], False, True,
                                'cpu', config['max_sequences'], config['test_mode'])

    model = create_model()

    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_dataset.dataloader,
        val_loader=val_dataset.dataloader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        gradient_clip=config['gradient_clip'],
        subsample_ratio=config['subsample_ratio']
    )

    pipeline.train(use_subsample=True)


if __name__ == "__main__":
    main()