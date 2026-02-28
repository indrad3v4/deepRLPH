import sys
import os
import subprocess
from pathlib import Path

# UNIVERSAL PATH INJECTOR
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import argparse

warnings.filterwarnings('ignore')

from src.models.deep_lob import create_model
from src.data.loader import OFIDataLoader
from src.losses.weighted_pearson import WeightedPearsonCorrelationLoss


class TrainingPipeline:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=3e-4, patience=5, max_epochs=20):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_val_pearson = -float('inf')
        self.no_improvement_count = 0

        self.pearson_criterion = WeightedPearsonCorrelationLoss(reduction='mean')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

    def train_epoch(self):
        self.model.train()
        total_pearson_loss = 0.0
        total_hybrid_loss = 0.0

        for batch in self.train_loader:
            features, targets = batch['features'].to(self.device), batch['targets'].to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(features)

            if preds.shape[1] != targets.shape[1]:
                targets = targets[:, -preds.shape[1]:, :]

            p_loss = self.pearson_criterion(preds, targets)
            mse_loss = F.mse_loss(preds, targets)
            hybrid_loss = 0.5 * p_loss + 0.5 * mse_loss

            hybrid_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_pearson_loss += p_loss.item()
            total_hybrid_loss += hybrid_loss.item()

        batches = len(self.train_loader)
        avg_hybrid = total_hybrid_loss / batches
        avg_pearson = 1.0 - (total_pearson_loss / batches)
        return avg_hybrid, avg_pearson

    def validate(self):
        self.model.eval()
        total_p_loss = 0.0
        total_hybrid_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                f, t = batch['features'].to(self.device), batch['targets'].to(self.device)
                p = self.model(f)
                if p.shape[1] != t.shape[1]:
                    t = t[:, -p.shape[1]:, :]

                p_loss = self.pearson_criterion(p, t)
                mse_loss = F.mse_loss(p, t)

                total_p_loss += p_loss.item()
                total_hybrid_loss += (0.5 * p_loss + 0.5 * mse_loss).item()

        batches = len(self.val_loader)
        avg_hybrid = total_hybrid_loss / batches
        avg_pearson = 1.0 - (total_p_loss / batches)
        return avg_hybrid, avg_pearson

    def run(self):
        print(f"🚀 Starting Training Pipeline | Params: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(1, self.max_epochs + 1):
            train_hybrid, train_pearson = self.train_epoch()
            val_hybrid, val_pearson = self.validate()
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.max_epochs} | Train Loss: {train_hybrid:.4f} | Train Pearson: {train_pearson:.4f} | Val Loss: {val_hybrid:.4f} | Val Pearson: {val_pearson:.4f} | LR: {lr:.6f}")

            if val_pearson > self.best_val_pearson:
                self.best_val_pearson = val_pearson
                self.no_improvement_count = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"  🌟 NEW BEST! Saved weights to best_model.pt")
            else:
                self.no_improvement_count += 1
                print(f"  ⚠️ No improvement ({self.no_improvement_count}/{self.patience})")

            if self.no_improvement_count >= self.patience:
                print(f"\n🛑 EARLY STOPPING: Plateau reached at {self.best_val_pearson:.4f}")
                break

        self.deploy()

    def deploy(self):
        print("\n" + "=" * 50 + "\n📦 PACKAGING SUBMISSION\n" + "=" * 50)

        if os.path.exists("best_model.pt"):
            self.model.load_state_dict(torch.load("best_model.pt", map_location=self.device))
            self.model.eval()
            print("✅ Loaded peak alpha weights.")

            print("⚡ Generating TorchScript JIT graph...")
            dummy = torch.randn(1, 1000, 32, device=self.device)
            with torch.no_grad():
                traced = torch.jit.trace(self.model, dummy)
                torch.jit.save(traced, "model_jit.pt")
            print("✅ Saved model_jit.pt")

        if os.path.exists("prerun.py"):
            print("\n🧪 Executing prerun.py sanity check...")
            res = subprocess.run(["python", "prerun.py"], capture_output=True, text=True)
            if res.returncode == 0:
                print("✅ Prerun Passed")
            else:
                print(f"❌ Prerun Failed:\n{res.stderr}")

        if os.path.exists("scripts/make_submission.py"):
            print("\n🗜️ Creating submission.zip...")
            subprocess.run(["python", "scripts/make_submission.py", "."])
            print("✅ Pipeline Complete.")


def create_data_loaders(train_file, valid_file, batch_size=32, test_mode=False):
    """Create data loaders for training and validation."""
    train_ds = OFIDataLoader(train_file, 1000, 32, True, 'cpu', 20000, test_mode)
    val_ds = OFIDataLoader(valid_file, 1000, 32, False, 'cpu', 5000, test_mode)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--fast_dev_run', action='store_true')
    args = parser.parse_args()

    train_file = 'data/raw/train.parquet'
    valid_file = 'data/raw/valid.parquet'

    # Create model
    model = create_model()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_file, valid_file, 
        batch_size=4 if args.test_mode or args.fast_dev_run else 32,
        test_mode=args.test_mode or args.fast_dev_run
    )
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        max_epochs=2 if args.test_mode or args.fast_dev_run else 20
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()