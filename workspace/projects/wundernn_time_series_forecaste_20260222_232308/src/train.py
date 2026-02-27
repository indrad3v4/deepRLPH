import sys
import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings('ignore')

from src.models.t_kan import create_model as create_t_kan
from src.data.loader import OFIDataLoader
from src.losses.weighted_pearson import WeightedPearsonCorrelationLoss


class TrainingPipeline:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=3e-4, patience=5, max_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_val_pearson = -float('inf')
        self.no_improvement_count = 0

        # TRIZ: Merging MSE for stability and Pearson for target metric
        self.pearson_criterion = WeightedPearsonCorrelationLoss(reduction='mean')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

    def train_epoch(self):
        self.model.train()
        total_pearson_loss = 0
        for batch in self.train_loader:
            features, targets = batch['features'].to(self.device), batch['targets'].to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(features)

            if preds.shape[1] != targets.shape[1]:
                targets = targets[:, -preds.shape[1]:, :]

            p_loss = self.pearson_criterion(preds, targets)
            mse_loss = F.mse_loss(preds, targets)
            loss = 0.5 * p_loss + 0.5 * mse_loss  # Hybrid Anchor

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_pearson_loss += p_loss.item()
        return 1.0 - (total_pearson_loss / len(self.train_loader))

    def validate(self):
        self.model.eval()
        total_p_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                f, t = batch['features'].to(self.device), batch['targets'].to(self.device)
                p = self.model(f)
                if p.shape[1] != t.shape[1]: t = t[:, -p.shape[1]:, :]
                total_p_loss += self.pearson_criterion(p, t).item()
        return 1.0 - (total_p_loss / len(self.val_loader))

    def run(self):
        print(f"🚀 Starting Engine | Native PyTorch | Params: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(1, self.max_epochs + 1):
            t_pearson = self.train_epoch()
            v_pearson = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch} | Train: {t_pearson:.4f} | Val: {v_pearson:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if v_pearson > self.best_val_pearson:
                self.best_val_pearson = v_pearson
                self.no_improvement_count = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"  🌟 NEW BEST! Saved weights to best_model.pt")
            else:
                self.no_improvement_count += 1
                print(f"  ⚠️ No improvement ({self.no_improvement_count}/{self.patience})")

            if self.no_improvement_count >= self.patience:
                print(f"\n🛑 EARLY STOPPING: No progress for {self.patience} epochs.")
                break

        self.deploy()

    def deploy(self):
        print("\n" + "=" * 50 + "\n📦 PACKAGING SUBMISSION (Native Mode)\n" + "=" * 50)
        # 1. Load the absolute best weights
        if os.path.exists("best_model.pt"):
            self.model.load_state_dict(torch.load("best_model.pt"))
            print("✅ Loaded best model weights.")

        # 2. Prerun Test (Native model)
        if os.path.exists("prerun.py"):
            print("🧪 Executing prerun.py sanity check...")
            res = subprocess.run(["python", "prerun.py"], capture_output=True, text=True)
            print("✅ Prerun Passed" if res.returncode == 0 else f"❌ Prerun Failed: {res.stderr}")

        # 3. Final Packaging
        if os.path.exists("scripts/make_submission.py"):
            print("🗜️ Creating submission.zip...")
            subprocess.run(["python", "scripts/make_submission.py", "."])
            print("✅ Done! submission.zip is ready for upload.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()

    train_file = 'data/raw/train.parquet'
    valid_file = 'data/raw/valid.parquet'

    # Using 20,000 sequences for robust training [cite: 83]
    train_ds = OFIDataLoader(train_file, 1000, 32, True, 'cpu', 20000, args.test_mode)
    val_ds = OFIDataLoader(valid_file, 1000, 32, False, 'cpu', 5000, args.test_mode)

    model = create_t_kan()
    pipeline = TrainingPipeline(model, train_ds.dataloader, val_ds.dataloader)
    pipeline.run()


if __name__ == "__main__":
    main()