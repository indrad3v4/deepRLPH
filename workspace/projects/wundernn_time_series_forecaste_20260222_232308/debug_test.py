import sys, os; sys.path.insert(0, os.path.abspath('.'))
import torch
import numpy as np
import tempfile
import shutil
import pandas as pd
from src.train import TrainingPipeline, create_default_configs

# Create temporary directories for test data
test_dir = tempfile.mkdtemp()
train_dir = os.path.join(test_dir, "train")
valid_dir = os.path.join(test_dir, "valid")
checkpoint_dir = os.path.join(test_dir, "checkpoints")
log_dir = os.path.join(test_dir, "logs")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Create dummy parquet files for testing
n_train = 1000  # Smaller for testing
train_data = {}
for i in range(1, 9):
    train_data[f'bid_price_{i}'] = np.random.normal(100, 1, n_train)
    train_data[f'ask_price_{i}'] = np.random.normal(101, 1, n_train)
    train_data[f'bid_volume_{i}'] = np.random.randint(10, 100, n_train)
    train_data[f'ask_volume_{i}'] = np.random.randint(10, 100, n_train)

train_df = pd.DataFrame(train_data)
train_file = os.path.join(train_dir, "train.parquet")
train_df.to_parquet(train_file)

# Create validation data
n_valid = 500  # Smaller for testing
valid_data = {}
for i in range(1, 9):
    valid_data[f'bid_price_{i}'] = np.random.normal(100, 1, n_valid)
    valid_data[f'ask_price_{i}'] = np.random.normal(101, 1, n_valid)
    valid_data[f'bid_volume_{i}'] = np.random.randint(10, 100, n_valid)
    valid_data[f'ask_volume_{i}'] = np.random.randint(10, 100, n_valid)

valid_df = pd.DataFrame(valid_data)
valid_file = os.path.join(valid_dir, "valid.parquet")
valid_df.to_parquet(valid_file)

print(f"Created test data with {n_train} training rows and {n_valid} validation rows")

# Get default configs
model_config, training_config = create_default_configs()

# Modify training config for faster testing
training_config['epochs'] = 1  # Only 1 epoch for debugging
training_config['batch_size'] = 4
training_config['log_interval'] = 100  # Don't log during tests
training_config['num_workers'] = 0  # Avoid multiprocessing issues

print("Creating pipeline...")
pipeline = TrainingPipeline(
    train_data_dir=train_dir,
    valid_data_dir=valid_dir,
    model_config=model_config,
    training_config=training_config,
    checkpoint_dir=checkpoint_dir,
    log_dir=log_dir
)

print("Initializing data loaders...")
pipeline._initialize_data_loaders()

print(f"Training batches: {len(pipeline.train_loader)}")
print(f"Validation batches: {len(pipeline.valid_loader)}")

print("Training for one epoch...")
try:
    train_loss = pipeline.train_epoch()
    print(f"Training loss: {train_loss}")
    
    print("Validating...")
    valid_loss = pipeline.validate()
    print(f"Validation loss: {valid_loss}")
    
    print("Test completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

# Clean up
shutil.rmtree(test_dir)
print("Cleanup complete.")