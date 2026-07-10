# This contains pure functions like train_one_epoch() and evaluate(). 
# They take your model and data as inputs and run the math.
import torch
from src.dataset import HARDataset
from pathlib import Path

## Data preparation:

def get_train_test_data():
    # Train Dataset
    current_file_dir = Path(__file__).resolve().parent
    x_train_path = current_file_dir / "data" / "train" / "Inertial Signals"
    y_train_path = current_file_dir / "data" / "train" / "y_train.txt"
    train_dataset = HARDataset(x_train_path, y_train_path)
    print(f"Dataset Shape: {train_dataset[:][0].shape}")
    # Test Dataset
    x_test_path = current_file_dir / "data" / "test" / "Inertial Signals"
    y_path = current_file_dir / "data" / "test" / "y_test.txt"
    test_dataset = HARDataset(x_test_path, y_path)
    print(f"Dataset Shape: {test_dataset[:][0].shape}")

    return train_dataset, test_dataset

def train_one_epoch(my_model,data,num_hidden):
    # Model assignment
    out = my_model(data)
    # Loss calculate

    # Grad zero

    # Back Propagation