# This contains pure functions like train_one_epoch() and evaluate(). 
# They take your model and data as inputs and run the math.
import torch
from src.dataset import HARDataset

## Data preparation:

def get_train_test_data(parent_folder_path):
    # Train Dataset
    x_train_path = parent_folder_path / "data" / "train" / "Inertial Signals"
    y_train_path = parent_folder_path / "data" / "train" / "y_train.txt"
    train_dataset = HARDataset(x_train_path, y_train_path)
    print(f"Dataset Shape: {train_dataset[:][0].shape}")
    # Test Dataset
    x_test_path = parent_folder_path / "data" / "test" / "Inertial Signals"
    y_path = parent_folder_path / "data" / "test" / "y_test.txt"
    test_dataset = HARDataset(x_test_path, y_path)
    print(f"Dataset Shape: {test_dataset[:][0].shape}")

    return train_dataset, test_dataset

def train_one_epoch(train_loader,my_model,criterion,optimizer,device):
    running_loss = 0.0
    for X,Y in train_loader:
        # Move to the device specified
        X = X.to(device)
        Y = Y.to(device)

        # Forward Pass
        model_out = my_model(X)

        # Loss calculate
        loss = criterion(model_out,Y)

        # Grad zero
        optimizer.zero_grad()

        # Back Propagation
        loss.backward()

        # Update the Optimizer
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)

    # 4. Return the true average loss across all samples
    return running_loss / len(train_loader.dataset)
