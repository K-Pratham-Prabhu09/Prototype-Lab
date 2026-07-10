# This contains classes that inherit from torch.utils.data.Dataset. 
# It handles reading your CSVs or images from the data/ folder and converting them into PyTorch Tensors.
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path


class HARDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x_data = pd.concat([pd.read_csv(file, sep=r'\s+', header=None) for file in Path(x_path).glob('*.txt')], axis=1)
        arrays = [pd.read_csv(file, sep=r'\s+', header=None).to_numpy() for file in Path(x_path).glob('*.txt')]
        self.x_data = np.stack(arrays, axis=2)
        self.y_data = pd.read_csv(y_path, sep=r'\s+', header=None)

    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.x_data)

    def __getitem__(self, idx: int):
        row_features = self.x_data[idx]
        row_label = self.y_data.iloc[idx, 0] - 1
        feature_tensor = torch.tensor(row_features, dtype=torch.float32)
        label_tensor = torch.tensor(row_label, dtype=torch.long)
        return feature_tensor, label_tensor


if __name__ == "__main__":
    current_file_dir = Path(__file__).resolve().parent
    x_path = current_file_dir / ".." / "data" / "train" / "Inertial Signals"
    y_path = current_file_dir / ".." / "data" / "train" / "y_train.txt"
    dataset = HARDataset(x_path, y_path)
    print(f"Dataset Shape: {dataset[:][0].shape}")
