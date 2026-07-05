# This contains classes that inherit from torch.utils.data.Dataset. 
# It handles reading your CSVs or images from the data/ folder and converting them into PyTorch Tensors.
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from pathlib import Path

class MyCustomDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Load your data from the CSV or images here
        self.x_data = pd.read_csv(x_path, sep=r'\s+', header=None)
        self.y_data = pd.read_csv(y_path, sep=r'\s+', header=None)

    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return a single sample (input, label) as PyTorch Tensors
        x_row = self.x_data.iloc[idx]
        y_row = self.y_data.iloc[idx]
        input_tensor = torch.tensor(x_row.values, dtype=torch.float32)
        label_tensor = torch.tensor(y_row.values[0], dtype=torch.long)
        return input_tensor, label_tensor


if __name__ == "__main__":
    current_file_dir = Path(__file__).resolve().parent
    x_path = current_file_dir / ".." / "data" / "train" / "X_train.txt"
    y_path = current_file_dir / ".." / "data" / "train" / "y_train.txt"
    dataset = MyCustomDataset(x_path=x_path, y_path=y_path)
    print(len(dataset))
    print(dataset[0])

