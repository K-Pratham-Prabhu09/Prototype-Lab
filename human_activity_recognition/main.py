import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.model import MyNeuralNetwork
from torch.utils.data import DataLoader
from src.engine import train_one_epoch, get_train_test_data
import config
from pathlib import Path
from datetime import datetime


if __name__ == "__main__":
    # Data preparation
    parent_folder_path = Path(__file__).resolve().parent
    train_dataset, test_dataset = get_train_test_data(parent_folder_path)
   
    # CPU and GPU switching
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # All Initializations:
    num_features = train_dataset[:][0].shape[2]
    lstm_model = MyNeuralNetwork(num_features,config.NUM_HIDDEN).to(device)
    optimizer = optim.Adam(lstm_model.parameters(),lr=config.LEARNING_RATE)
    criterion =nn.CrossEntropyLoss().to(device)

    train_data = DataLoader(train_dataset,batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8,pin_memory=True)
    test_data = DataLoader(test_dataset,batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True)
    # Record start time
    starttime = datetime.now()
    print(f"Training Started on:{starttime.strftime("%H:%M:%S")}")

    for epoch in range(config.EPOCHS):
        loss = train_one_epoch(train_dataset,lstm_model,criterion,optimizer,device)
        print(f"{datetime.now().strftime("%H:%M:%S")}:Epoch={epoch}||Loss={loss:0.10f}")
    # Record end time
    endtime = datetime.now()
    print(f"End Time:{endtime.strftime("%H:%M:%S")}")

    elapsed = endtime - starttime
    print(elapsed)
#Todo:
# Accuracy calculations MLFLOW init and logging
# CPU vs GPU timing comparison

