import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.model import MyNeuralNetwork
from torch.utils.data import DataLoader
from src.engine import train_one_epoch, get_train_test_data
import config
from pathlib import Path

if __name__ == "__main__":
    # CPU and GPU switching
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")

    mlflow.set_experiment(experiment_name=config.EXPERIMENT_NAME)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    run_name = "LSTM_basic_run"
    # Data preparation
    parent_folder_path = Path(__file__).resolve().parent
    train_dataset, test_dataset = get_train_test_data(parent_folder_path)

    # All Initializations:
    num_features = train_dataset[:][0].shape[2]
    lstm_model = MyNeuralNetwork(num_features,config.NUM_HIDDEN).to(device)
    optimizer = optim.Adam(lstm_model.parameters(),lr=config.LEARNING_RATE)
    criterion =nn.CrossEntropyLoss().to(device)
    
    criterion_name = type(criterion).__name__
    optimizer_name = type(optimizer).__name__

    train_data = DataLoader(train_dataset,batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8,pin_memory=True)
    test_data = DataLoader(test_dataset,batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True)
    # Record start time
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate":config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "weight_decay":config.WEIGHT_DECAY,
            "num_hidden":config.NUM_HIDDEN,
            "device": device,
            "criterion": criterion_name,
            "optimizer_name":optimizer_name,
            "num_features":num_features,
            "train_data_size": len(train_data),
            "test_data_size": len(test_data)
        })

        for epoch in range(config.EPOCHS):
            loss = train_one_epoch(train_data,lstm_model,criterion,optimizer,device)
            print(f"Epoch={epoch}||Loss={loss:0.10f}")
            mlflow.log_metric("loss",loss,step=epoch+1)

