import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.model import MyNeuralNetwork
from src.engine import train_one_epoch, get_train_test_data
import config

if __name__ == "__main__":

    train_dataset, test_dataset = get_train_test_data()
    #All Initializations:
    num_features = train_dataset.shape[1]
    my_model = MyNeuralNetwork(num_features,config.NUM_HIDDEN)
    
