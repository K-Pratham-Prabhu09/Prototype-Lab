## This is where you write your Neural Networks. 
# They should always be classes that inherit from torch.nn.Module.
import torch
import torch.nn as nn
import torchinfo

class MyNeuralNetwork(nn.Module):
    def __init__(self,num_features,num_hidden):
        super().__init__()
        self.lstm = nn.LSTM(num_features,num_hidden,batch_first=True)
        self.fc = nn.Linear(num_hidden,10)
        
    def forward(self,x):
        out, _ = self.lstm(x)
        out = self.fc(out[-1,:])
        return out
    

if __name__ == "__main__":
    model = MyNeuralNetwork(num_features=9, num_hidden=32)
    torchinfo.summary(model, [7352, 128, 9])
