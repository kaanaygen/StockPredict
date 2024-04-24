import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers_size = [512, 256, 128, 64]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(667, self.hidden_layers_size[0]))
        self.layers.append(nn.ReLU())

        for layer in range(1, len(hidden_layers_size)):
            self.layers.append(nn.Linear(self.hidden_layers_size[i-1], self.hidden_layers_size[i]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.hidden_layers_size[-1], 1))


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        a_i = None
        for layer in self.layers:
            a_i = layer(x)
        output = a_i
        return output
