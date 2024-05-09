import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class DNN(nn.Module):
    def __init__(self, num_tickers, num_features):
        super().__init__()
        ticker_embedding_dim = num_tickers
        concat_input_size = ticker_embedding_dim + num_features
        self.hidden_layers_size = [4056, 2048, 1024, 512, 256, 128, 64, 32, 16, 1]
        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=8)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(concat_input_size, self.hidden_layers_size[0]))
        self.layers.append(nn.ReLU())

        for layer in range(1, len(self.hidden_layers_size)):
            self.layers.append(nn.Linear(self.hidden_layers_size[layer-1], self.hidden_layers_size[layer]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.hidden_layers_size[-1], 1))


    def forward(self, X_features: torch.Tensor, X_tickers: torch.Tensor) -> torch.Tensor:
        embedded_tickers = self.ticker_embedding(X_tickers)

        combined_input = torch.cat((X_features, embedded_tickers), dim=1)
        a_i = combined_input
        for layer in self.layers:
            a_i = layer(a_i)
        output = a_i
        return output
