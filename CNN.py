import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler


class CNN(nn.Module):
    
    def __init__(self, device, num_tickers, num_sectors, num_industries, num_features):

        super().__init__()
        self.device = device
        ticker_embedding_dim = 30
        sector_embedding_dim = 10
        industry_embedding_dim = 15
        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=ticker_embedding_dim).to(self.device)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embedding_dim).to(self.device)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embedding_dim).to(self.device)
        concat_input_size = ticker_embedding_dim + num_features + industry_embedding_dim + sector_embedding_dim 
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels=16, kernel_size=3, stride = 1).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride = 1).to(self.device)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels= 64, kernel_size=3, stride = 1).to(self.device)
        
        self.flatten = nn.Flatten().to(self.device)

        self.conv_output_size = self.get_conv_output_shape(torch.zeros(1, 1 ,concat_input_size, device=self.device))

        self.fully_cnnctd_1 = nn.Linear(self.conv_output_size, 512).to(self.device)
        self.fully_cnnctd_2 = nn.Linear(512, 256).to(self.device)
        self.fully_cnnctd_3 = nn.Linear(256, 128).to(self.device)
        self.fully_cnnctd_4 = nn.Linear(128, 64).to(self.device)
        self.fully_cnnctd_5 = nn.Linear(64, 1).to(self.device)


    def get_conv_output_shape(self, x):
        shape = None
        with torch.no_grad(): 
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            shape = x.shape[1]
        return shape

    def forward(self, X: torch.Tensor, X_tickers: torch.Tensor, X_sectors: torch.Tensor, X_industries: torch.Tensor) -> torch.Tensor:
        embedded_tickers = self.ticker_embedding(X_tickers)
        embedded_sectors = self.sector_embedding(X_sectors)
        embedded_industries = self.industry_embedding(X_industries)
        embedded = torch.cat([embedded_tickers, embedded_sectors, embedded_industries], dim=1).unsqueeze(1)
        
        if X.dim() == 2:
            X = X.unsqueeze(1)  # Adding channel dimension

        X = torch.cat((X, embedded), dim=2).to(self.device)
        o1 = torch.relu(self.conv1(X))
        o2 = torch.relu(self.conv2(o1))
        o3 = torch.relu(self.conv3(o2))
        o4 = self.flatten(o3)
        o5 = torch.relu(self.fully_cnnctd_1(o4))
        o6 = torch.relu(self.fully_cnnctd_2(o5))
        o7 = torch.relu(self.fully_cnnctd_3(o6))
        o8 = torch.relu(self.fully_cnnctd_4(o7))
        o9 = torch.relu(self.fully_cnnctd_5(o8))
        output = o9
        return output 
    