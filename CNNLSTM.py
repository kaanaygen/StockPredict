import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class CNNLSTM(nn.Module):
    def __init__(self, device, num_tickers, num_sectors, num_industries, num_features):
        super().__init__()
        self.device = device
        ticker_embedding_dim = 30
        sector_embedding_dim = 10
        industry_embedding_dim = 15

        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=ticker_embedding_dim).to(device)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embedding_dim).to(device)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embedding_dim).to(device)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1).to(device)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1).to(device)
        self.flatten = nn.Flatten().to(device)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=8, batch_first=True).to(device)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=4, batch_first=True).to(device)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True).to(device)

        
        self.fc1 = nn.Linear(512, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, 64).to(device)
        self.fc5 = nn.Linear(64, 1).to(device)

        self.relu = nn.ReLU().to(device)
        self.batch_norm1 = nn.BatchNorm1d(512).to(device)
        self.batch_norm2 = nn.BatchNorm1d(256).to(device)
        self.batch_norm3 = nn.BatchNorm1d(128).to(device)
        self.batch_norm4 = nn.BatchNorm1d(64).to(device)

    def forward(self, X: torch.Tensor, X_tickers: torch.Tensor, X_sectors: torch.Tensor, X_industries: torch.Tensor):
        embedded_tickers = self.ticker_embedding(X_tickers)
        embedded_sectors = self.sector_embedding(X_sectors)
        embedded_industries = self.industry_embedding(X_industries)
        embedded = torch.cat([embedded_tickers, embedded_sectors, embedded_industries], dim=1).unsqueeze(1)

        if X.dim() == 2:
            X = X.unsqueeze(1)  
        X = torch.cat((X, embedded), dim=2).to(self.device)
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = self.relu(self.conv3(X))
        X = self.flatten(X).unsqueeze(1)  

        lstm_out, _ = self.lstm1(X)
        lstm_out, _ = self.lstm2(X)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = lstm_out[:, -1, :] 

        X = self.fc1(lstm_out)
        X = self.relu(self.batch_norm1(X))
        X = self.fc2(X)
        X = self.relu(self.batch_norm2(X))
        X = self.fc3(X)
        X = self.relu(self.batch_norm3(X))
        X = self.fc4(X)
        X = self.relu(self.batch_norm4(X))
        X = self.fc5(X)

        return X
