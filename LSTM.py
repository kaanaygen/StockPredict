import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class EnhancedLSTM(nn.Module):
    def __init__(self, device, num_tickers, num_sectors, num_industries, num_features):
        super().__init__()
        self.device = device
        ticker_embedding_dim = 30
        sector_embedding_dim = 10
        industry_embedding_dim = 15

        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=ticker_embedding_dim).to(device)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embedding_dim).to(device)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embedding_dim).to(device)

        concat_input_size = ticker_embedding_dim + sector_embedding_dim + industry_embedding_dim + num_features

        # Configuring the LSTM with multiple layers and bi-directionality
        self.lstm = nn.LSTM(input_size=concat_input_size, hidden_size=256, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True).to(device)
        
        # Fully connected layers, with the first layer taking the doubled output size from the bidirectional LSTM
        self.fc1 = nn.Linear(512, 512).to(device)  # Adjusted for bidirectional output
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, 1).to(device)
        self.relu = nn.ReLU().to(device)
        self.batch_norm1 = nn.BatchNorm1d(512).to(device)
        self.batch_norm2 = nn.BatchNorm1d(256).to(device)

    def forward(self, X: torch.Tensor, X_tickers: torch.Tensor, X_sectors: torch.Tensor, X_industries: torch.Tensor) -> torch.Tensor:
        embedded_tickers = self.ticker_embedding(X_tickers)
        embedded_sectors = self.sector_embedding(X_sectors)
        embedded_industries = self.industry_embedding(X_industries)
        embedded = torch.cat([embedded_tickers, embedded_sectors, embedded_industries], dim=1).unsqueeze(1)

        if X.dim() == 2:
            X = X.unsqueeze(1)  # Ensuring it has a channel dimension

        X = torch.cat((X, embedded), dim=2).to(self.device)
        
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]  # Taking the output from the last LSTM cell across both directions

        X = self.fc1(lstm_out)
        X = self.relu(self.batch_norm1(X))
        X = self.fc2(X)
        X = self.relu(self.batch_norm2(X))
        X = self.fc3(X)

        return X
