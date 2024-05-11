import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class LSTM(nn.Module):
    def __init__(self, device, num_tickers, num_sectors, num_industries, num_features):
        super().__init__()
        self.device = device
        ticker_embedding_dim = 30
        sector_embedding_dim = 10
        industry_embedding_dim = 15

        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=ticker_embedding_dim).to(device)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embedding_dim).to(device)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embedding_dim).to(device)

        # Input size should consider the concatenated size of features and embedded dimensions
        concat_input_size = ticker_embedding_dim + sector_embedding_dim + industry_embedding_dim + num_features

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=concat_input_size, hidden_size=256, num_layers=2, batch_first=True).to(device)

        # Linear layers to process the output of the LSTM
        self.fc1 = nn.Linear(256, 128).to(device)
        self.fc2 = nn.Linear(128, 64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)

        # Activation and BatchNorm layers
        self.relu = nn.ReLU().to(device)
        self.batch_norm1 = nn.BatchNorm1d(128).to(device)
        self.batch_norm2 = nn.BatchNorm1d(64).to(device)

    def forward(self, X_features, X_tickers, X_sectors, X_industries):
        # Embed and concatenate features
        X_tickers = self.ticker_embedding(X_tickers)
        X_sectors = self.sector_embedding(X_sectors)
        X_industries = self.industry_embedding(X_industries)
        X = torch.cat([X_features, X_tickers, X_sectors, X_industries], dim=1)  

        # LSTM output
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last sequence step

        # Pass through linear layers
        X = self.fc1(lstm_out)
        X = self.relu(self.batch_norm1(X))
        X = self.fc2(X)
        X = self.relu(self.batch_norm2(X))
        X = self.fc3(X)

        return X
