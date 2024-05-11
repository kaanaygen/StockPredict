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

        concat_input_size = ticker_embedding_dim + sector_embedding_dim + industry_embedding_dim + num_features

        self.lstm = nn.LSTM(input_size=concat_input_size, hidden_size=256, num_layers=2, batch_first=True).to(device)
        self.fc1 = nn.Linear(256, 128).to(device)
        self.fc2 = nn.Linear(128, 64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)
        self.relu = nn.ReLU().to(device)
        self.batch_norm1 = nn.BatchNorm1d(128).to(device)
        self.batch_norm2 = nn.BatchNorm1d(64).to(device)

    def forward(self, X_features, X_tickers, X_sectors, X_industries):
        X_tickers = self.ticker_embedding(X_tickers)
        X_sectors = self.sector_embedding(X_sectors)
        X_industries = self.industry_embedding(X_industries)

        # Ensure the features tensor is the same dimension as the embeddings
        X_features = X_features.unsqueeze(1)

        # Concatenate all inputs along the feature dimension (dimension=2)
        X = torch.cat([X_features, X_tickers.unsqueeze(1), X_sectors.unsqueeze(1), X_industries.unsqueeze(1)], dim=2)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(X)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last LSTM cell

        X = self.fc1(lstm_out)
        X = self.relu(self.batch_norm1(X))
        X = self.fc2(X)
        X = self.relu(self.batch_norm2(X))
        X = self.fc3(X)

        return X
