import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler

class DNN(nn.Module):
    def __init__(self, device, num_tickers, num_sectors, num_industries, num_features):
        super().__init__()
        self.device = device
        ticker_embedding_dim = 30
        sector_embedding_dim = 10
        industry_embedding_dim = 15  

        self.hidden_layers_size = [512, 384, 288, 216, 162, 128, 64, 32, 1]
        self.ticker_embedding = nn.Embedding(num_embeddings=num_tickers, embedding_dim=ticker_embedding_dim).to(self.device)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embedding_dim).to(self.device)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embedding_dim).to(self.device)
        concat_input_size = ticker_embedding_dim + num_features + industry_embedding_dim + sector_embedding_dim 

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(concat_input_size, self.hidden_layers_size[0])).to(self.device)
        self.layers.append(nn.ReLU()).to(self.device)

        for layer in range(1, len(self.hidden_layers_size)):
            self.layers.append(nn.Linear(self.hidden_layers_size[layer-1], self.hidden_layers_size[layer])).to(self.device)
            self.layers.append(nn.ReLU()).to(self.device)
            self.layers.append(nn.Dropout(p=0.3)).to(self.device)
        
        self.layers.append(nn.Linear(self.hidden_layers_size[-1], 1)).to(self.device)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)



    def forward(self, X_features: torch.Tensor, X_tickers: torch.Tensor, X_sectors: torch.Tensor, X_industries: torch.Tensor) -> torch.Tensor:
        embedded_tickers = self.ticker_embedding(X_tickers)
        embedded_sectors = self.sector_embedding(X_sectors)
        embedded_industries = self.industry_embedding(X_industries)

        combined_input = torch.cat((X_features, embedded_tickers, embedded_sectors, embedded_industries), dim=1)
        a_i = combined_input
        for layer in self.layers:
            a_i = layer(a_i)
        output = a_i
        return output


def train(device: torch.device, model: nn.Module, dataloader: DataLoader, 
          loss_func: nn.MSELoss, optimizer: torch.optim, 
          lr_scheduler: optim.lr_scheduler, num_epochs: int) -> list[float]:
    
    model.train().to(device)
    best_loss = float('inf')
    patience = 20
    trigger_times = 0
    epoch_average_losses = []
    display_interval = 10000
    
    for train_epoch in range(num_epochs):
        running_epoch_loss = 0.0
        total_samples_processed = 0
        
        for i, (X_numeric, X_ticker_indices, X_sector_indices, X_industry_indices, Y_b) in enumerate(dataloader):
            X_numeric, X_ticker_indices, Y_b = X_numeric.to(device), X_ticker_indices.to(device), Y_b.to(device)
            X_sector_indices, X_industry_indices = X_sector_indices.to(device), X_industry_indices.to(device)
            optimizer.zero_grad()
            batch_prediction = model(X_numeric, X_ticker_indices, X_sector_indices, X_industry_indices)
            batch_loss = loss_func(batch_prediction, Y_b)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()


            batch_loss_value = batch_loss.item()
            batch_samples = X_numeric.size(0)
            running_epoch_loss += batch_loss_value * batch_samples
            total_samples_processed += batch_samples

            if (i + 1) % display_interval == 0:
                print(f"Epoch {train_epoch + 1}, Batch {i + 1}: Loss = {batch_loss_value:.2e}")

        
        epoch_loss = running_epoch_loss / total_samples_processed
        epoch_average_losses.append(epoch_loss)

        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch_loss)
        
        print(f"Epoch: {train_epoch + 1} | Loss: {epoch_loss:.2e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    return epoch_average_losses



def test_model(device: torch.device, model: nn.Module, dataloader: DataLoader, 
                loss_func: nn.MSELoss) -> float:
    
    model.eval()to(device)
    total_test_loss = 0.0
    total_samples_processed = 0
    with torch.no_grad():
        for i, (X_numeric, X_ticker_indices, X_sector_indices, X_industry_indices, Y_b) in enumerate(dataloader):
            X_numeric, X_ticker_indices, Y_b = X_numeric.to(device), X_ticker_indices.to(device), Y_b.to(device)
            X_sector_indices, X_industry_indices = X_sector_indices.to(device), X_industry_indices.to(device)
            batch_prediction = model(X_numeric, X_ticker_indices, X_sector_indices, X_industry_indices)
            batch_loss = loss_func(batch_prediction, Y_b)
            batch_size = X_numeric.size(0)
            total_samples_processed += batch_size
            total_test_loss += batch_loss.item() * batch_size
    avg_test_loss = total_test_loss total_samples_processed
    print("Test Loss:", avg_test_loss)
    return avg_test_loss  

