import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim.lr_scheduler


class CNN(nn.Module):
    
    def __init__(self, unique_tickers):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=unique_tickers, embedding_dim=20)

        self.conv1 = nn.Conv1d(in_channels=21, out_channels=16, kernel_size=3, stride = 1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride = 1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels= 64, kernel_size=3, stride = 1)
        self.flatten = nn.Flatten()

        self.conv_output_size = self.get_conv_output_shape(torch.zeros(1, 21, 34))

        self.fully_cnnctd_1 = nn.Linear(self.conv_output_size, 512)
        self.fully_cnnctd_2 = nn.Linear(512, 256)
        self.fully_cnnctd_3 = nn.Linear(256, 128)
        self.fully_cnnctd_4 = nn.Linear(128, 64)
        self.fully_cnnctd_5 = nn.Linear(64, 1)


    def get_conv_output_shape(self, x):
        shape = None
        with torch.no_grad(): 
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            shape = x.shape[1]
        return shape

    def forward(self, X: torch.Tensor, X_tickers: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3 and X.shape[1] == 1:  # Ensure X has shape [batch, 1, seq_length]
            X = X.expand(-1, 20, -1)  # Expanding X to match the embedding dimension
        X = torch.cat((embedded, X), dim=1)  # Correct dimension for concatenation        o1 = torch.relu(self.conv1(X))
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
    


def train(model: nn.Module, dataloader: DataLoader, 
          loss_func: nn.MSELoss, optimizer: torch.optim, 
          lr_scheduler: optim.lr_scheduler, num_epochs: int) -> list[float]:
    
    model.train()
    epoch_average_losses = []
    
    for train_epoch in range(num_epochs):
        running_epoch_loss = 0.0
        total_samples_processed = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {train_epoch + 1}/{num_epochs}")
        
        for i, (X_numeric, X_ticker_indices, Y_b) in progress_bar:
            optimizer.zero_grad()
            batch_prediction = model(X_numeric, X_ticker_indices)
            batch_loss = loss_func(batch_prediction, Y_b)
            batch_loss.backward()
            optimizer.step()
            running_epoch_loss += batch_loss.item() * X_numeric.size(0)
            total_samples_processed += X_numeric.size(0)
            progress_bar.set_postfix(loss=(running_epoch_loss / total_samples_processed))

        epoch_loss = running_epoch_loss / len(dataloader.dataset)
        epoch_average_losses.append(epoch_loss)
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch_loss)
        
        print(f"Epoch: {train_epoch + 1} | Loss: {epoch_loss:.4f}")

    return epoch_average_losses



def test_model(model: nn.Module, dataloader: DataLoader, 
                loss_func: nn.MSELoss) -> float:
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (input_vectors, scalar_label) in dataloader:
            test_predictions = model.forward(input_vectors)
            loss_on_test_set = loss_func(test_predictions, scalar_label)
            test_loss += (loss_on_test_set.item() * input_vectors.shape[0]) / float(len(dataloader.dataset))
    
    print("Test Loss:", test_loss)
    return test_loss  

