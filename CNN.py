import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    
    def __init__(self, data):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride = 1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride = 1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels= 128, kernel_size=3, stride = 1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1)
        self.flatten = nn.Flatten()
        self.fully_cnnctd_1 = nn.Linear(256 * (data.shape[1] + 4 * (-3 + 1)), 1024)
        self.fully_cnnctd_2 = nn.Linear(1024, 512)
        self.fully_cnnctd_3 = nn.Linear(512, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        o1 = torch.relu(self.conv1(X))
        o2 = torch.relu(self.conv2(o1))
        o3 = torch.relu(self.conv3(o2))
        o4 = torch.relu(self.conv4(o3))
        o4 = self.flatten(o4)
        o5 = torch.relu(self.fully_cnnctd_1(o4))
        o6 = torch.relu(self.fully_cnnctd_2(o5))
        output = self.fully_cnnctd_3(o6)
        return output 
    
    @staticmethod
    def train(model: nn.Module, dataloader: DataLoader, 
            loss_func: nn.MSELoss, optimizer: torch.optim, num_epochs: int ) -> list[float]:
        
        model.train()
        epoch_average_losses = []
        
        for train_epoch in range(num_epochs):
            running_epoch_loss = 0.0
            
            for X_b, Y_b in dataloader:
                optimizer.zero_grad()
                batch_prediction = model.forward(X_b)
                batch_loss = loss_func(batch_prediction, Y_b)
                batch_loss.backward()
                optimizer.step()
                running_epoch_loss += batch_loss.item() * X_b.shape[0]
                epoch_average_losses.append(running_epoch_loss / len(dataloader.dataset))
    
    @staticmethod
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




