import os
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import pandas_market_calendars as pmc 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CNN import CNN 
from DNN import DNN 
from CNN import train, test_model

class Preprocess:

    def __init__(self, path_to_dataset):
        
        self.sp500_index_price_data = None 
        self.sp500_companies_data = None 
        self.sp500_companies_stock_price_data = None
        self.path_to_dataset = path_to_dataset

    def load_data(self):
                
        for dirname, _, filenames in os.walk(self.path_to_dataset):
            for filename in filenames:
                data_path = os.path.join(dirname, filename)
                if filename == 'sp500_stocks.csv':
                    self.sp500_companies_stock_price_data = data_path
                if filename == 'sp500_companies.csv':
                    self.sp500_companies_data = data_path
                if filename == 'sp500_index.csv':
                    self.sp500_index_price_data = data_path

    def data_preprocess(self):
        
        stocks = pd.read_csv(self.sp500_companies_stock_price_data, parse_dates= ['Date'])
        index = pd.read_csv(self.sp500_index_price_data, parse_dates= ['Date'])
        companies = pd.read_csv(self.sp500_companies_data)
        stocks['Date'] = pd.to_datetime(stocks['Date']).dt.date
        index['Date'] = pd.to_datetime(index['Date']).dt.date

        start_date = max(stocks['Date'].min(), index['Date'].min())
        end_date = min(stocks['Date'].max(), index['Date'].max())

        nyse_calendar = pmc.get_calendar('NYSE')
        nyse_schedule = nyse_calendar.schedule(start_date=start_date, end_date=end_date)
        days_nyse_open = pmc.date_range(nyse_schedule, frequency = '1D')
        days_nyse_open = days_nyse_open.tz_localize(None).normalize().date
        days_nyse_open_timeSeries = pd.Series(days_nyse_open)

        filtered_stocks = stocks[stocks['Date'].isin(days_nyse_open_timeSeries)]
        filtered_index = index[index['Date'].isin(days_nyse_open_timeSeries)]

        working_data = filtered_stocks.merge(filtered_index, on='Date', how='inner')
        working_data['Date'] = pd.to_datetime(working_data['Date']).dt.date

        companies_info_req = ['Symbol', 'Exchange', 'Sector', 'Industry']
        filtered_companies = companies[companies_info_req]

        working_data = working_data.merge(filtered_companies, on='Symbol', how='left')
        working_data = working_data.dropna()
        self.working_data_final = pd.get_dummies(working_data, columns=['Symbol', 'Exchange', 'Sector', 'Industry'], dtype = np.float16)
        self.working_data_final['Date'] = pd.to_datetime(self.working_data_final['Date'], format='%Y-%m-%d')

        self.working_data_final['Year'] = self.working_data_final['Date'].dt.year
        self.working_data_final['Month'] = self.working_data_final['Date'].dt.month
        self.working_data_final['Day_of_Week'] = self.working_data_final['Date'].dt.dayofweek 

        self.working_data_final = pd.get_dummies(self.working_data_final, columns=['Year', 'Month', 'Day_of_Week'], dtype = np.float16)

        self.working_data_final.drop(columns=['Date'], inplace=True)

    def get_preprocessed_data(self):
        return self.working_data_final

class runCNNModel:

    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 25
    

    def split_normalize_XY(self, data):
        X = data.drop(columns=['Close']).values
        Y = data['Close'].values.reshape(-1, 1)
        normalized_X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2 , random_state = 8)
        return (X_train, X_test, y_train, y_test)


    def run(self):           

        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        data = data_preprocessor.get_preprocessed_data()

        X_train, X_test, y_train, y_test = self.split_normalize_XY(data)
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.cnn_model = CNN().float()
        self.CNN_loss_func = nn.MSELoss()
        self.CNN_optimizer = optim.Adam(self.cnn_model.parameters(), lr = self.learning_rate)

        train(self.cnn_model, dataloader_train_set, self.CNN_loss_func, self.CNN_optimizer, self.epochs)
        test_model(self.cnn_model, dataloader_test_set, self.CNN_loss_func)

class runDNNModel:

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.01
        self.epochs = 30
    
    def split_normalize_XY(self, data):
        X = data.drop(columns=['Close']).values
        Y = data['Close'].values.reshape(-1, 1)
        normalized_X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2 , random_state = 8)
        return (X_train, X_test, y_train, y_test)


    def run(self):           

        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        data = data_preprocessor.get_preprocessed_data()

        X_train, X_test, y_train, y_test = self.split_normalize_XY(data)
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dnn_model = DNN()
        self.DNN_loss_func = nn.MSELoss()
        self.DNN_optimizer = optim.Adam(self.dnn_model.parameters(), lr = self.learning_rate)

        train(self.dnn_model, dataloader_train_set, self.DNN_loss_func, self.DNN_optimizer, self.epochs)
        test_model(self.dnn_model, dataloader_test_set, self.DNN_loss_func)


if __name__ == "__main__":
    model_runner = runDNNModel()  
    model_runner.run()