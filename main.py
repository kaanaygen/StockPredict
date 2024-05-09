import os
import nasdaqdatalink
import zipfile
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

    def __init__(self, api_key):
        nasdaqdatalink.ApiConfig.api_key = api_key
        self.EOD_tickers = None 
        self.EOD_prices = None 
       
    def load_data(self, tickers_table, prices_table):
        self.EOD_tickers = nasdaqdatalink.export_table(tickers_table, filename = '/content/drive/MyDrive/Dset1.zip')
        self.EOD_prices = nasdaqdatalink.export_table(prices_table, filename = '/content/drive/MyDrive/Dset2.zip')
        with zipfile.ZipFile('/content/drive/MyDrive/Dset1.zip', 'r') as zip_ref:
            zip_ref.extractall('/content/drive/MyDrive/tickers_Data')
        with zipfile.ZipFile('/content/drive/MyDrive/Dset2.zip', 'r') as zip_ref:
            zip_ref.extractall('/content/drive/MyDrive/prices_Data')            
    
    def data_preprocess(self):
        """
        nyse_calendar = pmc.get_calendar('NYSE')
        nyse_schedule = nyse_calendar.schedule(start_date=start_date, end_date=end_date)
        days_nyse_open = pmc.date_range(nyse_schedule, frequency = '1D')
        days_nyse_open = days_nyse_open.tz_localize(None).normalize().date
        days_nyse_open_timeSeries = pd.Series(days_nyse_open)
        """
        ticker_file = [f for f in os.listdir('/content/drive/MyDrive/tickers_Data') if f.endswith('.csv')][0]
        price_file = [f for f in os.listdir('/content/drive/MyDrive/prices_Data') if f.endswith('.csv')][0]
        self.tickerData = pd.read_csv(f'/content/drive/MyDrive/tickers_Data/{ticker_file}')
        self.priceData = pd.read_csv(f'/content/drive/MyDrive/prices_Data/{price_file}')
        self.data = pd.merge(self.tickerData, self.priceData, on='ticker', how='inner')
        self.data.dropna(inplace=True)
        
        ticker_to_int, unique_tickers = pd.factorize(self.data['ticker'])
        self.data['ticker_encoded'] = ticker_to_int
        self.ticker_encoded = self.data['ticker_encoded'].values
        self.int_to_ticker_map = {i: ticker for i, ticker in enumerate(unique_tickers)}
        self.data.drop(columns=['ticker', 'ticker_encoded'], inplace=True)
        
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d')
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.dayofweek 
        self.data = pd.get_dummies(self.data, columns=['month', 'day_of_week'], dtype = np.float16)
        self.data.drop(columns = ['date'], inplace=True)


        self.data = pd.get_dummies(self.data, columns=['exchange'], dtype = np.float16)
        self.data.drop(columns=['company_name'], inplace=True)
        self.data.columns = [col.replace(' ', '_') for col in self.data.columns]

    def get_preprocessed_data(self):
        return self.data
    
    def get_encoded_tickers(self):
        return self.ticker_encoded

    def get_num_unique_tickers(self):
        return len(self.int_to_ticker_map)


class runCNNModel:

    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 100
    

    def run(self):           

        data_preprocessor = Preprocess(api_key='hwPz-N4Amv1UHR8j5z3C')
        data_preprocessor.load_data('QUOTEMEDIA/TICKERS', 'QUOTEMEDIA/DAILYPRICES')
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()

        y = dataSet['close'].values.reshape(-1, 1)  
        dataSet.drop(columns=['close'], inplace=True)

        X_train, X_test, X_train_tickers, X_test_tickers, y_train, y_test = train_test_split(
            dataSet, tickers, y, 
            train_size=0.8, 
            test_size=0.2, 
            random_state=8)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # Only transform, do not fit!
       
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long)  
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_y_test)

        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.cnn_model = CNN(max_ticker_index + 1, dataSet.shape[1])
        self.CNN_loss_func = nn.MSELoss()
        self.CNN_optimizer = optim.SGD(self.cnn_model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.CNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.CNN_optimizer, mode='min', factor=0.5, patience= 2)


        train(self.cnn_model, dataloader_train_set, self.CNN_loss_func, self.CNN_optimizer, self.CNN_scheduler, self.epochs)
        test_model(self.cnn_model, dataloader_test_set, self.CNN_loss_func)

class runDNNModel:

    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 100
    
   
    def run(self):           

        data_preprocessor = Preprocess(api_key='hwPz-N4Amv1UHR8j5z3C')
        data_preprocessor.load_data('QUOTEMEDIA/TICKERS', 'QUOTEMEDIA/DAILYPRICES')
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        num_unique_tickers = data_preprocessor.get_num_unique_tickers()

        print("Max Ticker Index:", max_ticker_index)
        print("Number of Unique Tickers:", num_unique_tickers)


        y = dataSet['close'].values.reshape(-1, 1)  
        dataSet.drop(columns=['close'], inplace=True)

        X_train, X_test, X_train_tickers, X_test_tickers, y_train, y_test = train_test_split(
            dataSet, tickers, y, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=8)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 

        tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long)  
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_y_test)


        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dnn_model = DNN(max_ticker_index + 1, dataSet.shape[1])
        self.DNN_loss_func = nn.MSELoss()
        self.DNN_optimizer = optim.Adam(self.dnn_model.parameters(), lr = self.learning_rate)
        self.DNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.DNN_optimizer, mode='min', factor=0.5, patience= 2)


        train(self.dnn_model, dataloader_train_set, self.DNN_loss_func, self.DNN_optimizer, self.DNN_scheduler, self.epochs)
        test_model(self.dnn_model, dataloader_test_set, self.DNN_loss_func)


if __name__ == "__main__":
    model_runner = runDNNModel()  
    model_runner.run()