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
from DNN import train, test_model

class Preprocess:

    def __init__(self, path_to_dataset):
        
        self.sp500_index_price_data = None 
        self.sp500_companies_data = None 
        self.sp500_companies_stock_price_data = None
        self.path_to_dataset = path_to_dataset

    def load_data(self):
        print("Loading data...")        
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
        print("Preprocessing data...")

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

        self.data = filtered_stocks.merge(filtered_index, on='Date', how='inner')

        companies_info_req = ['Symbol', 'Exchange', 'Sector', 'Industry']
        filtered_companies = companies[companies_info_req]

        self.data = self.data.merge(filtered_companies, on='Symbol', how='left')
        self.data.dropna(inplace=True)



        self.data = pd.get_dummies(self.data, columns=['Exchange', 'Sector', 'Industry'], dtype = int)

       

        ticker_to_int, unique_tickers = pd.factorize(self.data['Symbol'])
        self.data['Symbol_encoded'] = ticker_to_int
        self.ticker_encoded = self.data['Symbol_encoded'].values
        self.int_to_ticker_map = {i: ticker for i, ticker in enumerate(unique_tickers)}
        self.data.drop(columns=['Symbol', 'Symbol_encoded'], inplace=True)
        
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d')
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day_of_Week'] = self.data['Date'].dt.day 
        self.data.drop(columns = ['Date'], inplace=True)
        print("Data preprocessing completed.")

    def print_all_columns(self, data):
        # Convert column names to a single string with each name separated by a newline
        columns_string = "\n".join(data.columns)
        print(columns_string)

    def get_preprocessed_data(self):
        return self.data
    
    def get_encoded_tickers(self):
        return self.ticker_encoded

    def get_num_unique_tickers(self):
        return len(self.int_to_ticker_map)
"""

#class runCNNModel:

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
"""

class runDNNModel:

    def __init__(self):
        self.batch_size = 256
        self.learning_rate = 0.01
        self.epochs = 500
    
   
    def run(self):           

        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        num_unique_tickers = data_preprocessor.get_num_unique_tickers()
        data_preprocessor.print_all_columns(dataSet)



        y = dataSet['Close'].values.reshape(-1, 1)  
        dataSet.drop(columns=['Close'], inplace=True)
        pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
        pd.set_option('display.expand_frame_repr', False)  # Prevents wrapping of columns
        pd.set_option('display.max_colwidth', None)  # Allows full width of column display
        pd.set_option('display.width', 1000)  # Sets the width of the display for wide DataFrames

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
        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.int)  
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.int)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_y_test)


        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dnn_model = DNN(max_ticker_index + 1, dataSet.shape[1])
        self.DNN_loss_func = nn.MSELoss()
        self.DNN_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.learning_rate)
        self.DNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.DNN_optimizer, mode='min', factor=0.5, patience= 2)


        train(self.dnn_model, dataloader_train_set, self.DNN_loss_func, self.DNN_optimizer, self.DNN_scheduler, self.epochs)
        test_model(self.dnn_model, dataloader_test_set, self.DNN_loss_func)


if __name__ == "__main__":
    model_runner = runDNNModel()  
    model_runner.run()