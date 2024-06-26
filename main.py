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
from LSTM import LSTM 
from CNNLSTM import CNNLSTM
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



        self.data = pd.get_dummies(self.data, columns=['Exchange'], dtype = int)
        sector_to_int, _ = pd.factorize(self.data['Sector'])
        industry_to_int, _ = pd.factorize(self.data['Industry'])

        self.data['Sector_encoded'] = sector_to_int
        self.data['Industry_encoded'] = industry_to_int
        num_sectors = len(self.data['Sector_encoded'].unique())
        num_industries = len(self.data['Industry_encoded'].unique())

        self.data.drop(columns=['Sector', 'Industry'], inplace=True)
       

        ticker_to_int, unique_tickers = pd.factorize(self.data['Symbol'])
        self.data['Symbol_encoded'] = ticker_to_int
        num_tickers = len(self.data['Symbol_encoded'].unique())
        print(f"Unique Tickers: {num_tickers}, Unique Sectors: {num_sectors}, Unique Industries: {num_industries}")

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

    def get_encoded_industries(self):
        return self.data['Industry_encoded'].values

    def get_encoded_sectors(self):
        return self.data['Sector_encoded'].values

class runCNNModel:

    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 500
    

    def run(self):           

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        industries = data_preprocessor.get_encoded_industries()
        sectors = data_preprocessor.get_encoded_sectors()
        dataSet.drop(columns = ['Sector_encoded', 'Industry_encoded'], inplace=True)
        max_industry_index = torch.max(torch.tensor(industries)).item()
        max_sector_index = torch.max(torch.tensor(sectors)).item()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        data_preprocessor.print_all_columns(dataSet)

        y = dataSet['Close'].values.reshape(-1, 1)  
        features = dataSet.drop(columns=['Close'])
        pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
        pd.set_option('display.expand_frame_repr', False)  # Prevents wrapping of columns
        pd.set_option('display.max_colwidth', None)  # Allows full width of column display
        pd.set_option('display.width', 1000)  # Sets the width of the display for wide DataFrames

        X_train, X_test, y_train, y_test, X_train_tickers, X_test_tickers, X_train_sectors, X_test_sectors, X_train_industries, X_test_industries = train_test_split(
            features, y, tickers, sectors, industries, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=8)


        # Normalizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Converting all to tensors for PyTorch
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long, device=device)
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long, device=device)
        tensor_X_sector_train = torch.tensor(X_train_sectors, dtype=torch.long, device=device)
        tensor_X_sector_test = torch.tensor(X_test_sectors, dtype=torch.long, device=device)
        tensor_X_industry_train = torch.tensor(X_train_industries, dtype=torch.long, device=device)
        tensor_X_industry_test = torch.tensor(X_test_industries, dtype=torch.long, device=device)

        # Creating datasets
        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_X_sector_train, tensor_X_industry_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_X_sector_test, tensor_X_industry_test, tensor_y_test)


        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.cnn_model = CNN(device, max_ticker_index + 1, max_sector_index + 1, max_industry_index + 1, dataSet.shape[1] - 1).to(device)
        self.CNN_loss_func = nn.MSELoss()
        self.CNN_optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate,  weight_decay=1e-5)
        self.CNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.CNN_optimizer, mode='min', factor=0.9, patience = 2)


        train(device, self.cnn_model, dataloader_train_set, self.CNN_loss_func, self.CNN_optimizer, self.CNN_scheduler, self.epochs)
        test_model(device, self.cnn_model, dataloader_test_set, self.CNN_loss_func)

class runDNNModel:

    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 500
    
   
    def run(self):           
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        industries = data_preprocessor.get_encoded_industries()
        sectors = data_preprocessor.get_encoded_sectors()
        dataSet.drop(columns = ['Sector_encoded', 'Industry_encoded'], inplace=True)
        max_industry_index = torch.max(torch.tensor(industries)).item()
        max_sector_index = torch.max(torch.tensor(sectors)).item()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        data_preprocessor.print_all_columns(dataSet)



        y = dataSet['Close'].values.reshape(-1, 1)  
        features = dataSet.drop(columns=['Close'])
        pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
        pd.set_option('display.expand_frame_repr', False)  # Prevents wrapping of columns
        pd.set_option('display.max_colwidth', None)  # Allows full width of column display
        pd.set_option('display.width', 1000)  # Sets the width of the display for wide DataFrames

        X_train, X_test, y_train, y_test, X_train_tickers, X_test_tickers, X_train_sectors, X_test_sectors, X_train_industries, X_test_industries = train_test_split(
            features, y, tickers, sectors, industries, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=8)


        # Normalizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Converting all to tensors for PyTorch
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long, device=device)
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long, device=device)
        tensor_X_sector_train = torch.tensor(X_train_sectors, dtype=torch.long, device=device)
        tensor_X_sector_test = torch.tensor(X_test_sectors, dtype=torch.long, device=device)
        tensor_X_industry_train = torch.tensor(X_train_industries, dtype=torch.long, device=device)
        tensor_X_industry_test = torch.tensor(X_test_industries, dtype=torch.long, device=device)

        # Creating datasets
        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_X_sector_train, tensor_X_industry_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_X_sector_test, tensor_X_industry_test, tensor_y_test)

        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.dnn_model = DNN(device, max_ticker_index + 1, max_sector_index + 1, max_industry_index + 1, dataSet.shape[1] - 1).to(device)
        self.DNN_loss_func = nn.MSELoss()
        self.DNN_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.learning_rate,  weight_decay=1e-5)
        self.DNN_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.DNN_optimizer, mode='min', factor=0.9, patience = 4)


        train(device, self.dnn_model, dataloader_train_set, self.DNN_loss_func, self.DNN_optimizer, self.DNN_scheduler, self.epochs)
        test_model(self.dnn_model, dataloader_test_set, self.DNN_loss_func)

class runLSTMModel:
    
    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 500

    def run(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        industries = data_preprocessor.get_encoded_industries()
        sectors = data_preprocessor.get_encoded_sectors()
        dataSet.drop(columns = ['Sector_encoded', 'Industry_encoded'], inplace=True)
        max_industry_index = torch.max(torch.tensor(industries)).item()
        max_sector_index = torch.max(torch.tensor(sectors)).item()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        data_preprocessor.print_all_columns(dataSet)

        y = dataSet['Close'].values.reshape(-1, 1)  
        features = dataSet.drop(columns=['Close'])
        pd.set_option('display.max_columns', None)  # Ensures all columns are displayed
        pd.set_option('display.expand_frame_repr', False)  # Prevents wrapping of columns
        pd.set_option('display.max_colwidth', None)  # Allows full width of column display
        pd.set_option('display.width', 1000)  # Sets the width of the display for wide DataFrames

        X_train, X_test, y_train, y_test, X_train_tickers, X_test_tickers, X_train_sectors, X_test_sectors, X_train_industries, X_test_industries = train_test_split(
            features, y, tickers, sectors, industries, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=8)

        # Normalizing the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Converting all to tensors for PyTorch
        tensor_X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long, device=device)
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long, device=device)
        tensor_X_sector_train = torch.tensor(X_train_sectors, dtype=torch.long, device=device)
        tensor_X_sector_test = torch.tensor(X_test_sectors, dtype=torch.long, device=device)
        tensor_X_industry_train = torch.tensor(X_train_industries, dtype=torch.long, device=device)
        tensor_X_industry_test = torch.tensor(X_test_industries, dtype=torch.long, device=device)

        # Creating datasets
        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_X_sector_train, tensor_X_industry_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_X_sector_test, tensor_X_industry_test, tensor_y_test)


        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        lstm_model = LSTM(device, max_ticker_index + 1, max_sector_index + 1, max_industry_index + 1, dataSet.shape[1] - 1).to(device)
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)

        train(device, lstm_model, dataloader_train_set, loss_func, optimizer, scheduler, self.epochs)
        test_model(device, lstm_model, dataloader_test_set, loss_func)


class runCNNLSTMModel:

    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.epochs = 500
    

    def run(self):           

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        data_preprocessor = Preprocess('/content/drive/MyDrive/stock_predict_data')
        data_preprocessor.load_data()
        data_preprocessor.data_preprocess()
        dataSet = data_preprocessor.get_preprocessed_data()
        tickers = data_preprocessor.get_encoded_tickers()
        industries = data_preprocessor.get_encoded_industries()
        sectors = data_preprocessor.get_encoded_sectors()
        dataSet.drop(columns = ['Sector_encoded', 'Industry_encoded'], inplace=True)
        max_industry_index = torch.max(torch.tensor(industries)).item()
        max_sector_index = torch.max(torch.tensor(sectors)).item()
        max_ticker_index = torch.max(torch.tensor(tickers)).item()
        data_preprocessor.print_all_columns(dataSet)

        y = dataSet['Close'].values.reshape(-1, 1)  
        features = dataSet.drop(columns=['Close'])
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.expand_frame_repr', False)  
        pd.set_option('display.max_colwidth', None)  
        pd.set_option('display.width', 1000)  

        X_train, X_test, y_train, y_test, X_train_tickers, X_test_tickers, X_train_sectors, X_test_sectors, X_train_industries, X_test_industries = train_test_split(
            features, y, tickers, sectors, industries, 
            train_size=0.9, 
            test_size=0.1, 
            random_state=8)


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        tensor_X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        tensor_X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        tensor_y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

        tensor_X_train_tickers = torch.tensor(X_train_tickers, dtype=torch.long, device=device)
        tensor_X_test_tickers = torch.tensor(X_test_tickers, dtype=torch.long, device=device)
        tensor_X_sector_train = torch.tensor(X_train_sectors, dtype=torch.long, device=device)
        tensor_X_sector_test = torch.tensor(X_test_sectors, dtype=torch.long, device=device)
        tensor_X_industry_train = torch.tensor(X_train_industries, dtype=torch.long, device=device)
        tensor_X_industry_test = torch.tensor(X_test_industries, dtype=torch.long, device=device)

        train_dataset = TensorDataset(tensor_X_train, tensor_X_train_tickers, tensor_X_sector_train, tensor_X_industry_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_X_test, tensor_X_test_tickers, tensor_X_sector_test, tensor_X_industry_test, tensor_y_test)


        dataloader_train_set = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dataloader_test_set = DataLoader(test_dataset, batch_size=self.batch_size)

        self.cnn_lstm_model = CNN(device, max_ticker_index + 1, max_sector_index + 1, max_industry_index + 1, dataSet.shape[1] - 1).to(device)
        self.CNN_lstm_loss_func = nn.MSELoss()
        self.CNN_lstm_optimizer = optim.Adam(self.cnn_lstm_model.parameters(), lr=self.learning_rate,  weight_decay=1e-5)
        self.CNN_lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.CNN_lstm_optimizer, mode='min', factor=0.9, patience = 3)


        train(device, self.cnn_lstm_model, dataloader_train_set, self.CNN_lstm_loss_func,  self.CNN_lstm_optimizer, self.CNN_lstm_scheduler, self.epochs)
        test_model(device, self.cnn_lstm_model, dataloader_test_set, self.CNN_lstm_loss_func)

if __name__ == "__main__":
    model_runner = runDNNModel()  
    model_runner.run()