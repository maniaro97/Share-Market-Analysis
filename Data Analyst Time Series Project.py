#!/usr/bin/env python
# coding: utf-8

# 
# # INSTALL LIBRARIES

# In[2]:


get_ipython().system('pip install pandas_datareader')


# In[3]:


get_ipython().system('pip install dash')


# In[4]:


get_ipython().system('pip install yfinance')


# # IMPORT LIBRARIES

# In[9]:


import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.tsa.arima.model import ARIMA
get_ipython().run_line_magic('matplotlib', 'inline')


# # DATA PROCESSING

# In[67]:


start="2020-03-01"
end="2024-03-31"
tcs=yf.download('TCS',start,end)
infy=yf.download('INFY',start,end)
wipro=yf.download('WIPRO.NS',start,end)
plt.show()


# In[30]:


tcs


# In[31]:


wipro


# In[32]:


infy


# In[33]:


MRF


# In[50]:


start_date = '2024-01-01'
end_date = '2024-01-31'
# Fetch historical stock data for TCS, Infosys, and Wipro
tcs = yf.download('TCS', start=start_date, end=end_date)
infy = yf.download('INFY', start=start_date, end=end_date)
wipro = yf.download('WIPRO.NS', start=start_date, end=end_date)
# Plot the volume of stock traded
tcs['Volume'].plot(label='TCS', figsize=(15, 7))
infy['Volume'].plot(label='INFOSYS')
wipro['Volume'].plot(label='WIPRO')
plt.title('Volume of Stock Traded ({} to {})'.format(start_date, end_date))
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()


# In[69]:


# Set start and end dates for the data download
start_date = "2020-01-01"
end_date = "2024-03-31"
# Calculate moving averages for each company
for company, df in zip(["TCS", "Wipro", "Infosys","MRF"], [tcs, wipro, infy,MRF]):
    df["MA50"] = df["Open"].rolling(50).mean()
    df["MA200"] = df["Open"].rolling(200).mean()
# Plotting
plt.figure(figsize=(15, 7))

# Plot TCS data
plt.plot(tcs['Open'], label='TCS Open Price')
plt.plot(tcs['MA50'], label='TCS 50-Day Moving Average')
plt.plot(tcs['MA200'], label='TCS 200-Day Moving Average')

# Plot Wipro data
plt.plot(wipro['Open'], label='Wipro Open Price')
plt.plot(wipro['MA50'], label='Wipro 50-Day Moving Average')
plt.plot(wipro['MA200'], label='Wipro 200-Day Moving Average')

# Plot Infosys data
plt.plot(infy['Open'], label='Infosys Open Price')
plt.plot(infy['MA50'], label='Infosys 50-Day Moving Average')
plt.plot(infy['MA200'], label='Infosys 200-Day Moving Average')
# Plot MRF data
plt.plot(MRF['Open'], label='MRF Open Price')
plt.plot(MRF['MA50'], label='MRF 50-Day Moving Average')
plt.plot(MRF['MA200'], label='MRF 200-Day Moving Average')
# Adding titles and labels
plt.title('Stock Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Displaying the plot
plt.show()


# In[47]:


# Plotting
plt.figure(figsize=(15, 7))
# Plot TCS data
plt.plot(tcs['Open'], label='TCS Open Price')
plt.plot(tcs['MA50'], label='TCS 50-Day Moving Average')
plt.plot(tcs['MA200'], label='TCS 200-Day Moving Average')
# Displaying the plot
plt.show()


# In[70]:


import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
stock_symbols = {"Infosys (INFY.NS)": "INFY.NS","Reliance Industries (RELIANCE.BO)": "RELIANCE.BO",
"Tata Consultancy Services (TCS.NS)": "TCS.NS","HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
"ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS","Sensex (^BSESN)": "^BSESN",
"Nifty 50 (^NSEI)": "^NSEI","Bank Nifty (^NSEBANK)": "^NSEBANK"}
# Function to fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data['Close']
# Fit ARIMA model to the stock data and make predictions
def fit_arima_model(stock_data):
    # Fit ARIMA model
    model = ARIMA(stock_data, order=(5,1,0))
    model_fit = model.fit()
    # Make predictions
    predictions = model_fit.forecast(steps=365)
    return predictions
# Plot actual vs. predicted stock prices
def plot_predictions(stock_data, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data.values, label='Actual', color='blue')
    plt.plot(pd.date_range(start=stock_data.index[-1], periods=len(predictions), freq='D'), predictions, label='Predicted', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    # Display predefined list of stock symbols to the user
    print("Choose a stock:")
    for i, (company, symbol) in enumerate(stock_symbols.items()):
        print(f"{i+1}. {company}")
    # Allow the user to select a stock symbol
    selection = int(input("Enter the number corresponding to the stock: "))
    selected_symbol = list(stock_symbols.values())[selection-1]
    # Get user input for start date and end date
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    # Fetch historical stock data
    stock_data = fetch_stock_data(selected_symbol, start_date, end_date)
    # Fit ARIMA model and make predictions
    predictions = fit_arima_model(stock_data)
    # Plot actual vs. predicted stock prices
    plot_predictions(stock_data, predictions)


# In[ ]:




