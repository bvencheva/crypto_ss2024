#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:05:41 2024

@author: radostinageorgieva
"""

import pandas as pd
import glob
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Specify the folder path containing CSV files
folder_path = '/Users/radostinageorgieva/Projects/Summer_School/data'  # Replace with your folder path

# Create a file pattern to match all CSV files in the folder
file_pattern = os.path.join(folder_path, '*.csv')

# Get the list of all CSV files in the folder
all_files = glob.glob(file_pattern)

# Load each CSV file into a separate DataFrame and name the DataFrame after the file (without "_1m" and in lowercase)
dataframes = {}
for file in all_files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    df_name = file_name.replace(" ", "_").replace("-", "_").lower()
    if df_name.endswith('_usdt_1m'):
        df_name = df_name[:-8]  # Remove the last 3 characters '_usdt_1m'
        dataframes[df_name] = pd.read_csv(file)

for name, df in dataframes.items():
    # Check for null values in the 'close' column
    print(f"{name} 'close' column null values: {df['close'].isnull().sum()}")
    # Create 'close_1' column by shifting 'close' and filling NA with 'open'
    df['close_1'] = df['close'].shift(1).fillna(df['open'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Drop rows with NaT timestamps if there are any
    df = df.dropna(subset=['timestamp'])

# Function to check stationarity and difference the data if necessary
def ensure_stationarity(series, max_diff=2):
    """
    Ensures that a time series is stationary by applying differencing if necessary.

    Parameters:
    series (pd.Series): The time series to check for stationarity.
    max_diff (int): The maximum number of differences to apply.

    Returns:
    pd.Series: The stationary time series.
    """
    for i in range(max_diff + 1):
        p_value = adfuller(series)[1]  # Perform ADF test and get the p-value
        if p_value < 0.05:
            return series  # The series is stationary, return it
        series = series.diff().dropna()  # Apply differencing and drop NaN values
    return series  # Return the differenced series

# Apply transformations and identify outliers
for name, df in dataframes.items():
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with NaT timestamps if there are any
    df = df.dropna(subset=['timestamp'])
    
    # Resample data to 15-minute intervals
    df = df.set_index('timestamp').resample('15T').agg({'close': 'last', 'open': 'first'}).dropna().reset_index()
    
    # Create a 'quarter' column
    df['quarter'] = df['timestamp'].dt.to_period('Q')
    
    print(df['quarter'])
    
    # Group by quarter and split into separate DataFrames
    quarters = dict(tuple(df.groupby('quarter')))
    
    print(df['quarter'].unique())
    
    # Check for null values in the 'close' column
    print(f"{name} 'close' column null values: {df['close'].isnull().sum()}")
    
    # Create 'close_1' column by shifting 'close' and filling NA with 'open'
    df['close_1'] = df['close'].shift(1).fillna(df['open'])
    
    # Plot 'close_1' with title as the name of the DataFrame
    df["close_1"].plot(kind="line", title=f"{name} Close_1")
    plt.xlabel("Index")
    plt.ylabel("Close_1")
    plt.title(f"{name} Close_1")
    plt.show()
    
    # Now group by 3-month cluster and apply ARIMA model
    grouped = df.groupby('quarter')
    clusters = list(grouped.groups.keys())
    
    # Group by quarter and split into separate DataFrames
    quarters = dict(tuple(df.groupby('quarter')))

    # Iterate over each quarter except the last one
    for quarter in list(quarters.keys())[:-1]:
        train_cluster = quarters[quarter]
        train_cluster = train_cluster.sort_values(by='timestamp')
        train_cluster['close_1'] = ensure_stationarity(train_cluster['close_1'])

        # Test on the first day starting from the end of the current cluster
        end_of_train_period = train_cluster['timestamp'].iloc[-1]
        test_cluster = df[(df['timestamp'] > end_of_train_period) & (df['timestamp'] <= end_of_train_period + pd.DateOffset(days=1))]
        test_cluster = test_cluster.sort_values(by='timestamp')
        test_cluster['close_1'] = ensure_stationarity(test_cluster['close_1'])

        try:
            model = ARIMA(train_cluster['close_1'], order=(5, 1, 0))
            model_fit = model.fit()
         
            # Forecast the next day (24 hours * 4 intervals per hour = 96 intervals)
            forecast = model_fit.forecast(steps=96)

            # Ensure the test and forecast periods align
            forecast_index = pd.date_range(start=test_cluster['timestamp'].iloc[0], periods=96, freq='15T')
             
            # Plot the combined test and forecast data
            plt.figure(figsize=(10, 6))
            plt.plot(test_cluster['timestamp'], test_cluster['close_1'], label='Test', color='orange')
            plt.plot(forecast_index, forecast, color='green', label='Forecast')
            plt.title(f'Test and Forecast Data for {name} - Day following Quarter {quarter}')
            plt.xlabel('Timestamp')
            plt.ylabel('Close_1')
            plt.xlim(test_cluster['timestamp'].iloc[0], forecast_index[-1])
            plt.legend()
            plt.xticks(rotation=90)  # Rotate the x-axis labels by 90 degrees
            plt.show()
        except Exception as e:
            print(f"ARIMA model failed for train cluster {quarter}: {e}")

    # Predict for the last quarter
    last_quarter = list(quarters.keys())[-1]
    train_cluster = quarters[last_quarter]
    train_cluster = train_cluster.sort_values(by='timestamp')
    train_cluster['close_1'] = ensure_stationarity(train_cluster['close'].shift(1).fillna(train_cluster['open']))

    try:
        model = ARIMA(train_cluster['close_1'], order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the next day (24 hours * 4 intervals per hour = 96 intervals)
        forecast = model_fit.forecast(steps=96)
                
        # Plot the forecast for the fourth cluster
        plt.figure(figsize=(10, 6))
        forecast_index = pd.date_range(start=train_cluster['timestamp'].iloc[-1] + pd.Timedelta(minutes=15), periods=96, freq='15T')
        plt.plot(forecast_index, forecast, color='green', label='Forecast')
        plt.title(f'ARIMA Model Forecast for {name} - Day following Quarter {last_quarter}')
        plt.xlabel('Timestamp')
        plt.ylabel('Close_1')
        plt.legend()
        plt.xticks(rotation=90)  # Rotate the x-axis labels by 90 degrees
        plt.show()
            
        # Save the forecast to the results DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': forecast_index,
            'forecast': forecast,
        }).iloc[:4]
        
        # Append forecast results to the original DataFrame
        merged_df = df.merge(forecast_df, on='timestamp', how='left')
        dataframes[name] = merged_df

        # Save the updated DataFrame with forecasts to a CSV file
        merged_df.to_csv(f'{name}_with_forecasts.csv', index=False)
        print(forecast_df)

    except Exception as e:
        print(f"ARIMA model failed for quarter {last_quarter}: {e}")
