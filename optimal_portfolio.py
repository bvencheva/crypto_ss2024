import numpy as np
import requests
import pandas as pd
from datetime import datetime


# Mean-Variance Optimization without Risk-Free Rate
from scipy.optimize import minimize
# Define get_optimal_weights function

def get_optimal_weights(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for _ in range(num_assets))
    
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
        p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        return -p_ret / p_var
    
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return std_dev, returns
    
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def get_latest_data(crypto):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': f'{crypto}USDT',  # Trading pair
        'interval': '1m',     # Interval (1 minute)
        'limit': 1000      # Number of data points to retrieve (max 1000)
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
    
        df = pd.DataFrame(data, columns=[ 
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].apply(lambda x:float(x))
    
    else:
        print(f"Error: {response.status_code} - {response.text}")

    return df


def get_latest_price(crypto):
    base_url = "https://api.binance.com"
    endpoint = f"/api/v3/ticker/price"
    params = {
        'symbol': f'{crypto}USDT'
    }
    response = requests.get(base_url + endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return float(data['price'])
    else:
        raise Exception(f"Error fetching data from Binance API: {response.status_code}, {response.text}")
        
