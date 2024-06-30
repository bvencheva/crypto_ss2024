import os
import numpy as np
import pandas as pd
from datetime import datetime

FREQ = [5,10,15]

class LoadTransformCryptoData():
    
    def __init__(self,minutes=FREQ):
        self.minutes = minutes
        self.load_crypto_data()
        self.resample()
                
    def load_crypto_data(self):
        _dict = {}
        for file in os.listdir('data'):
            if '.csv' in file:# and ('ADA' in file or 'ALGO' in file):
                _df = pd.read_csv(f'data/{file}',delimiter=',')
                _df['timestamp']= _df['timestamp'].apply(lambda x:  datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                _dict[file.split('_')[0]] = _df.set_index('timestamp')

        self.raw_data=_dict.copy()
        
    # check for missing data points? minutes?
    def resample(self):

        _dict = {}
        _dict[1] = {}
        for minutes in self.minutes:
            _dict[minutes]={}
            
            for k, df in self.raw_data.items():
                _df = df.copy()
                #add main variables for same freq
                _df['close%'] = _df['close']/_df['open']-1
                _df['volume%'] = _df['volume']/_df['volume'].shift(1)-1
                _dict[1][k] = _df.copy()

                #adjusting values for other freq
                df_resampled = df.resample(f'{minutes}min',label='right').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
#                     'close': 'last',
                    'volume': 'sum'
                })
                #get correct close price as function takes previous minute value
                df_resampled = df_resampled.merge(df[['close']],left_index=True,right_index=True)[['open','high','low','close','volume']]

                #add main targets:
                df_resampled['close%'] = df_resampled['close']/df_resampled['open']-1
                df_resampled['volume%'] = df_resampled['volume']/df_resampled['volume'].shift(1)-1

                _dict[minutes][k] = df_resampled.copy()

        print(f"Resampled Data ({self.minutes} minutes):")

        self.resampled_data = _dict.copy()



