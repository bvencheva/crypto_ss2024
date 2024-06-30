import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor 


_dict_weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday', 6:'Sunday'}


def calculate_vif(X, threshold = 10):
    
    vif_data = pd.DataFrame() 
    vif_data["feature"] = X.columns 
      
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                              for i in range(len(X.columns))]

    df_vif = pd.DataFrame(vif_data).sort_values('VIF',ascending=False).set_index('feature')
    first_feature = df_vif.index[0]
    value_first = df_vif['VIF'][first_feature]
    
    drop_feature = first_feature if value_first > threshold else None

    return drop_feature, df_vif
    

def pca_analysis(df,endog, variance_threshold=0.80):
    data = df.copy()
    BINARY_COLUMNS = list(data.columns[data.isin([0,1]).all()])
    EXOG_NUMERICAL = [i for i in data.columns if i not in BINARY_COLUMNS and i!= endog]
    
    # rescaling the numerical dataset
    scaler = StandardScaler()
    for col in EXOG_NUMERICAL:
        data[col] = scaler.fit_transform(data[[col]])
        
    #split exog and endog
    X = data[EXOG_NUMERICAL]

    # PCA Analysis
    pca = PCA()
    pca.fit(X)

    # Determine number of components to explain the given threshold of variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = next(i for i, cumulative_var in enumerate(cumulative_variance) if cumulative_var >= variance_threshold) + 1

    # Fit PCA with the selected number of components
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Get the contribution of each parameter to the components
    components_df = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(n_components)]).T
    components_df = np.abs(components_df)
    components_df = components_df.sort_values(by = ['PC1','PC2','PC3'], ascending=[False, False,False])
    
    return cumulative_variance, n_components, components_df



class FeatureEngineering(object):
    """
    preselected resampled frequency
    
    import talib
    https://ta-lib.org/functions/
    
    
    """
    
    def __init__(self,_dict,nlags = range(1,20),windows= [3,5,10,20]):
        self.windows = windows
        self.nlags = nlags
        self._dict = _dict.copy()
        
        
    def run(self):
        _dict = self._dict.copy()
        
        for crypto,df in _dict.items():
            # df_target = self.add_target(df)
            df_target = df.copy()
            df_time = self.time_features(df_target)
            df_lags = self.get_lags(df_time)
            df_rol = self.rolling_window_same_period(df_lags)
            df_ema = self.calculate_ema_window_weights(df_rol)
            df_macd = self.calculate_macd(df_ema)
            df_add = self.additional(df_macd)
            
            _dict[crypto] = df_add.copy()
            
        self.crypto_added_features = _dict.copy()
        
        self.df_modelling()
        
    # # add main currencies changes..
    # def add_target(self,df):
    #     df['up_dummy'] = df['close%'].apply(lambda x: 1 if x>0 else 0)  #dummy increaase = 1, decrease=1 or classification?        
    #     return df
        
    def time_features(self,df):
        df['year'] = df.index.year
        df['year'] = df['year'].apply(lambda x: 1 if x == 2024 else 0)
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.weekday
        df['weekday'] = df['day_of_week'].apply(lambda x: 1 if x<5 else 0)
        df['day_of_week'] = df['day_of_week'].apply(lambda x: _dict_weekdays[x])
        df['hour'] = df.index.hour
        df = pd.get_dummies(df, columns=['day_of_week'],dtype=float,drop_first=True)
        return df
    
    
    #e-	Lags: #AUC & PAUC to be done 
    def get_lags(self,df,columns = ['close%']):
        for column in columns:
            for n in self.nlags:
                df[f'{column}_{n}lags']= df[column].shift(n)

        return df

    #-	Rolling window statistics - same frequency based
    def rolling_window_same_period(self,df):
        for window in self.windows:
            df[f'rol_mean_close_{window}'] = df['close%'].rolling(window).mean()
            df[f'rol_mean_volume_{window}'] = df['volume%'].rolling(window).mean()
            df[f'rol_std_close_{window}'] = df['close%'].rolling(window).std()
            df[f'rol_std_volume_{window}'] = df['volume%'].rolling(window).std()
            
        return df

    #-	Rolling window statistics - within windows based?
    
    #Calculate the Exponential Moving Average
    def calculate_ema_window_weights(self,df):
        """Calculate the Exponential Moving Average (EMA)"""
        for window in self.windows:
            df[f'ema_mean_{window}'] = df['close%'].ewm(span=window, adjust=False).mean()
            df[f'ema_std_{window}'] = df['close%'].ewm(span=window, adjust=False).std()
        
        return df
    
    
    # Calculate the EMA
    def calculate_ema(self, df, n=12):
        """Calculate the Exponential Moving Average (EMA)"""
        column = [i for i in df.columns][0]
        ema = df[column].ewm(span=n, adjust=False).mean()
        
        return ema

    
    # Calculate the MACD
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate the Moving Average Convergence Divergence (MACD)"""
        ema_fast = self.calculate_ema(df[['close%']], fast)
        ema_slow = self.calculate_ema(df[['close%']], slow)
        macd = ema_fast - ema_slow
        df['macd'] = macd
        
        signal_line = self.calculate_ema(df[['macd']], signal)
        df['signal_line'] = signal_line

        hist = macd - signal_line
        df['hist'] = hist
        
        return df


    # -	Harmonic decomposition, Fourier
    #additional features

    def additional(self,df):
        # df['HLC']=df['high']/df['low'] - 1
        df['HLC']=(df['high']-df['low'])/df['close']
        
        return df
    
    def df_modelling(self):
        _dict_clean = {}
        for crypto, df in self.crypto_added_features.items():
            _dict_clean[crypto] = df.drop(columns = ['open','high','low','close','volume']).dropna()
        
        self.crypto_modelling = _dict_clean.copy()



class FeatureCrypto(FeatureEngineering):

    def __init__(self,raw_df,windows=[3,5,10,20],nlags=range(1,20)):
        self.windows = windows
        self.nlags = nlags
        self.df = raw_df

    def run(self):
        df = self.resample()
        df_time = super().time_features(df)
        df_lags = super().get_lags(df_time)
        df_rol = super().rolling_window_same_period(df_lags)
        df_ema = super().calculate_ema_window_weights(df_rol)
        df_macd = super().calculate_macd(df_ema)
        df_add = super().additional(df_macd)
        self.crypto_added_features = df_add


    def resample(self):
        minutes = 15
        #adjusting values for other freq
        df_resampled = self.df.resample(f'{minutes}min',label='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        })
        #get correct close price as function takes previous minute value
        df_resampled = df_resampled.merge(self.df[['close']],left_index=True,right_index=True)[['open','high','low','close','volume']]
    
        #add main targets:
        df_resampled['close%'] = df_resampled['close']/df_resampled['open']-1
        df_resampled['volume%'] = df_resampled['volume']/df_resampled['volume'].shift(1)-1
    
        return df_resampled

