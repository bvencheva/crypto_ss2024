import numpy as np
import pandas as pd

class CustomTimeSeriesSplit:
    
    
    def __init__(self,X, train_size, test_size):
        self.train_size = train_size
        self.test_size = test_size
        self.X = X
        
    def run(self):
        self.split()
        self.get_n_splits()
        

    def split(self):
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        splits = []
        
        start_train = 0
        while start_train + self.train_size + self.test_size <= n_samples:
            end_train = start_train + self.train_size
            start_test = end_train
            end_test = start_test + self.test_size
            
            train_indices = indices[start_train:end_train]
            test_indices = indices[start_test:end_test]
            
            splits.append((self.X[train_indices],self.X[test_indices]))
            # splits.append([train_indices, test_indices])
            
            start_train = end_test  # move the window to the next non-overlapping position

        self.splits = splits

    def get_n_splits(self, y=None, groups=None):
        n_samples = len(self.X)
        self.n_splits = (n_samples - self.train_size) // (self.train_size + self.test_size)

        

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


class CVresults(object):
    """
    per crypto
    
    """
    
    def __init__(self,df,model,crypto,splits):
        self.model=model
        self.crypto=crypto
        self.splits=splits
        self.endog = 'close%'
        self.df = df
        assert model in ['linear','random-forest','SVR','XGBoost'], 'Please select relevant model'
        
    def run(self,**args):
        self.calculate_test_results(**args)
    
    def calculate_test_results(self,**args):
        test_results = []
        
        #columns split
        EXOG = [i for i in self.df.columns if i != self.endog]
        BINARY_COLUMNS = self.df.columns[self.df.isin([0,1]).all()]
        EXOG_NUMERICAL = [i for i in self.df.columns if i not in BINARY_COLUMNS and i!= self.endog]
        
        for cv_set in range(0,len(self.splits)):

            x_train = np.array(self.df[EXOG].reindex(self.splits[cv_set][0]))
            x_test = np.array(self.df[EXOG].reindex(self.splits[cv_set][1]))
            
            #linear
            if self.model == 'linear':
                y_train=np.array(self.df[self.endog].reindex(self.splits[cv_set][0]).values)
                y_test = self.df.reindex(self.splits[cv_set][1])[self.endog].values[0]
                
                reg = LinearRegression().fit(x_train, y_train)
                y_pred = np.round(reg.predict(x_test)[0],8)
                                  
                R = reg.score(x_train, y_train)

                test_results.append([self.splits[cv_set][1][0] #time
                                     , y_pred #pred
                                     , np.round(y_test,8) #actual
#                                      ,R
                                    ]) #index, predicted, actual, R

            if self.model == 'random-forest':
                y_train=np.array(self.df[self.endog].reindex(self.splits[cv_set][0]).values)
                y_test=self.df.reindex(self.splits[cv_set][1])[self.endog].values[0]

                regressor = RandomForestRegressor(**args, oob_score=True)
                regressor.fit(x_train, y_train)

                # Making predictions on the same data or new data
                y_pred = regressor.predict(x_test)
                y_pred = np.round(y_pred[0],8)
                
                test_results.append([self.splits[cv_set][1][0]
                     , y_pred
                     , np.round(y_test,8)
                    ]) #index, predicted, actual, R
                
                
            if self.model == 'SVR':                
                data = self.df.copy()
                # rescaling the numerical dataset
                scaler = MinMaxScaler()
                for col in EXOG_NUMERICAL + [self.endog]:
                    data[col] = scaler.fit_transform(data[[col]])
                    
                #split exog and endog
                X = data[EXOG_NUMERICAL+list(BINARY_COLUMNS)]
                y = data[[self.endog]]
                
                #split train, test
                x_train = X.loc[self.splits[cv_set][0]].values
                y_train = np.array(y.loc[self.splits[cv_set][0]][[self.endog]])
                x_test = X.loc[self.splits[cv_set][1]].values
                y_test = np.array(y.loc[self.splits[cv_set][1]][[self.endog]])
                
                #SVR
                arguments = [v for k,v in args.items()]
                kernel, gamma, C, epsilon = arguments[0],arguments[1],arguments[2],arguments[3]
                
                regressor = SVR(kernel=kernel,gamma=gamma,C=C,epsilon=epsilon)
                regressor.fit(x_train, y_train)
                
                y_pred = regressor.predict(x_test)
                y_pred = scaler.inverse_transform(y_pred.reshape(1,1))
                y_pred = np.round(y_pred[0][0],8)
                y_actual = np.round(scaler.inverse_transform(y_test)[0][0],8)

                test_results.append([self.splits[cv_set][1][0] #time
                         , y_pred 
                         , y_actual
#                                      ,R
                        ]) #index, predicted, actual, R

            if self.model == 'XGBoost':
                pass

        assert len(test_results)== len(self.splits)
        self.test_results = test_results
                
