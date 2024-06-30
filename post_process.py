import numpy as np
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

models ={#'linear':None,
         'SVR(1)':{'kernel':'linear','gamma':0.2,'C':10,'epsilon':0.05},
          'SVR(2)':{'kernel':'rbf','gamma':0.2,'C':5,'epsilon':0.01},
          'SVR(3)':{'kernel':'poly','gamma':0.2,'C':2,'epsilon':0.05}, #['poly',0.2,10,0.01],
          'random-forest(1)':{'n_estimators':30,'random_state':0},
          'random-forest(2)':{'n_estimators':50,'random_state':3},
        }


class ProcessResultsCrypto(object):
    """
    df - resamppled with close and open price for a given crypto
    cv_results - results from cv for a given crypto
    """

    def __init__(self,crypto,df_resampled,cv_results):
        self.df = df_resampled
        self.cv_results = cv_results
        self.crypto=crypto

    def run(self):
        self.transform_results()
        self.create_graphs()

    def transform_results(self):
        converted_dict = {}
        stats_dict = {}
        
        for model,data in self.cv_results.items():
            
            df = pd.DataFrame(data,columns = ['timestamp','predicted%','actual%']).set_index('timestamp')
            #join actual from resampled, open
            price_open = self.df[['open','close']]
            
            converted = df.merge(price_open, how='left',left_index=True,right_index=True).rename(columns = {'close':'actual_price'})
            
            converted['predicted_price'] = converted['open']*(1+converted['predicted%'])
            # converted['actual%'] = converted['actual%']*100
            # converted['predicted%'] = converted['predicted%']*100
            converted['d'] = converted.apply(lambda x: 1 if x['actual%']*x['predicted%'] > 0 else 0,axis=1)            
            
            D = np.sum(converted['d'])*(100/(len(converted)-1))
            MSE =  1/(len(converted) - 1)* np.sum((converted['actual%'] - converted['predicted%'])**2)
            MAPE = 100/len(converted) * np.sum(abs((converted['actual_price'] - converted['predicted_price'])/converted['actual_price']))

            converted_dict[model] = converted.copy()
            stats_dict[model] = {'D%':D,'MSE(%)':MSE,'MAPE':MAPE}
        
        self.stats_dict = stats_dict
        self.converted_dict = converted_dict


    def create_graphs(self):
        models_colors = {'linear':'red'
                         ,'SVR(1)':'darkblue'
                         ,'SVR(2)':'blue'
                         ,'SVR(3)':'lightblue'
                         ,'random-forest(2)':'green'
                         ,'random-forest(1)':'lightgreen'}
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for model in models:
            df = self.converted_dict[model].copy()
            stats = self.stats_dict[model]
            D =np.round(stats["D%"],1)
            MAPE = np.round(stats["MAPE"],2)
        
            fig.add_trace(go.Scatter(y = df['predicted%']
                                        ,x = df.index
                                        ,name = f'{model}:(D {D}%, MAPE {MAPE}%)'
                                        ,marker_color = models_colors[model]
                                       ,line=dict(color=models_colors[model]
                                                  , width=2.5,dash='dash') #,dash='dash'
                                    ),secondary_y = False)        
        
        fig.add_trace(go.Scatter(y = df['actual%']
                                    ,x = df.index
                                    ,name = 'actual'
                                    ,line=dict(color='red', width=3)
                                ),secondary_y = False)
        
        fig.update_layout(title_text=F'Crypto:{self.crypto}, actual vs forecast 1-period ahead price % for {len(df)} cv sets'
                                ,width = 1500
                                ,height = 500
                                ,template = 'presentation'
                                ,showlegend = True
                                ,font = {'size':12}
                                , xaxis = {"title": '','showgrid':False}
                                , yaxis = {"title": '','tickformat':".2%"}
                                ,legend = {'orientation':"h",'yanchor':"top",'title':''}
                         )
    
        
        self.fig = fig