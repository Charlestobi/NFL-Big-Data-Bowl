#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:21:29 2021

@author: charlestobin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

pd.set_option('max_columns', 100)

############################################################################################################################################

players_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/players.csv")

pff_data_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/PFFScoutingData.csv")

games_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/games.csv")

plays_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/plays.csv")

seasons = ["2018", "2019", "2020"] #"2018", 
tracking_df = pd.DataFrame()

for s in seasons:
    trackingTemp_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking"+s+".csv")
    tracking_df = tracking_df.append(trackingTemp_df)


tracking_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking2020.csv")

tracking_df.loc[tracking_df['playDirection'] == "left", 'x'] = 120-tracking_df.loc[tracking_df['playDirection'] == "left", 'x']
tracking_df.loc[tracking_df['playDirection'] == "left", 'y'] = 160/3-tracking_df.loc[tracking_df['playDirection'] == "left", 'y']

############################################################################################################################################

puntPlays_df = plays_df.copy()

puntPlays_df = pd.merge(puntPlays_df, pff_data_df, on = ['gameId', 'playId'])

puntPlays_df = puntPlays_df[(puntPlays_df['specialTeamsPlayType'] == 'Punt')]

puntPlays_df['returnerId'] = pd.to_numeric(puntPlays_df['returnerId'], errors='coerce')

puntPlays_df = puntPlays_df[['gameId', 'playId', 'playDescription',
                            'possessionTeam', 'specialTeamsPlayType', 'specialTeamsResult',
                            'kickerId', 'returnerId', 'yardlineSide',
                            'yardlineNumber',
                            'kickLength', 'kickReturnYardage', 'playResult',
                            'absoluteYardlineNumber', 'snapDetail', 'snapTime', 'operationTime',
                            'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual',
                            'returnDirectionIntended', 'returnDirectionActual', 'missedTackler',
                            'assistTackler', 'tackler', 'gunners',
                            'puntRushers', 'specialTeamsSafeties', 'vises', 'kickContactType']]

puntPlays_df.columns

puntPlays_df.head()

############################################################################################################################################

expectedReturnYards = puntPlays_df.copy()

expectedReturnYards

expectedReturnYards = expectedReturnYards[['gameId', 'playId',
                            'possessionTeam', 'specialTeamsPlayType', 'specialTeamsResult',
                            'kickerId', 'returnerId', 'yardlineSide',
                            'yardlineNumber', 'snapDetail',
                            'kickLength', 'kickReturnYardage',
                            'operationTime',
                            'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual',
                            'returnDirectionIntended', 'gunners',
                            'vises', 'kickContactType']]

expectedReturnYards

expectedReturnYards = pd.merge(expectedReturnYards, tracking_df, on = ['gameId', 'playId'])

#expectedReturnYards = expectedReturnYards[(expectedReturnYards['event'] == 'punt_received') | (expectedReturnYards['event'] == 'punt_downed') #| (pp_df['event'] == 'punt_land')
#              | (expectedReturnYards['event'] == 'fair_catch') | (expectedReturnYards['event'] == 'touchback') | (expectedReturnYards['event'] == 'out_of_bounds')]

expectedReturnYards = expectedReturnYards[(expectedReturnYards['event'] == 'ball_snap')]

expectedReturnYards = expectedReturnYards[(expectedReturnYards['displayName'] == 'football')]


dummies = pd.get_dummies(expectedReturnYards.snapDetail)

expectedReturnYards = pd.concat([expectedReturnYards, dummies], axis=1)

expectedReturnYards

expectedReturnYards = expectedReturnYards[['gameId', 'playId', 'kickerId', 'yardlineNumber', 'kickLength',
                                           'hangTime', 'operationTime', 'x', 'y', 'H', 'L',
                                           'OK', '<', '>']]

expectedReturnYards = expectedReturnYards[expectedReturnYards['yardlineNumber'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['hangTime'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['operationTime'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['x'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['y'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['H'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['L'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['OK'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['<'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['>'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['kickerId'].notna()]

X = expectedReturnYards[['x', 'y', 'operationTime', 'hangTime', 'kickerId', 'H', 'L',
                         'OK', '<', '>']]
y = expectedReturnYards[['kickLength']]

from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y) 

X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)    

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=10, kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
 
#model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

#model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
 
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 5, epochs = 100, verbose=1)    



def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[5, 10, 15, 20]
    epoch_list  =   [5, 10, 50, 100]
    
    import pandas as pd
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
 
            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
            
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
 
            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))
 
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
 
            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
            
            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)
 

 
######################################################
# Calling the function
ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)

ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')
    
model.fit(X_train, y_train, batch_size = 5, epochs = 100, verbose=0)
 
# Generating Predictions on testing data
Predictions=model.predict(X_train)
 
# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_train)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_train)
 
TestingData=pd.DataFrame(data=Test_Data, columns=['x', 'hangTime', 'operationTime', 'hangTime', 'kickerId', 'H', 'L',
                                                  'OK', '<', '>'])
TestingData['kickLength']=y_test_orig
TestingData['PredictedLength']=Predictions
TestingData.head()

APE=100*(abs(TestingData['kickLength']-TestingData['PredictedLength'])/TestingData['kickLength'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.tail(250)




