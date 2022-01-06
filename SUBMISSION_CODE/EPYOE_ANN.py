#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:21:29 2021

@author: charlestobin
"""

import numpy as np
import pandas as pd
from numpy import absolute

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

import seaborn as sns

pd.set_option('max_columns', 25)

############################################################################################################################################

players_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/players.csv")

pff_data_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/PFFScoutingData.csv")

games_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/games.csv")

plays_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/plays.csv")

seasons = ["2018", "2019"] #"2020",
tracking_df = pd.DataFrame()

for s in seasons:
    trackingTemp_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking"+s+".csv")
    tracking_df = tracking_df.append(trackingTemp_df)

tracking_df.loc[tracking_df['playDirection'] == "left", 'x'] = 120-tracking_df.loc[tracking_df['playDirection'] == "left", 'x']
tracking_df.loc[tracking_df['playDirection'] == "left", 'y'] = 160/3-tracking_df.loc[tracking_df['playDirection'] == "left", 'y']

coord_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/stadium_coordinates.csv")
weather_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/games_weather.csv")
weatherGames_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/weather.csv")

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

weatherEffects_df = pd.merge(weather_df, weatherGames_df, on = ['game_id'])

weatherEffects_df = pd.merge(weatherEffects_df, coord_df, on = ['StadiumName'])

weatherGroupBy = weatherEffects_df.copy()

weatherGroupBy = weatherEffects_df.groupby(['game_id']).mean()

weatherGroupBy = weatherGroupBy.reset_index()

weatherGroupBy = weatherGroupBy.rename({'game_id': 'gameId'}, axis=1)

weatherGroupBy

weatherGroupBy = weatherGroupBy[['gameId', 'Temperature', 'Humidity', 'Precipitation', 'WindSpeed', 'WindDirection',
                                 'Pressure', 'Season', 'StadiumAzimuthAngle', 'elevationFt']]

############################################################################################################################################

pressAlt = weatherGroupBy.copy()

pressAlt['basePA'] = 29.92

pressAlt['pressAltit'] = ((pressAlt['basePA'] - pressAlt['Pressure'])*1000) + pressAlt['elevationFt']

densAlt = pressAlt.copy()

############################################################################################################################################

puntWx_df = pd.merge(pressAlt, puntPlays_df, on = ['gameId'])

############################################################################################################################################

expectedReturnYards = puntWx_df.copy()

expectedReturnYards = expectedReturnYards[['gameId', 'playId', 'Temperature', 'Humidity', 'Precipitation',
                                           'WindSpeed', 'WindDirection', 'StadiumAzimuthAngle',
                                           'pressAltit',
                                           'possessionTeam', 'specialTeamsPlayType', 'specialTeamsResult',
                                           'kickerId', 'returnerId', 'yardlineSide',
                                           'yardlineNumber', 'snapDetail',
                                           'kickLength', 'kickReturnYardage',
                                           'operationTime',
                                           'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual',
                                           'returnDirectionIntended', 'gunners',
                                           'vises', 'kickContactType']]

expectedReturnYards = pd.merge(expectedReturnYards, tracking_df, on = ['gameId', 'playId'])

expectedReturnYards = expectedReturnYards[(expectedReturnYards['event'] == 'ball_snap')]

expectedReturnYards = expectedReturnYards[(expectedReturnYards['displayName'] == 'football')]

dummies = pd.get_dummies(expectedReturnYards.snapDetail)

expectedReturnYards = pd.concat([expectedReturnYards, dummies], axis=1)

expectedReturnYards = expectedReturnYards[['gameId', 'playId', 'kickerId', 'yardlineNumber', 'kickLength',
                                           'hangTime', 'operationTime', 'x', 'y', 'H', 'L',
                                           'OK', '<', '>', 'Precipitation',
                                           'WindDirection', 'StadiumAzimuthAngle',
                                           'pressAltit']]

expectedReturnYards = expectedReturnYards.dropna()

expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] < 63]
expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] > 30]

############################################################################################################################################

X = expectedReturnYards[['x', 'y', 'kickerId', 'Precipitation',
                         'StadiumAzimuthAngle',
                         'pressAltit', 'WindDirection']]
y = expectedReturnYards[['kickLength']]


PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y) 

X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)    

############################################################################################################################################

model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5, input_dim=7, kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
 
#model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

#model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
 
# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 10, epochs = 100, verbose=10)    

############################################################################################################################################

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
 

 
############################################################################################################################################
# Calling the function
ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)

ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')
    
model.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose=0)
 
# Generating Predictions on testing data
Predictions=model.predict(X_test)
 
# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)
 
TestingData=pd.DataFrame(data=Test_Data, columns=['x', 'y', 'kickerId', 'Precipitation',
                                                  'StadiumAzimuthAngle',
                                                  'pressAltit', 'WindDirection'])
TestingData['kickLength']=y_test_orig
TestingData['PredictedLength']=Predictions
TestingData.head()

APE=100*(abs(TestingData['kickLength']-TestingData['PredictedLength'])/TestingData['kickLength'])
TestingData['APE']=APE

print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.tail(25)

TestingData.describe()

TestingData.to_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/annExpectedPuntYards.csv")

