#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:21:29 2021

@author: charlestobin
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn import preprocessing

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

pressAlt = weatherGroupBy.copy()

pressAlt['basePA'] = 29.92

pressAlt['pressAltit'] = ((pressAlt['basePA'] - pressAlt['Pressure'])*1000) + pressAlt['elevationFt']

densAlt = pressAlt.copy()

############################################################################################################################################

puntWx_df = pd.merge(pressAlt, puntPlays_df, on = ['gameId'])

############################################################################################################################################

expectedReturnYards = puntWx_df.copy()

expectedReturnYards

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
                                           'OK', '<', '>', 'Precipitation',
                                           'WindDirection', 'StadiumAzimuthAngle',
                                           'pressAltit']]

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
expectedReturnYards = expectedReturnYards[expectedReturnYards['Precipitation'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['StadiumAzimuthAngle'].notna()]
expectedReturnYards = expectedReturnYards[expectedReturnYards['pressAltit'].notna()]

expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] < 63]
expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] > 30]

expectedReturnYards.info()

X = expectedReturnYards[['x', 'y', 'operationTime', 'kickerId', 'H', 'L',
                         'OK', '<', '>', 'Precipitation', 'hangTime',
                         'StadiumAzimuthAngle',
                         'pressAltit']]

X = X.rename({'<': 'Left', '>': 'Right'}, axis=1)

y = expectedReturnYards[['kickLength']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

train = xgb.DMatrix(X_train,label=y_train)
test = xgb.DMatrix(X_test,label=y_test)

y_train['kickLength'].unique()
y_test['kickLength'].unique()


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)    


############################################################################################################################################

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

import shap

#seed = 1000
#test_size = 0.20
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

xgb_reg = xgb.XGBRegressor(max_depth=10, n_estimators=100, n_jobs=20,
                       objectvie='reg:squarederror', booster='gbtree',
                       random_state=42, learning_rate=0.05, subsample=0.75,
                       colsample_bytree=1, 
                       reg_alpha=120)


#xgb_reg = xgb.XGBRegressor()

xgb_reg.fit(X_train, y_train)
sco = xgb_reg.score(X_train, y_train)
sco
scores = cross_val_score(xgb_reg, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


explainer = shap.Explainer(xgb_reg)
shap_values = explainer(X_train)

shap.plots.waterfall(shap_values[200])

#shap.plots.force(shap_values[0])

#shap.plots.force(shap_values)
#shap.plots.force is slow for many thousands of rows, try subsampling your data.

#shap.plots.beeswarm(shap_values)


xgy_pred = xgb_reg.predict(X_test)
xgy_pred
xgy_true = y_test

xgMSE = mse(xgy_true, xgy_pred)
xgRMSE = np.sqrt(xgMSE)
xgR_squared = r2_score(xgy_true, xgy_pred)

print("XGBoost RMSE: ", np.round(xgRMSE, 2))
print("XGBoost MSE: ", np.round(xgMSE, 2))
print("XGBoost R-Squared: ", np.round(xgR_squared, 2))

_, ax = plt.subplots()

ax.scatter(x = range(0, xgy_true.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, xgy_pred.size), y=xgy_pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values for XGBoost')
plt.xlabel('Observations')
plt.ylabel('Yards')
plt.legend()
plt.show()

#XGBoost RMSE:  6.14
#XGBoost MSE:  37.7
#XGBoost R-Squared:  0.31

model = XGBRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=None)
# evaluate model
scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

############################################################################################################################################

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
rf.fit(X_train, y_train.values.ravel())  
print(rf.feature_importances_)
importances = rf.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

rf_shap_values = shap.KernelExplainer(rf.predict, X_test)

rfy_pred = rf.predict(X_test)
rfy_pred
rfy_true = y_test

rfMSE = mse(rfy_true, rfy_pred)
rfRMSE = np.sqrt(rfMSE)
rfR_squared = r2_score(rfy_true, rfy_pred)

print("Random Forest RMSE: ", np.round(rfRMSE, 2))
print("Random Forest MSE: ", np.round(rfMSE, 2))
print("Random Forest R-Squared: ", np.round(rfR_squared, 2))

#Random Forest RMSE:  6.41
#Random Forest MSE:  41.09
#Random Forest R-Squared:  0.27

_, ax = plt.subplots()

ax.scatter(x = range(0, rfy_true.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, rfy_pred.size), y=rfy_pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values for Random Forest')
plt.xlabel('Observations')
plt.ylabel('Yards')
plt.legend()
plt.show()

model = XGBRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=None)
# evaluate model
scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

############################################################################################################################################

import lightgbm as lgb
from sklearn.model_selection import train_test_split

hyper_params = {'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': ['l1','l2'], 'learning_rate': 0.005,
                'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 10, 'verbose': 0, "max_depth": 8, "num_leaves": 128,  
                "max_bin": 512, "num_iterations": 1000}

gbm = lgb.LGBMRegressor(**hyper_params)

gbm = gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=1000)

lgbmy_pred= gbm.predict(X_test, num_iteration=gbm.best_iteration_)
lgbmy_true = y_test

lgbmMSE = mse(lgbmy_true, lgbmy_pred)
lgbmRMSE = np.sqrt(lgbmMSE)
lgbmR_squared = r2_score(lgbmy_true, lgbmy_pred)

print("Random Forest RMSE: ", np.round(lgbmRMSE, 2))
print("Random Forest MSE: ", np.round(lgbmMSE, 2))
print("Random Forest R-Squared: ", np.round(lgbmR_squared, 2))

#Random Forest RMSE:  7.29
#Random Forest MSE:  53.14
#Random Forest R-Squared:  0.17

############################################################################################################################################



