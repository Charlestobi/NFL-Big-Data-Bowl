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

seasons = ["2020"] #"2018", "2019",  
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

expectedReturnYards['operationTime'].describe()

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
                                           'OK', '<', '>', 'Precipitation', 'WindSpeed',
                                           'WindDirection', 'StadiumAzimuthAngle',
                                           'pressAltit']]

expectedReturnYards = expectedReturnYards.dropna()

#expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] < 63]
#expectedReturnYards = expectedReturnYards[expectedReturnYards['kickLength'] > 30]

expectedReturnYards.info()

X = expectedReturnYards[['x', 'y', 'operationTime', 'H', 'L', #'kickerId',
                         'OK', '<', '>', 'WindDirection', 'WindSpeed',
                         'StadiumAzimuthAngle',
                         'pressAltit']]

X = X.rename({'<': 'Left', '>': 'Right'}, axis=1)

y = expectedReturnYards[['kickLength']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

import shap

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

mean_train = np.mean(y_train)
mean_train

baseline_predictions = np.ones(y_test.shape) * 46.4
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

params = {'max_depth':6, 'min_child_weight': 1, 'eta':.3, 'subsample': 1,
          'colsample_bytree': 1, 'objective':'reg:squarederror',}
params['eval_metric'] = "mae"

num_boost_round = 999

model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], early_stopping_rounds=10)

cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=100, nfold=5, metrics={'mae'}, early_stopping_rounds=10)
cv_results

gridsearch_params = [(max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics={'mae'}, early_stopping_rounds=10)

    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['max_depth'] = 10
params['min_child_weight'] = 6

gridsearch_params = [(subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]]

min_mae = float("Inf")
best_params = None

for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
 
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics={'mae'}, early_stopping_rounds=10)

    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = .9
params['colsample_bytree'] = .7


min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
  
    params['eta'] = eta

    cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics=['mae'], early_stopping_rounds=10)

    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))

params['eta'] = .05

params

model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], early_stopping_rounds=10)

num_boost_round = model.best_iteration + 1
best_model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")])

mean_absolute_error(best_model.predict(dtest), y_test)

pred = best_model.predict(dtest)
pred
xgy_true = y_test

MSE = mse(xgy_true, pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(xgy_true, pred)

print("XGBoost RMSE: ", np.round(RMSE, 2))
print("XGBoost MSE: ", np.round(MSE, 2))
print("XGBoost R-Squared: ", np.round(R_squared, 2))

plt.scatter(xgy_true, pred)

import shap
model= XGBRegressor()
model.fit(X_train, y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

_, ax = plt.subplots()

ax.scatter(x = range(0, xgy_true.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, pred.size), y=pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values for XGBoost')
plt.xlabel('Observations')
plt.ylabel('Yards')
plt.legend()
plt.show()

#XGBoost RMSE:  6.31
#XGBoost MSE:  39.77
#XGBoost R-Squared:  0.35

total = xgb.DMatrix(X, label=y)

predictions= best_model.predict(total)

combine_df = X.copy()
combine_df['actualLength']= y
combine_df['prdictedLength']= predictions

combine_df.tail(25)

combine_df.to_csv('/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/epyoe.csv')

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

#Random Forest RMSE:  6.42
#Random Forest MSE:  41.17
#Random Forest R-Squared:  0.33

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
scores = cross_val_score(rf, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
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

#Random Forest RMSE:  6.35
#Random Forest MSE:  40.37
#Random Forest R-Squared:  0.34

cv = RepeatedKFold(n_splits=20, n_repeats=3, random_state=None)
# evaluate model
scores = cross_val_score(gbm, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

############################################################################################################################################



