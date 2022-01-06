#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:23:13 2021

@author: charlestobin
"""

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

import shap

############################################################################################################################################

games_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/games.csv")

plays_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/plays.csv")

players_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/players.csv")

#seasons = ["2018", "2019", "2020"]
#tracking_df = pd.DataFrame()

#for s in seasons:
#    trackingTemp_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking"+s+".csv")
#    tracking_df = tracking_df.append(trackingTemp_df)

tracking_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking2020.csv")

tracking_df.loc[tracking_df['playDirection'] == "left", 'x'] = 120-tracking_df.loc[tracking_df['playDirection'] == "left", 'x']
tracking_df.loc[tracking_df['playDirection'] == "left", 'y'] = 160/3-tracking_df.loc[tracking_df['playDirection'] == "left", 'y']

#tracking_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking2020.csv")

pff_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/PFFScoutingData.csv")

############################################################################################################################################

jerseyMap_df = tracking_df.drop_duplicates(subset = ["gameId", "team", "jerseyNumber", "nflId"])

jerseyMap_df = pd.merge(jerseyMap_df, games_df, left_on=['gameId'], right_on =['gameId'])

conditions = [(jerseyMap_df['team'] == "home"), (jerseyMap_df['team'] != "home"),]

values = [jerseyMap_df['homeTeamAbbr'], jerseyMap_df['visitorTeamAbbr']]

jerseyMap_df['team'] = np.select(conditions, values)

jerseyMap_df['jerseyNumber'] = jerseyMap_df['jerseyNumber'].astype(str)

jerseyMap_df.loc[jerseyMap_df['jerseyNumber'].map(len) < 4, 'jerseyNumber'] = "0"+jerseyMap_df.loc[jerseyMap_df['jerseyNumber'].map(len) < 4, 'jerseyNumber'].str[:2]

jerseyMap_df['jerseyNumber'] = jerseyMap_df['jerseyNumber'].str[:2]

jerseyMap_df['teamJersey'] = jerseyMap_df['team'] + ' ' + jerseyMap_df['jerseyNumber'].str[:2]

jerseyMap_df = jerseyMap_df[['gameId', 'nflId', 'teamJersey']]

jerseyMap_df = jerseyMap_df.sort_values(['gameId', 'nflId', 'teamJersey'])

############################################################################################################################################

puntPlays_df = plays_df.copy()

puntPlays_df = pd.merge(puntPlays_df, pff_df, on = ['gameId', 'playId'])

puntPlays_df = puntPlays_df[(puntPlays_df['specialTeamsPlayType'] == 'Punt') & (puntPlays_df['specialTeamsResult'] == 'Return')]

puntPlays_df['returnerId'] = pd.to_numeric(puntPlays_df['returnerId'], errors='coerce')

#print(puntPlays_df.info())

puntPlays_df.columns

puntPlays_df = pd.merge(puntPlays_df, tracking_df, on = ['gameId', 'playId'])

puntPlays_df.columns

puntPlays_df = puntPlays_df[(puntPlays_df['event'] == 'punt_received')]

puntPlays_df = puntPlays_df.sort_values(['gameId', 'playId', 'frameId'])

ball_df = puntPlays_df[(puntPlays_df['displayName'] == 'football')]

ball_df.head(24)

############################################################################################################################################

puntPlaysPR_df = puntPlays_df.copy()

puntPlaysPR_df = puntPlays_df[puntPlays_df['returnerId'] == puntPlays_df['nflId']]

puntPlaysPR_df = puntPlaysPR_df[['gameId', 'playId', 'possessionTeam', 'returnerId',
                                 'x', 'y', 's', 'a', 'o', 'frameId', 'kickLength', 'kickReturnYardage']]

puntPlaysPR_df.rename(columns={"x": "ret_x", "y": "ret_y", "s": "ret_s", "a": "ret_a", "o": "ret_o"})

############################################################################################################################################

puntPlaysPT_df = puntPlays_df.copy()

puntPlaysPT_df = puntPlays_df[puntPlays_df['returnerId'] != puntPlays_df['nflId']]

puntPlaysPT_df = puntPlaysPT_df[['gameId', 'playId', 'possessionTeam', 'returnerId',
                                 'x', 'y', 's', 'a', 'o', 'team', 'frameId', 'nflId']]

puntPlaysPT_df.head(24)

puntPlaysPT_df = pd.merge(puntPlaysPT_df, games_df, on = 'gameId')

puntPlaysPT_df = puntPlaysPT_df[((puntPlaysPT_df['team']=='home') & (puntPlaysPT_df['possessionTeam'] == puntPlaysPT_df['homeTeamAbbr']) |
                                 (puntPlaysPT_df['team']=='away') & (puntPlaysPT_df['possessionTeam'] == puntPlaysPT_df['visitorTeamAbbr']))]

puntPlaysPT_df = puntPlaysPT_df[['gameId', 'playId', 'x', 'y', 's', 'a']]

############################################################################################################################################

mergePlays_df = pd.merge(puntPlaysPT_df, puntPlaysPR_df, on = ['gameId', 'playId'])

mergePlays_df = pd.merge(mergePlays_df, ball_df, on = ['gameId', 'playId'])

pd.set_option('display.max_columns', None)

mergePlays_df.head(50)

mergePlays_df['distance'] = np.sqrt((((mergePlays_df['x_y'] - mergePlays_df['x_x'])**2)
                                       + (mergePlays_df['y_x'] - mergePlays_df['y_y'])**2))

mergePlays_df['totalTime'] = mergePlays_df['snapTime'] + mergePlays_df['operationTime'] + mergePlays_df['hangTime']

mergePlays_df = mergePlays_df[['gameId', 'playId', 'x_x', 'y_x', 'x_y', 'y_y', 's_x', 'kickLength_x',
                               'kickReturnYardage_x', 'y', 'snapTime',
                               'operationTime', 'hangTime', 'totalTime', 'distance']]

mergePlays_df = mergePlays_df[mergePlays_df['kickReturnYardage_x'] < 50]
mergePlays_df = mergePlays_df[mergePlays_df['distance'] < 15]

#mergePlays_df.reset_index()

#mergePlays_df['distance'] = np.sqrt((((mergePlays_df['x_y'] - mergePlays_df['x_x'])**2)
#                                       + (mergePlays_df['y_x'] - mergePlays_df['y_y'])**2))

#mergePlays_df['totalTime'] = mergePlays_df['snapTime'] + mergePlays_df['operationTime'] + mergePlays_df['hangTime']

#mergePlays_df['distance'] = mergePlays_df.[mergePlays_df['distance'] < 20]['distance']

#mergePlays_df = mergePlays_df.drop(mergePlays_df[(mergePlays_df['distance']<20) & (mergePlays_df['kickReturnYardage_x']< 50)].index, inplace=True)

#mergePlays_df['kickReturnYardage_x'] = mergePlays_df[mergePlays_df['kickReturnYardage_x'] < 50]['kickReturnYardage_x']

mergePlays_df.describe()

############################################################################################################################################

mergePlays_df = mergePlays_df[mergePlays_df['kickReturnYardage_x'].notna()]
mergePlays_df = mergePlays_df[mergePlays_df['kickLength_x'].notna()]
mergePlays_df = mergePlays_df[mergePlays_df['distance'].notna()]
mergePlays_df = mergePlays_df[mergePlays_df['y'].notna()]
mergePlays_df = mergePlays_df[mergePlays_df['s_x'].notna()]
mergePlays_df = mergePlays_df[mergePlays_df['totalTime'].notna()]

#mergePlays_df = mergePlays_df.groupby(['gameId', 'playId'])

#type(mergePlays_df)

#mergePlays_df.groups

#mergePlays_df.head(50)

X = mergePlays_df[['kickLength_x', 'distance', 'y', 's_x', 'totalTime']]
y = mergePlays_df[['kickReturnYardage_x']]

seed = 1000
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#xgb_reg = xgb.XGBRegressor(max_depth=3, n_estimators=2000, n_jobs=20,
#                           objectvie='reg:squarederror', booster='gbtree',
#                           random_state=42, learning_rate=0.001)

xgb_reg = xgb.XGBRegressor()

xgb_reg = xgb_reg.fit(X_train, y_train)


explainer_xgb = shap.Explainer(xgb_reg)
shap_values_xgb = explainer_xgb(X_train)


model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, y_train.values.ravel())

shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

shap.summary_plot(shap_values, X_train)

shap.dependence_plot('kickLength_x', shap_values, X_train)

shap.dependence_plot('distance', shap_values, X_train)

shap.dependence_plot('y', shap_values, X_train)

shap.dependence_plot('s_x', shap_values, X_train)

shap.dependence_plot('totalTime', shap_values, X_train)

X_output = X_test.copy()
X_output.loc[:,'predict'] = np.round(model.predict(X_output),2)

y_pred = xgb_reg.predict(X_test)
y_pred
y_true = y_test

MSE = mse(y_true, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y_true, y_pred)

print("RMSE: ", np.round(RMSE, 2))

print("MSE: ", np.round(MSE, 2))

print("R-Squared: ", np.round(R_squared, 2))


_, ax = plt.subplots()

ax.scatter(x = range(0, y_true.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, y_pred.size), y=y_pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('Yards')
plt.legend()
plt.show()

model = xgb_reg

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

explainer = shap.Explainer(xgb_reg)
shap_values = explainer(X_train)

shap.plots.waterfall(shap_values[30])

############################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Imputer
from sklearn.model_selection import cross_val_score, cross_val_predict
import os



hyper_params = {'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': ['l1','l2'], 'learning_rate': 0.005,
                'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 10, 'verbose': 0, "max_depth": 8, "num_leaves": 128,  
                "max_bin": 512, "num_iterations": 100000}

gbm = lgb.LGBMRegressor(**hyper_params)

gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=1000)

y_predlgb = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

lgbmy_pred = y_predlgb
lgbmy_pred
lgbmy_true = y_test

lgbmMSE = mse(lgbmy_true, lgbmy_pred)
lgbmRMSE = np.sqrt(lgbmMSE)
lgbmR_squared = r2_score(lgbmy_true, lgbmy_pred)

print("RMSE: ", np.round(lgbmRMSE, 2))

print("MSE: ", np.round(lgbmMSE, 2))

print("R-Squared: ", np.round(lgbmR_squared, 2))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(xgb_reg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# Randomly pick some observations
random_picks = np.arange(1,330,50) # Every 50 rows
S = X_output.iloc[random_picks]
S

def shap_plot(j):
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(S)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

shap_plot(0)

############################################################################################################################################

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




