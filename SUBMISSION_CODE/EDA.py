#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:18:59 2021

@author: charlestobin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

pd.set_option('max_columns', 100)
scaler = MinMaxScaler()
############################################################################################################################################

players_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/players.csv")

pff_data_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/PFFScoutingData.csv")

games_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/games.csv")

plays_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/plays.csv")

tracking_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/tracking2020.csv")

tracking_df.loc[tracking_df['playDirection'] == "left", 'x'] = 120-tracking_df.loc[tracking_df['playDirection'] == "left", 'x']
tracking_df.loc[tracking_df['playDirection'] == "left", 'y'] = 160/3-tracking_df.loc[tracking_df['playDirection'] == "left", 'y']

epyoe_df = pd.read_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/epyoe.csv")

############################################################################################################################################

pt_dist = plays_df['specialTeamsPlayType'].value_counts().reset_index()
pt_dist.columns = ['specialTeamsPlayType', 'frequency']
pt_dist.head()

sorted_pt_dist = pt_dist.sort_values('specialTeamsPlayType').set_index('specialTeamsPlayType')
sorted_pt_dist.head()

sorted_pt_dist.plot(kind='bar', figsize=(20,4))
plt.title('Special Teams Play Type')

def find_dist(df, col_name):
    
    dist = df[col_name].value_counts().reset_index()
    dist.columns = [col_name, 'frequency']
    sorted_dist = dist.sort_values(col_name, ascending=True).set_index(col_name)
    sorted_dist.plot(kind='bar', figsize=(20,4))
    
    return True

find_dist(pff_data_df, 'snapDetail')
plt.title('Snap Placement')

clean_punt_df = pff_data_df[pff_data_df.snapDetail.isin(["<", ">", "H", "L"])]

snap_dist = pff_data_df['snapDetail'].value_counts().reset_index()
snap_dist.columns = ['snapDetail', 'frequency']
snap_dist.head()

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
jerseyMap_df.head()

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

############################################################################################################################################

gunners_df = puntPlays_df.copy()
gunners_df[['gunnerJersey1', 'gunnerJersey2', 
          'gunnerJersey3', 'gunnerJersey4', 
          ]] = gunners_df['gunners'].str.split('; ',expand=True)

gunners_df = gunners_df[['gameId', 'playId', 'gunnerJersey1', 'gunnerJersey2', 
          'gunnerJersey3', 'gunnerJersey4']]

gunners_df = pd.melt(gunners_df, id_vars =['gameId', 'playId'], 
                             value_vars =['gunnerJersey1', 'gunnerJersey2', 'gunnerJersey3', 'gunnerJersey4'],
                             value_name = 'teamJersey')

gunners_df = pd.merge(gunners_df, jerseyMap_df, on = ['gameId', 'teamJersey'])
gunners_df = gunners_df[['gameId', 'playId', 'nflId']]
gunners_df = pd.merge(gunners_df, players_df, on = ['nflId'])
gunners_df = gunners_df[['gameId', 'playId', 'nflId','displayName']]
gunners_df = gunners_df.sort_values(['gameId', 'playId', 'nflId'])
gunners_df = gunners_df.rename(columns={"displayName": "gunners"})
gunners_df = gunners_df.groupby(['gameId', 'playId'])['gunners'].apply('; '.join).reset_index()
gunners_df[['gunner1', 'gunner2','gunner3', 'gunner4']] = gunners_df['gunners'].str.split('; ',expand=True)
gunners_df = gunners_df[['gameId', 'playId', 'gunner1', 'gunner2', 'gunner3', 'gunner4']]
gunners_df.head()

############################################################################################################################################

tackler_df = puntPlays_df.copy()
tackler_df = tackler_df[['gameId', 'playId', 'tackler']]
tackler_df = pd.melt(tackler_df, id_vars =['gameId', 'playId'], 
                             value_vars =['tackler'],
                             value_name = 'teamJersey')

tackler_df = pd.merge(tackler_df, jerseyMap_df, on = ['gameId', 'teamJersey'])
tackler_df = tackler_df[['gameId', 'playId', 'nflId']]
tackler_df = tackler_df.sort_values(['gameId', 'playId', 'nflId'])
tackler_df = pd.merge(tackler_df, players_df, on = ['nflId'])
tackler_df = tackler_df[['gameId', 'playId', 'displayName']]
tackler_df = tackler_df.rename(columns={"displayName": "tackler"})
tackler_df

############################################################################################################################################

assTackler_df = puntPlays_df.copy()
assTackler_df = assTackler_df[['gameId', 'playId', 'assistTackler']]
assTackler_df = pd.melt(assTackler_df, id_vars =['gameId', 'playId'], 
                             value_vars =['assistTackler'],
                             value_name = 'teamJersey')

assTackler_df = pd.merge(assTackler_df, jerseyMap_df, on = ['gameId', 'teamJersey'])
assTackler_df = assTackler_df.sort_values(['gameId', 'playId', 'nflId'])
assTackler_df = assTackler_df[['gameId', 'playId', 'nflId']]
assTackler_df = pd.merge(assTackler_df, players_df, on = ['nflId'])
assTackler_df = assTackler_df[['gameId', 'playId', 'displayName']]
assTackler_df = assTackler_df.rename(columns={"displayName": "assTackler"})
assTackler_df.head()

############################################################################################################################################

numTackle_df = pd.merge(tackler_df, gunners_df, on = ['gameId', 'playId'])
numTackle_df = numTackle_df[(numTackle_df['tackler'] == numTackle_df['gunner1']) | (numTackle_df['tackler'] == numTackle_df['gunner2']) | 
                            (numTackle_df['tackler'] == numTackle_df['gunner3']) | (numTackle_df['tackler'] == numTackle_df['gunner4'])]

numTackle_df = numTackle_df.tackler.value_counts().reset_index().rename(columns={'index': 'displayName', 0: 'tackles'})
numTackle_df = numTackle_df[(numTackle_df['tackler'] > 2)]
find_dist(numTackle_df, 'tackler')
numTackle_df.plot.barh(x='displayName', y='tackler')

############################################################################################################################################

numAssTackle_df = pd.merge(assTackler_df, gunners_df, on = ['gameId', 'playId'])
numAssTackle_df = numAssTackle_df[(numAssTackle_df['assTackler'] == numAssTackle_df['gunner1']) | (numAssTackle_df['assTackler'] == numAssTackle_df['gunner2']) | 
                            (numAssTackle_df['assTackler'] == numAssTackle_df['gunner3']) | (numAssTackle_df['assTackler'] == numAssTackle_df['gunner4'])]

numAssTackle_df = numAssTackle_df.assTackler.value_counts().reset_index().rename(columns={'index': 'displayName', 0: 'assTackles'})

############################################################################################################################################

gunnerTracking_df = tracking_df.copy()
gunnerTracking_df = pd.merge(gunnerTracking_df, games_df, on = 'gameId')
gunnerTracking_df = pd.merge(gunnerTracking_df, plays_df, on = ['gameId', 'playId'])
gunnerTracking_df = gunnerTracking_df[(gunnerTracking_df['specialTeamsPlayType'] == 'Punt')]
gunnerTracking_df = pd.merge(gunnerTracking_df, gunners_df, on = ['gameId', 'playId'])
gunnerTracking_df = gunnerTracking_df[(gunnerTracking_df['displayName'] == gunnerTracking_df['gunner1']) | (gunnerTracking_df['displayName'] == gunnerTracking_df['gunner2']) |
                                      (gunnerTracking_df['displayName'] == gunnerTracking_df['gunner3']) | (gunnerTracking_df['displayName'] == gunnerTracking_df['gunner4'])]

gunnerTracking_df = gunnerTracking_df[(gunnerTracking_df['event'] == 'punt_received') | (gunnerTracking_df['event'] == 'punt_downed') | (gunnerTracking_df['event'] == 'punt_land')]
gunnerTracking_df

############################################################################################################################################

pp_df = plays_df.copy()
pp_df = pd.merge(pp_df, pff_data_df, on = ['gameId', 'playId'])
pp_df = pp_df[(pp_df['specialTeamsPlayType'] == 'Punt')]
pp_df['returnerId'] = pd.to_numeric(pp_df['returnerId'], errors='coerce')
pp_df = pd.merge(pp_df, tracking_df, on = ['gameId', 'playId'])

pp_df = pp_df[(pp_df['event'] == 'punt_received') | (pp_df['event'] == 'punt_downed') #| (pp_df['event'] == 'punt_land')
              | (pp_df['event'] == 'fair_catch') | (pp_df['event'] == 'touchback') | (pp_df['event'] == 'out_of_bounds')]
pp_df = pp_df.sort_values(['gameId', 'playId', 'frameId'])

############################################################################################################################################

ball = pp_df.copy()
ball = ball[(ball['displayName'] == 'football')]
ball = ball[['gameId', 'playId', 'x', 'y', 'event']]
ball = ball.rename(columns={"x": "ball_x", "y": "ball_y"})
ball

############################################################################################################################################

puntPlaysPR_df = pp_df.copy()
puntPlaysPR_df = puntPlaysPR_df[puntPlaysPR_df['returnerId'] == puntPlaysPR_df['nflId']]
puntPlaysPR_df = puntPlaysPR_df[['gameId', 'playId', 'possessionTeam', 'returnerId',
                                 'x', 'y', 's', 'a', 'o', 'frameId', 'kickLength', 'kickReturnYardage']]

puntPlaysPR_df = puntPlaysPR_df.rename(columns={"x": "ret_x", "y": "ret_y", "s": "ret_s", "a": "ret_a", "o": "ret_o"})

############################################################################################################################################

dist_df = pd.merge(puntPlaysPR_df, gunnerTracking_df, on = ['gameId', 'playId'])

dist_df['distance'] = np.sqrt((((dist_df['x'] - dist_df['ret_x'])**2)
                                + (dist_df['y'] - dist_df['ret_y'])**2))

dist_df = dist_df.groupby(['displayName']).mean()
dist_df = dist_df[['distance']]
dist_df.head(67)

############################################################################################################################################

distb_df = pd.merge(ball, gunnerTracking_df, on = ['gameId', 'playId'])

distb_df['distanceB'] = np.sqrt((((distb_df['x'] - distb_df['ball_x'])**2)
                                + (distb_df['y'] - distb_df['ball_y'])**2))

distb_df = distb_df.groupby(['displayName']).mean()
distb_df = distb_df[['distanceB']]
distb_df.head(67)

############################################################################################################################################

speed_df = pd.merge(puntPlaysPR_df, gunnerTracking_df, on = ['gameId', 'playId'])
speed_df = speed_df.groupby(['displayName']).mean()
speed_df = speed_df[['s']]
speed_df.head(67)

############################################################################################################################################

acc_df = pd.merge(puntPlaysPR_df, gunnerTracking_df, on = ['gameId', 'playId'])
acc_df.columns
acc_df = acc_df.groupby(['displayName']).mean()
acc_df = acc_df[['a']]
acc_df.head(67)

############################################################################################################################################

merged_df = pd.merge(left=numTackle_df, right=numAssTackle_df, how='left', left_on='displayName', right_on='displayName')
merged_df = pd.merge(merged_df, dist_df, on = 'displayName')
merged_df = pd.merge(merged_df, speed_df, on = 'displayName')
merged_df = pd.merge(merged_df, acc_df, on = 'displayName')
merged_df = pd.merge(merged_df, distb_df, on = 'displayName')
merged_df['assTackler'] = merged_df['assTackler'].fillna(0)
merged_df

############################################################################################################################################

ls_df = pp_df.copy()
ls_df.columns
ls_df = ls_df[ls_df['position'] == 'LS']
ls_df

############################################################################################################################################

def football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=55,
                          highlight_first_down_line=False,
                          yards_to_go=10,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
 
    rect = patches.Rectangle((10, 0), 100, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width()/2.0
    cy = ry + rect.get_height()/2.0
    ax.annotate("NFL", (cx, cy), color='black', weight='bold', alpha=0.8, fontsize=20, ha='center', va='center')
    
    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white', alpha=0.6)
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='g',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='g',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 12, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20, fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 12, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20, fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white', alpha=0.6)
        ax.plot([x, x], [22.91, 23.57], color='white', alpha=0.6)
        ax.plot([x, x], [29.73, 30.39], color='white', alpha=0.6)

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
        
    if highlight_first_down_line:
        fl = hl + yards_to_go
        plt.plot([fl, fl], [0, 53.3], color='yellow')
        plt.text(fl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax
############################################################################################################################################

ballpp_df = pp_df.copy()

ballpp_df = ballpp_df[(ballpp_df['displayName'] == 'football')]

ballpp_df = ballpp_df[['gameId', 'playId', 'event', 'kickerId', 'x', 'y', 'kickLength']]

ballpp_df = ballpp_df.rename(columns={"x": "ball_x", "y": "ball_y"})

ballpp_df['event'].unique()

ballpp_df = ballpp_df.rename(columns={'kickerId': 'nflId'})

ballpp_df = pd.merge(ballpp_df, players_df, on= 'nflId')

ballpp_df = ballpp_df[['gameId', 'playId', 'event', 'ball_x', 'ball_y', 'nflId', 'displayName', 'kickLength']]

#ballpp_df = ballpp_df.drop_duplicates(subset=['gameId', 'playId'], keep='first')

ballp_df = ballpp_df.copy()

ballpp_df['ball_y'] = ballpp_df['ball_y'] * -1

dup = ballp_df.copy()
dup.drop_duplicates(subset=['gameId', 'playId'], keep='first')


dup = ballpp_df[ballpp_df.duplicated(['gameId','playId'])]
dup.head(50)

############################################################################################################################################

def plot_length(punter_names = 'Tommy Townsend'):

  plt.style.use('fivethirtyeight')

  player = ballpp_df.loc[(ballpp_df['displayName'] == punter_names)]


  puntReceived = player.loc[player['event'] == 'punt_received']
  puntDowned = player.loc[player['event'] == 'punt_downed']
  fairCatch = player.loc[player['event'] == 'fair_catch']
  touchback = player.loc[player['event'] == 'touchback']
  outOfBounds = player.loc[player['event'] == 'out_of_bounds']

  fig, ax = plt.subplots(figsize=(10, 6))

  ax.scatter(puntReceived['ball_y'], puntReceived['kickLength'], color='red', alpha=0.6, edgecolors='white', zorder=2)
  ax.scatter(puntDowned['ball_y'], puntDowned['kickLength'], color='blue', alpha=0.8, edgecolors='white', zorder=3)
  ax.scatter(fairCatch['ball_y'], fairCatch['kickLength'], color='yellow', alpha=0.8, edgecolors='white', zorder=4)
  
  ax.scatter(touchback['ball_y'], touchback['kickLength'], color='purple', alpha=1, edgecolors='white', zorder=5)
  
  ax.scatter(outOfBounds['ball_y'], outOfBounds['kickLength'], color='black', alpha=0.8, edgecolors='white', zorder=6)

      
  ax.vlines(-12, 0, 70, color='white', alpha=0.5, lw=4, linestyles='dashed')
  ax.set_ylim(0, 70)

  ax.vlines(-41.3, 0, 70, color='white', alpha=0.5, lw=4, linestyles='dashed')
  ax.set_ylim(0, 70)
  
  ax.vlines(0, 0, 70, color='white', alpha=0.5, lw=4, linestyles='solid')
  ax.set_ylim(0, 70)

  ax.vlines(-53.3, 0, 70, color='white', alpha=0.5, lw=4, linestyles='solid')
  ax.set_ylim(0, 70)

  positions = (-6, -26.65, -47.3)
  labels = ("Outside Numbers R", "Inside Numbers", "Outside Numbers L")
  plt.xticks(positions, labels)

  #ax.set_yticks(np.arange(20, 80, 10));
  #ax.set_xticks(np.linspace(-53.3, 0, 7))

  ax.tick_params(axis='x', colors="black", grid_alpha=0.4)
  ax.tick_params(axis='y', colors="black", grid_alpha=0.4)
  plt.setp(ax.spines.values(), color='white', alpha=0.2, linewidth=1)

  ax.set_title('\n'+ ' Punt Location vs Distance ' + '\n', color='black', alpha=1, fontsize=18) #punter_names + 
  ax.set(xlabel="Short Field Position", ylabel = "Kick Length")

  ax.set_facecolor('green')
  fig.set_facecolor('white')

#233746

  ax.legend(['Punt Received', 'Punt Downed', 'Fair Catch', 'Touchback', 'Out of Bounds'],
            ncol=1, loc='upper left', bbox_to_anchor=(1.05, 1), facecolor='grey', prop={'size': 16})

plot_length()

plot_length(punter_names = 'Tommy Townsend')

############################################################################################################################################

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Punt Received', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Punt Downed', markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Fair Catch', markerfacecolor='yellow', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Punt Received', markerfacecolor='purple', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Punt Received', markerfacecolor='black', markersize=10)]





def plot_punts(punter_names = 'Tommy Townsend'):

  #plt.style.use('fivethirtyeight')

  player = ballp_df.loc[(ballp_df['displayName'] == punter_names)] #

  #grab completions, incompletions, interceptions, and tds separately
  puntReceived = player.loc[player['event'] == 'punt_received']
  #puntLand = player.loc[player['event'] == 'punt_land']
  puntDowned = player.loc[player['event'] == 'punt_downed']
  fairCatch = player.loc[player['event'] == 'fair_catch']
  touchback = player.loc[player['event'] == 'touchback']
  outOfBounds = player.loc[player['event'] == 'out_of_bounds']

  fig, ax = football_field()

  ax.scatter(puntReceived['ball_x'], puntReceived['ball_y'], color='red', alpha=0.8, edgecolors='white', zorder=2)

  ax.scatter(puntDowned['ball_x'], puntDowned['ball_y'], color='blue', alpha=0.8, edgecolors='white', zorder=3)

  ax.scatter(fairCatch['ball_x'], fairCatch['ball_y'], color='yellow', alpha=0.8, edgecolors='white', zorder=4)

  ax.scatter(touchback['ball_x'], touchback['ball_y'], color='purple', alpha=0.8, edgecolors='white', zorder=5)
  
  ax.scatter(outOfBounds['ball_x'], outOfBounds['ball_y'], color='black', alpha=1, edgecolors='black', zorder=6)

  ax.set_title('\n'+ punter_names + ' Punt Locations ' + '\n', color='black', alpha=1, fontsize=18) #punter_names + 
  
  #ax.legend(handles=legend_elements, loc='upper left')

  #ax.legend(['pr', 'pd', 'fc', 'tb', 'oob', 'a', 'b', 'c'], #['Punt Received', 'Punt Downed', 'Fair Catch', 'Out of Bounds']
            #ncol=1, loc= 'upper left', bbox_to_anchor=(1.05, 1), facecolor='grey', prop={'size': 16})
  fig.set_facecolor('white')
plot_punts()

plot_punts(punter_names = 'Tommy Townsend')

############################################################################################################################################

stats_df = pd.merge(pff_data_df, plays_df, on = ['gameId', 'playId'])

stats_df = stats_df[stats_df['specialTeamsPlayType'] == 'Punt']

stats_df = stats_df[['gameId', 'playId', 'operationTime', 'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual',
                     'specialTeamsResult', 'kickerId', 'kickLength', 'yardlineNumber', 'absoluteYardlineNumber', 'specialTeamsPlayType']]

stats_df = stats_df.rename(columns={'kickerId': 'nflId'})

stats_df.head(400)

stats_df = pd.merge(players_df, stats_df, on = 'nflId')

stats_df = pd.merge(stats_df, games_df, on = 'gameId')

stats_df

############################################################################################################################################

stats20 = stats_df[stats_df['season'] == 2020]

stats20

basicStats20 = stats20[['displayName', 'weight', 'height']]

basicStats20 = basicStats20.groupby(['displayName', 'height']).mean()

basicStats20

stats20['yardlineNumber'].unique()

stats20_15 = stats20[stats20['yardlineNumber'].between(0, 15)]
stats20_15 = stats20_15[['displayName', 'operationTime', 'hangTime', 'kickLength']]
stats20_15 = stats20_15.rename(columns={'operationTime': 'pinnedOT', 'hangTime': 'pinnedHT', 'kickLength': 'pinnedKL'})
stats20_15 = stats20_15.groupby('displayName').mean()
stats20_15

stats20_40 = stats20[stats20['yardlineNumber'].between(16, 40)]
stats20_40 = stats20_40[['displayName', 'operationTime', 'hangTime', 'kickLength']]
stats20_40 = stats20_40.groupby('displayName').mean()
stats20_40

stats20_50 = stats20[stats20['yardlineNumber'].between(41, 50)]
stats20_50 = stats20_50[['displayName', 'operationTime', 'hangTime', 'kickLength']]
stats20_50 = stats20_50.rename(columns={'operationTime': 'pinOT', 'hangTime': 'pinHT', 'kickLength': 'pinKL'})
stats20_50 = stats20_50.groupby('displayName').mean()
stats20_50

totalPunt20 = ballp_df[['displayName', 'playId']]
totalPunt20 = totalPunt20.rename(columns={'playId': 'totalPunts'})
totalPunt20 = totalPunt20.groupby('displayName').count()
totalPunt20

puntLeft20 = ballp_df[ballp_df['ball_y'] >= 41.3]
puntLeft20 = puntLeft20[['displayName', 'playId']]
puntLeft20 = puntLeft20.rename(columns={'playId': 'puntLeft'})
puntLeft20 = puntLeft20.groupby('displayName').count()
puntLeft20

puntRight20 = ballp_df[ballp_df['ball_y'] <= 12]
puntRight20 = puntRight20[['displayName', 'playId']]
puntRight20 = puntRight20.rename(columns={'playId': 'puntRight'})
puntRight20 = puntRight20.groupby('displayName').count()
puntRight20

puntInsideNums20 = ballp_df[ballp_df['ball_y'].between(12.1, 41.2)]
puntInsideNums20 = puntInsideNums20[['displayName', 'playId']]
puntInsideNums20 = puntInsideNums20.rename(columns={'playId': 'puntCenter'})
puntInsideNums20 = puntInsideNums20.groupby('displayName').count()
puntInsideNums20

puntsReturnedTotal = ballp_df.copy()
puntsReturnedTotal = puntsReturnedTotal[puntsReturnedTotal['event'] == 'punt_received']
puntsReturnedTotal = puntsReturnedTotal[['displayName', 'playId']]
puntsReturnedTotal = puntsReturnedTotal.rename(columns={'playId': 'returnedTotal'})
puntsReturnedTotal = puntsReturnedTotal.groupby('displayName').count()
puntsReturnedTotal.reset_index()
puntsReturnedTotal

puntInNumsRet = ballp_df.copy()
puntInNumsRet = puntInNumsRet[puntInNumsRet['event'] == 'punt_received']
puntInNumsRet = puntInNumsRet[puntInNumsRet['ball_y'].between(12.1, 41.2)]
puntInNumsRet = puntInNumsRet[['displayName', 'playId']]
puntInNumsRet = puntInNumsRet.rename(columns={'playId': 'returnedInsideNumsTotal'})
puntInNumsRet = puntInNumsRet.groupby('displayName').count()
puntInNumsRet.reset_index()
puntInNumsRet

puntOutNumsRet = pd.merge(puntInNumsRet, puntsReturnedTotal, on= ['displayName'])
puntOutNumsRet['returnedOutsideNumsTotal'] = puntOutNumsRet['returnedTotal'] - puntOutNumsRet['returnedInsideNumsTotal']
puntOutNumsRet.reset_index()
puntOutNumsRet

puntPerRetInOut = puntOutNumsRet.copy()
puntPerRetInOut['perInRet'] = puntPerRetInOut['returnedInsideNumsTotal'] / puntPerRetInOut['returnedTotal']
puntPerRetInOut['perOutRet'] = puntPerRetInOut['returnedOutsideNumsTotal'] / puntPerRetInOut['returnedTotal']
puntPerRetInOut.describe()
puntPerRetInOut


puntWI520 = ballp_df[ballp_df['ball_x'].between(105, 109.9)]
puntWI520 = puntWI520[(puntWI520['event'] == 'punt_received') | (puntWI520['event'] == 'punt_downed') 
                  | (puntWI520['event'] == 'fair_catch') |(puntWI520['event'] == 'out_of_bounds')]
puntWI520 = puntWI520[['displayName', 'playId']]
puntWI520 = puntWI520.rename(columns={'playId': 'puntWI5'})
puntWI520 = puntWI520.groupby('displayName').count()


puntWI1020 = ballp_df[ballp_df['ball_x'].between(100, 104.9)]
puntWI1020 = puntWI1020[(puntWI1020['event'] == 'punt_received') | (puntWI1020['event'] == 'punt_downed') 
                  | (puntWI1020['event'] == 'fair_catch') |(puntWI1020['event'] == 'out_of_bounds')]
puntWI1020 = puntWI1020[['displayName', 'playId']]
puntWI1020 = puntWI1020.rename(columns={'playId': 'puntWI10'})
puntWI1020 = puntWI1020.groupby('displayName').count()
puntWI1020

puntWI20020 = ballp_df[ballp_df['ball_x'].between(90, 99.9)]
puntWI20020 = puntWI20020[(puntWI20020['event'] == 'punt_received') | (puntWI20020['event'] == 'punt_downed') 
                  | (puntWI20020['event'] == 'fair_catch') |(puntWI20020['event'] == 'out_of_bounds')]
puntWI20020 = puntWI20020[['displayName', 'playId']]
puntWI20020 = puntWI20020.rename(columns={'playId': 'puntWI20'})
puntWI20020 = puntWI20020.groupby('displayName').count()
puntWI20020

puntWI30020 = ballp_df[ballp_df['ball_x'].between(80, 89.9)]
puntWI30020 = puntWI30020[(puntWI30020['event'] == 'punt_received') | (puntWI30020['event'] == 'punt_downed') 
                  | (puntWI30020['event'] == 'fair_catch') |(puntWI30020['event'] == 'out_of_bounds')]
puntWI30020 = puntWI30020[['displayName', 'playId']]
puntWI30020 = puntWI30020.rename(columns={'playId': 'puntWI30'})
puntWI30020 = puntWI30020.groupby('displayName').count()
puntWI30020

puntWI40020 = ballp_df[ballp_df['ball_x'].between(70, 79.9)]
puntWI40020 = puntWI40020[(puntWI40020['event'] == 'punt_received') | (puntWI40020['event'] == 'punt_downed') 
                  | (puntWI40020['event'] == 'fair_catch') |(puntWI40020['event'] == 'out_of_bounds')]
puntWI40020 = puntWI40020[['displayName', 'playId']]
puntWI40020 = puntWI40020.rename(columns={'playId': 'puntWI40'})
puntWI40020 = puntWI40020.groupby('displayName').count()
puntWI40020

puntWI50020 = ballp_df[ballp_df['ball_x'].between(60, 69.9)]
puntWI50020 = puntWI50020[(puntWI50020['event'] == 'punt_received') | (puntWI50020['event'] == 'punt_downed') 
                  | (puntWI50020['event'] == 'fair_catch') |(puntWI50020['event'] == 'out_of_bounds')]
puntWI50020 = puntWI50020[['displayName', 'playId']]
puntWI50020 = puntWI50020.rename(columns={'playId': 'puntWI50'})
puntWI50020 = puntWI50020.groupby('displayName').count()
puntWI50020

notReturned20 = ballp_df[(ballp_df['event'] == 'punt_downed') | (ballp_df['event'] == 'fair_catch') |(ballp_df['event'] == 'out_of_bounds')]
notReturned20 = notReturned20[['displayName', 'playId']]
notReturned20 = notReturned20.rename(columns={'playId': 'puntsNR'})
notReturned20 = notReturned20.groupby('displayName').count()
notReturned20

returned20 = ballp_df[(ballp_df['event'] == 'punt_received') | (ballp_df['event'] == 'touchback')]
returned20 = returned20[['displayName', 'playId']]
returned20 = returned20.rename(columns={'playId': 'puntsReturned'})
returned20 = returned20.groupby('displayName').count()
returned20

accuracy20 = stats_df.copy()
accuracy20 = accuracy20[accuracy20['kickDirectionIntended'] != accuracy20['kickDirectionActual']]
accuracy20 = accuracy20[['displayName', 'playId']]
accuracy20 = accuracy20.rename(columns={'playId': 'missDir'})
accuracy20 = accuracy20.groupby('displayName').count()
accuracy20

epyoe_df = epyoe_df.rename(columns={'kickerId': 'nflId'})
epyoe_df = pd.merge(epyoe_df, players_df, on = 'displayName')
epyoe_df = epyoe_df[['displayName', 'actualLength', 'prdictedLength']]
epyoe_df['delta'] = epyoe_df['actualLength'] - epyoe_df['prdictedLength']
epyoe_df = epyoe_df.groupby('displayName').sum()
epyoe_df

data_frames20 = [totalPunt20, notReturned20, returned20, puntLeft20, puntRight20,
                 stats20_15, stats20_40, stats20_50, puntWI520, puntWI1020, puntWI20020,
                 puntWI30020, puntWI40020, puntWI50020, accuracy20, puntInsideNums20, epyoe_df]
merged20 = reduce(lambda  left,right: pd.merge(left,right,on=['displayName'], how='outer'), data_frames20).fillna(0)
merged20['totalDirectional'] = merged20['puntLeft'] + merged20['puntRight']

merged20['percentDirectional'] = merged20['totalDirectional'] / merged20['totalPunts']
merged20['percentRight'] = merged20['puntRight'] / merged20['totalPunts']
merged20['percentLeft'] = merged20['puntLeft'] / merged20['totalPunts']
merged20['percentNR'] = merged20['puntsNR'] / merged20['totalPunts']
merged20['percentReturned'] = merged20['puntsReturned'] / merged20['totalPunts']
merged20['epaOppSub'] = -1 * (((merged20['puntWI5'] * -1.32) + (merged20['puntWI10'] * -1.02) + (merged20['puntWI20'] * -0.55) + (merged20['puntWI30'] * .15) + (merged20['puntWI40'] * .865) + (merged20['puntWI50'] * 1.275)) / (merged20['puntWI5'] + merged20['puntWI10'] + merged20['puntWI20']) + merged20['puntWI30'] + merged20['puntWI40'] + merged20['puntWI50'])
merged20['pinnedPlayValue'] = (merged20['pinnedHT']) / merged20['pinnedKL']
merged20['PlayValue'] = (merged20['hangTime']) / merged20['kickLength']
merged20['pinPlayValue'] = (merged20['pinHT']) / merged20['pinKL']
merged20['pinnedPlayValue'] = merged20['pinnedPlayValue'].fillna(.1)
merged20['totalPlayValue'] = (merged20['pinnedPlayValue'] + merged20['PlayValue'] + merged20['pinPlayValue']) / 3
merged20['percentMiss'] = 1 - (merged20['missDir'] / merged20['totalPunts'])
merged20['percent5'] = (merged20['puntWI5'] / merged20['totalPunts'])
merged20['percent10'] = (merged20['puntWI10'] / merged20['totalPunts'])
merged20['percent20'] = (merged20['puntWI20'] / merged20['totalPunts'])
merged20 = merged20[merged20['totalPunts'] > 40]
merged20 = merged20.fillna(0)
merged20

accSpider20_df = merged20[['percentMiss', 'percentDirectional', 'percent5', 'percent10', 'percent20','totalPlayValue', 'epaOppSub', 'percentNR', 'delta']].reset_index()
accSpider20_df
accSpider20_df[['CR', 'NR', 'P5', 'P10','P20', 'PPR', 'EPSR', 'RR', 'PYOE']] = scaler.fit_transform(accSpider20_df[['percentMiss', 'percentDirectional', 'percent5', 'percent10', 'percent20', 'totalPlayValue', 'epaOppSub', 'percentNR', 'delta']])
accSpider20_df = accSpider20_df[['displayName', 'CR', 'NR', 'P5', 'P10','P20', 'PPR', 'EPSR', 'RR', 'PYOE']]

accSpider20_df.to_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/totaldata.csv")

accuracyMetric = accSpider20_df[['displayName', 'P5', 'P10','P20', 'NR', 'CR']]
accuracyMetric['accTotal'] = accuracyMetric.sum(axis=1, numeric_only=True)
accuracyMetric = accuracyMetric[['displayName', 'accTotal']]
accuracyMetric.to_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/totaldata.csv")

playMetric = accSpider20_df[['displayName', 'PYOE', 'PPR', 'RR', 'EPSR']]
playMetric['playTotal'] = playMetric.sum(axis=1, numeric_only=True)
playMetric = playMetric[['displayName', 'playTotal']]
playMetric

totalMetric = pd.merge(accuracyMetric, playMetric, on = 'displayName')
totalMetric['Total Score'] = totalMetric.sum(axis=1, numeric_only=True)
totalMetric = totalMetric.sort_values(by='Total Score', ascending=False)
#totalMetric = totalMetric.head(5)
totalMetric = totalMetric.rename(columns={'displayName': 'Name', 'accTotal': 'AE', 'playTotal': 'GPE', 'Total Score': 'TPR'})
totalMetric

display_df = pd.merge(accSpider20_df, playMetric, on = 'displayName')
display_df = display_df.rename(columns={'displayName': 'Name'})
display_df = pd.merge(display_df, totalMetric, on = 'Name')
display_df = display_df.sort_values(by='TPR', ascending=False)
display_df

display_df.to_csv("/Users/charlestobin/Desktop/CLEAN_FILES/nfl-big-data-bowl-2022/displayTest.csv")

############################################################################################################################################

plt.style.use('ggplot')

x = totalMetric['AE'].values
y = totalMetric['GPE'].values
annotations = totalMetric['Name'].unique()

fig, ax = plt.subplots(figsize=(10, 10))

ax.grid(alpha=0.5)
# plot a vertical and horixontal line to create separate quadrants
ax.vlines(2, .75, 3.5, color='green', alpha=0.7, lw=4, linestyles='dashed')
ax.hlines(2, .75, 4.5, color='green', alpha=0.7, lw=4, linestyles='dashed')
ax.set_ylim(.75, 3.5)
ax.set_xlim(.75, 4.5)
ax.set_xlabel('AE Rating', fontsize=20)
ax.set_ylabel('GPE Rating', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

annot_styles = {
    'bbox': {'boxstyle': 'round,pad=0.5', 'facecolor': 'none', 'edgecolor':'green'},
    'fontsize': 100,
    'color': '#202f52'
}

#ax.annotate('Bad game play effect, good accuracy', xy=(x.max() - 1.8, y.min() - .4), **annot_styles)
#ax.annotate('Bad game play effect, bad accuracy', xy=(x.min() - .7, y.min() - .4), **annot_styles)
#ax.annotate('Good game play effect, good accuracy', xy=(x.max() - 1.8, y.max() + .3), **annot_styles)
#ax.annotate('Good game play effect, bad accuracy', xy=(x.min() - .7, y.max() + .3), **annot_styles)

ax.scatter(x,y,s=50,color="red")

for i, label in enumerate(annotations):
    plt.text(x[i], y[i],label)

ax.set_title('Game Play Effect (GPE) vs Accuracy Effect (AE) Comparison 2020', fontsize=20);

plt.show()

############################################################################################################################################

sum_df = accSpider20_df.copy()
sum_df['sum'] = sum_df.sum(axis=1, numeric_only=True)
sum_df = sum_df[['displayName', 'sum']]
sum_df = sum_df.sort_values(['sum'], ascending=False)
sum_df

from mplsoccer import Radar, FontManager

URL1 = ('https://github.com/googlefonts/SourceSerifProGFVersion/blob/main/'
        'fonts/SourceSerifPro-Regular.ttf?raw=true')
serif_regular = FontManager(URL1)
URL2 = ('https://github.com/googlefonts/SourceSerifProGFVersion/blob/main/'
        'fonts/SourceSerifPro-ExtraLight.ttf?raw=true')
serif_extra_light = FontManager(URL2)
URL3 = ('https://github.com/google/fonts/blob/main/ofl/rubikmonoone/'
        'RubikMonoOne-Regular.ttf?raw=true')
rubik_regular = FontManager(URL3)
URL4 = 'https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Thin.ttf?raw=true'
robotto_thin = FontManager(URL4)
URL5 = 'https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true'
robotto_regular = FontManager(URL5)
URL6 = 'https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Bold.ttf?raw=true'
robotto_bold = FontManager(URL6)


params = list(accSpider20_df.columns)
params = params[1:]
params

ranges = []
a_values = []


a = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
b = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

ranges.append((a,b))

for x in range(len(accSpider20_df['displayName'])):
    if accSpider20_df['displayName'][x] == 'Jake Bailey':
        a_values = accSpider20_df.iloc[x].values.tolist()
        
a_values = a_values[1:]

a_values


radar = Radar(params=params, min_range=a, max_range=b,round_int=[False]*8, num_rings=4, ring_width=1, center_circle_radius=1)


def radar_mosaic(radar_height=0.915, title_height=0.06, figheight=14):
  
    if title_height + radar_height > 1:
        error_msg = 'Reduce one of the radar_height or title_height so the total is ≤ 1.'
        raise ValueError(error_msg)
    endnote_height = 1 - title_height - radar_height
    figwidth = figheight * radar_height
    figure, axes = plt.subplot_mosaic([['title'], ['radar'], ['endnote']],
                                      gridspec_kw={'height_ratios': [title_height, radar_height,
                                                    endnote_height],
                                                   'bottom': 0, 'left': 0, 'top': 1,
                                                   'right': 1, 'hspace': 0},
                                      figsize=(figwidth, figheight))
    axes['title'].axis('off')
    axes['endnote'].axis('off')
    return figure, axes

fig, axs = radar_mosaic(radar_height=0.915, title_height=0.06, figheight=14)

radar.setup_axis(ax=axs['radar'])
rings_inner = radar.draw_circles(ax=axs['radar'], facecolor='#ffb2b2', edgecolor='#fc5f5f')
radar_output = radar.draw_radar(a_values, ax=axs['radar'],
                                kwargs_radar={'facecolor': '#233747'},
                                kwargs_rings={'facecolor': '#66d8ba'})
radar_poly, rings_outer, vertices = radar_output
range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=25,
                                       fontproperties=robotto_thin.prop)
param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=25,
                                       fontproperties=robotto_regular.prop)


endnote_text = axs['endnote'].text(0.99, 0.5, 'Inspired By: StatsBomb / Rami Moghadam', fontsize=15,
                                   fontproperties=robotto_thin.prop, ha='right', va='center')
title1_text = axs['title'].text(0.01, 0.65, 'Jake Bailey', fontsize=25,
                                fontproperties=robotto_bold.prop, ha='left', va='center')
title2_text = axs['title'].text(0.01, 0.25, 'New England Patriots', fontsize=20,
                                fontproperties=robotto_regular.prop,
                                ha='left', va='center', color='#B6282F')
title3_text = axs['title'].text(0.99, 0.65, 'Radar Chart', fontsize=25,
                                fontproperties=robotto_bold.prop, ha='right', va='center')
title4_text = axs['title'].text(0.99, 0.25, 'Punt Stats: 2020', fontsize=20,
                                fontproperties=robotto_regular.prop,
                                ha='right', va='center', color='#B6282F')

merge20Stats = merged20.describe()

merge20Stats
                                                                                                                             
                                                                                                                                  
                                                                                                                                 
                                                                                                                                  
