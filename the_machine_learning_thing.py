import chess.pgn
import tqdm
import pandas as pd
pd.options.display.max_columns = 999
import datetime
import zipfile
import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import ast

""" loading the database pgn of games from march 2014 to a .csv file """
#NUM_GAMES = 795173
#rows = []
#with open(f'data/lichess_db_standard_rated_2014-03.pgn') as pgn:
#    for game in tqdm.tqdm(range(NUM_GAMES)):
#        row = {}
#        game = chess.pgn.read_game(pgn)
#        row['headers']=game.headers.__dict__
#        row['moves']=[x.uci() for x in game.mainline_moves()]
#        rows.append(row)
#games=pd.DataFrame(rows)
#games
#games.to_csv("data/loaded_games.csv", index=False)


""" Needed to shorten the csv file in order to get only the best games """
#df = pd.read_csv('data/loaded_games.csv')
#rows_to_drop = []
#
#for i in range(len(df)):
#    row = df.iloc[i]
#    headers_dict = ast.literal_eval(row['headers'])
#    white_elo = headers_dict["_others"]["WhiteElo"]
#    if white_elo.isdigit():
#        elo = int(white_elo)
#    else:
#        elo = 0
#    if elo < 2000:
#        rows_to_drop.append(i)
#
#df.drop(index=rows_to_drop, inplace=True)
#
#df.to_csv("data/filtered_games.csv", index=False)


