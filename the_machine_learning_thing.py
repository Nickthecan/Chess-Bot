import chess.pgn
import chess_game
import chess.engine
import keras
import tqdm
import pandas as pd
pd.options.display.max_columns = 999
import datetime
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
import json 
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
""" NOTES """
# X is going to be the board position
# y is going to be the evaluation of the board position
# use X and y in the model in order to train and let the model understand board positions and their values

# From Jacob
# btw for like an ANN, in super basic terms think of each like square on a chess board as a variable and you input those 64 squares as numerical values 
# and the machine learning model does some sort formula on each square and comes up with an answer (for example, the evlauation) 
# then, it compares it with the actual value and then tweaks all of those equations it used, and just does this like a bunch of times 
# and it just tries to get the minimum error across your entire dataset
# when you train a model, you actually give it the input (chess board position) and the output (actual evaluation)
# while training, the model needs the actual evaluations to check if it's predicted ones are close to the actual value
# at first, it's not close at all, since it's basically just putting random equations and hoping it works

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



#def cweate_dataset_uwu(df):
#    row = df.iloc[0]
#    moves = ast.literal_eval(row['moves'])
#    game = chess.pgn.Game()
#    board = chess.Board()
#    for move in moves:
#        board.push_uci(move)
#        game.add_main_variation(chess.Move.from_uci(board.peek().uci()))
#        X.append(board.copy())
#    print("done")


""" find all the FEN positions in the json file """
def populate_position_dataset(json_file):
    for i in range(10000):
        line = json_file.readline()
        block = json.loads(line)
        fen = block["fen"]
        X.append(fen)
    print("positions done")

""" finds the best move in the position with the corresponding evaluation """
def populate_evaluation_dataset(json_file):
    for i in range(10000):
        different_position_for_evaluation = {}
        line = json_file.readline()
        block = json.loads(line)

        highest_depth = 0
        cp = 0
        best_move = ""
        for evaluation in block['evals']:
            for pv in evaluation['pvs']:
                if 'cp' in pv:
                    if evaluation['depth'] > highest_depth:
                        highest_depth = evaluation['depth']
                        cp = pv['cp']
                        best_move = pv['line'].split()[0]
            different_position_for_evaluation = {cp, best_move}
        y.append(different_position_for_evaluation)
                
        


""" this will be the supervised learning to check the actual evaluation """ 

def evaluation(board, time_limit = 0.01):
    engine = chess_game.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
    result = engine.analyse(board, chess_game.engine.Limit(time = time_limit))
    score = result['score'].relative.score()
    if result is not None:
        """ if board.turn == chess.WHITE:
            return score
        else:
            return -score """
        return score
    else:
        return 0
    

#start of the program

X = []
y = []

with open("data/lichess_db_eval.jsonl", 'r') as json_file:
    populate_position_dataset(json_file)
    json_file.seek(0)
    populate_evaluation_dataset(json_file)

print(X[8])
print(y[8])
print(len(X), len(y))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    # relu: rectified Linear unit
    Dense(32, activation='relu', input_dim=len(X_train)),
    Dense(64, activation='relu'),
    # softmax: pick values for each neuron so that all neurons will add up to 1
    Dense(2, activation='softmax')
])

# accuracy defines the metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# epochs: how many times the model will see the information
model.fit(X_train, y_train, epochs=200, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_acc)
print("Test Loss: ", test_loss)
         

