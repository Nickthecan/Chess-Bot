import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np
import json
import chess_game

# finds all the FEN positions in the json file (Actually 8,000 positions)
def populate_position_dataset(json_file):
    for i in range(10000):
        line = json_file.readline()
        block = json.loads(line)
        fen = block["fen"]
        X.append(fen)
    print("positions done")

# finds the best move in the position with the corresponding evaluation (add the cp to the y dataset)
def populate_evaluation_dataset(json_file):
    for i in range(10000):
        # different_position_for_evaluation = {}
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
            # different_position_for_evaluation = {'cp': cp, 'best_move': best_move}
        y.append(cp)

# initialize the datasets
X = []
y = []

# open the json file from lichess evaluation database 
with open("data/lichess_db_eval.jsonl", 'r') as json_file:
    populate_position_dataset(json_file)
    # Go back to the top of the json file
    json_file.seek(0)
    populate_evaluation_dataset(json_file)

# Convert all the FEN positions from the X dataset into a 14x8x8 matrix in order to more easily pass it into the neural network
X = [chess_game.make_matrix(fen) for fen in X]

# convert to numpy arrays
X = np.array(X)
y = np.array(y)

#Split the datasets X and y into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model by intializing lineear stacks of layers
model = Sequential([
    # Flatten the input to represent a 14x8x8 matrix
    Flatten(input_shape=(14, 8, 8)),
    # relu: rectified Linear unit (Hidden Layer)
    Dense(128, activation='relu'),
    # Output layer of the model, outputs the evaluation
    Dense(1, activation='sigmoid')
])

# accuracy defines the metrics
model.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['accuracy'])
# epochs: how many times the model will see the information
model.fit(X_train, y_train, epochs=1000, batch_size=64)
# Saves the model for future use
model.save('model.keras')

#evaluates the test loss and accuracy of the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test_acc)
print("Test Loss: ", test_loss)