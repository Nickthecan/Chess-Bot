from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np
import json
import chess_game

# finds all the FEN positions in the json file (Actually 8,000 positions)
def populate_position_dataset(json_file):
    for i in range(200000):
        line = json_file.readline()
        block = json.loads(line)
        fen = block["fen"]
        X.append(fen)

# finds the best move in the position with the corresponding evaluation (add the cp to the y dataset)
def populate_evaluation_dataset(json_file):
    for i in range(200000):
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
                elif 'mate' in pv:
                    mate_in_moves = pv['mate']
                    cp = 10000 if mate_in_moves > 0 else -100000
                    best_move = pv['line'].split()[0]
            different_position_for_evaluation = {'cp': cp, 'best_move': best_move}
        cp_moves.append(different_position_for_evaluation) 

# function to build the model
# input -> 14 x 8 x 8 matrix representation of the board
# output -> evalutation of the board position
def build_model(conv_size, conv_depth):
    # defines the input layer of the model
    board3d = layers.Input(shape=(14, 8, 8))

    # applies a 2D convolutional layer to the inptu tensor, uses conv_size of 32 with a kernel size of 3x3
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(board3d)

    # for loop in order to apply multiple convolutional layers to the model for a depth of 4
    for _ in range(conv_depth):
        previous = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        # normalizes the activations of the previous layer in order to speed up training
        x = layers.BatchNormalization()(x)
        # applies a Rectified Linear Unit to the tensor allowing it to learn patterns in the data
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        # adds a connection between the current block and the previous layer to allow the model to learn more effectively
        x = layers.Add()([x, previous])
    # flattens the 3D tensor into a 1D tensor
    x = layers.Flatten()(x)
    # Adds a fully connected layer and applies sigmoid activation
    x = layers.Dense(64, 'sigmoid')(x)
    # outputs the evalution of the model
    x = layers.Dense(1, 'sigmoid')(x)

    # return model
    return models.Model(inputs=board3d, outputs=x)

if __name__ == "__main__":
    # initialize the datasets
    X = []
    y = []
    cp_moves = []

    # open the json file from lichess evaluation database 
    with open("data/lichess_db_eval.jsonl", 'r') as json_file:
        populate_position_dataset(json_file)
        # Go back to the top of the json file
        json_file.seek(0)
        populate_evaluation_dataset(json_file)

    # Convert all the FEN positions from the X dataset into a 14x8x8 matrix in order to more easily pass it into the neural network
    # also add all the cp values into the y dataset. This will act as the output
    for i in range(len(X)):
        X[i] = chess_game.make_matrix_CNN(X[i], cp_moves[i]['best_move'])
        y.append(cp_moves[i]['cp'])

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    #Split the datasets X and y into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #builds the model
    model = build_model(32, 4)

    # training the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # model summary
    model.summary()
    # fitting the model with the dataset
    model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=1, validation_steps=1)
            
    # saves the model for future use
    model.save('model2.keras')

    # evaluates the test loss and accuracy of the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy: ", test_acc)
    print("Test Loss: ", test_loss)