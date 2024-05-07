import chess
import chess.engine
import numpy as np
from keras.models import load_model # type: ignore
import requests

# load the model
model = load_model('model2.keras')

# square index to help with the board to 3d matrix conversions
squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,   
}

# start game function to run the game until checkmate occurs
def start_game(board):
    print(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            player_move(board)
            print(board)
        else:
            computer_move(board)
            print(board)

# allows the player to make a valid move to the chess position
def player_move(board):
    player_move = input("pick a move (ex: e2e4): ")
    try:
        move = chess.Move.from_uci(player_move)
        if move in board.legal_moves:
            board.push(move)
            #send_board_to_server(board)
        else:
            print("invalid move")
    except ValueError:
        print("invalid move")

# when it is the computer's turn, use the minimax function along with alpha beta pruning in order to calculate the best score
# from the model
def computer_move(board):
    depth = 3
    alpha = float("-inf")
    beta = float("inf")
    move, score = miniMax(board, depth, alpha, beta)
    board.push(move)
    #send_board_to_server(board)
    print(score)

def miniMax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return None, evaluation(board)
    
    best_move = None
    if board.turn == chess.WHITE:
        max_score = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            _, score = miniMax(board, depth - 1, alpha, beta)
            board.pop()
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break
        return best_move, max_score
    else:
        min_score = float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, score = miniMax(board, depth - 1, alpha, beta)
            board.pop()
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, min_score)
            if beta <= alpha:
                break
        return best_move, min_score

# evaluation function that allows the model to predict the evaluation of the board position
def evaluation(board):
    matrix = make_matrix(board.fen())
    score = model.predict(np.array([matrix]))[0][0]
    return score

# helper function in order to convert boards into 3d matrices
def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

# function used with the CNN in order to convert boards into 3d matrices
def make_matrix_CNN(fen, best_move):
    board = chess.Board(fen)
    board.push_uci(best_move)
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            i = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - i[0]][i[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            i = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - i[0]][i[1]] = 1

    temp = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = temp
    
    return board3d

# function used with the game to convert boards into 3d matrices in order for the model to use
def make_matrix(fen):
    board = chess.Board(fen)
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            i = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - i[0]][i[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            i = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - i[0]][i[1]] = 1

    temp = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = temp
    
    return board3d

# sending board to the front end (not needed)
def send_board_to_server(board):
    fen_string = board.fen()
    data = {"board_position": fen_string}
    try:
        requests.post("http://localhost:8080/update_board_position", json=data)
    except Exception as e:
        print("Error sending board position to server:", e)
    
# main function, initializes a chess board
def main():
    board = chess.Board()
    start_game(board)

if __name__ == "__main__":
    main()