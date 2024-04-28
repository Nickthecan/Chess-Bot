import chess
import chess.engine
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

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

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

#turns the board into a 3d matrix for funsies. But in actualkty it will be better for the neural network model to read
""" ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣶⠛⠛⠉⠁⢀⠀⠀⠀⠀⠉⠙⠓⠲⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⢀⣠⣶⣿⣛⣍⢋⡛⠀⠠⠐⠀⠀⠀⠀⠀⢀⡴⠞⠳⢤⣈⣑⣤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣀⣤⣶⣶⣿⠿⣿⣿⣿⡿⣿⡿⠿⠿⣿⣶⣦⣤⣀⡀⢀⣠⣤⣶⡶⠾⠟⠛⠛⠛⠛⠛⠻⡿⣿⣶⣶⣦⣤⣀⠀⠀⠀⠀
⠀⠀⠿⣿⣿⡿⠋⢀⣼⣿⣿⣻⣿⣿⣐⠀⠀⠐⠌⢙⢿⣿⣿⣿⡿⠛⢁⣰⠖⠀⠀⠀⠀⠀⠀⠀⠹⡄⠉⢿⣿⣿⡿⠀⠀⠀⠀
⠀⠀⠀⠸⣿⡿⣶⣾⣿⣿⣿⣿⣿⣳⣿⣾⣶⣄⠀⢊⢼⣿⠀⢸⡇⢠⣿⣯⣶⣶⣶⡒⠶⡀⠀⠀⠀⠹⣴⣿⣿⡟⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣿⡇⣼⣿⣿⣿⣿⣿⢧⣿⣿⣿⣿⣿⠱⣮⢼⡟⠀⣸⣇⠸⣟⣿⣿⣿⣿⣿⠀⢹⡀⠀⠀⠀⢻⣰⣿⠇⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⡿⢰⣿⣿⠧⡀⠉⣿⣄⣿⣿⣿⣿⣿⣿⠀⢸⠁⠀⠀⠀⠀⣹⡿⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣣⣿⡿⢿⣻⣅⠀⠘⢻⣿⣿⣿⢿⡿⠃⣠⠇⠀⠀⠀⢀⣴⣿⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣏⢳⢣⡓⣎⠳⡄⢀⠈⠛⠿⣿⣿⣿⣡⣤⣤⣴⡶⠟⠉⢸⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⣷⢷⡈⣇⠶⣉⡈⠷⡈⠆⣀⠀⠀⠀⠀⠈⠁⠉⠈⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣼⣳⡭⡜⡸⡄⡜⠡⠌⡐⠤⠈⠀⠀⣠⡴⠀⠀⠀⠀⠀⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣮⣕⣢⣌⣐⣠⣀⣠⣤⣶⠟⡉⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀
⢀⣤⣤⣤⣤⣌⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣿⣿⣿⣿⡟⠉⢹⠋⠁⢹⡟⢋⠡⠊⠀⠀⠀⠀⠀⠀⠀⠀⣸⠃⢀⣤⣄⠀⠀⠀⠀
⢸⣿⣿⣿⣿⡟⠈⣿⣿⣿⣿⣿⣿⣿⣿⣯⣿⢾⣻⣿⢿⡿⡿⣿⣛⠻⡌⡑⡂⢁⠢⠄⠀⠀⠠⠀⠀⢀⠏⠀⠺⣿⣿⡄⠀⠀⠀
⢀⣿⣿⡜⣿⣧⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣯⡿⣿⣽⣞⡽⣳⢳⡍⢶⢱⡐⡘⢢⠀⠄⡀⠀⠀⠀⢠⠟⠀⠀⣼⡿⠉⠀⠀⠀⠀
⣼⣿⣿⡿⣜⢹⣧⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣽⡾⣟⣷⣯⣯⡝⣎⢦⡱⠸⣄⠣⡘⠄⠒⡀⣲⠏⠀⢀⣼⠟⠀⠀⠀⠀⠀⠀
⣿⣿⣿⣿⣮⣃⠌⢳⡀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣻⣶⢿⣺⣝⡮⢷⣌⢳⠡⢎⣵⠞⠁⠀⣠⡟⠁⠀⠀⠀⠀⠀⠀⢰
⢻⣿⣿⡙⢿⣿⣦⠁⠻⡄⠀⠀⠉⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣿⢾⣯⣞⣧⠿⠋⠁⠀⢠⣾⣏⣀⡤⣾⣿⠖⠀⠀⠀⡿
⠈⠻⣿⣧⡌⠙⠿⣷⣤⣿⠀⠀⠀⠀⠀⠈⠉⠛⠛⠿⠿⢿⡿⣿⡿⠿⠟⠛⠛⠉⠀⠀⠀⠀⠀⠀⠻⠛⠋⢀⣿⠃⠀⣠⠇⣼⠃
⠀⠀⠙⢿⣿⣷⣤⡄⢀⠈⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⠏⠀⣰⠏⡴⠃⠀
⠀⠀⠀⠀⠙⠻⠿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡿⡁⢂⣴⣯⠞⠁⠀⠀
⠀⠀⠀⠀⠀⢀⣀⣀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠷⣷⡿⠟⠁⠀⠀⠀⠀ """
def split_dimensions(board):
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(square, chess.WHITE):
            i = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - i[0]][i[1]] = 1
        for square in board.pieces(square, chess.BLACK):
            i = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - i[0]][i[1]] = 1

    who_going = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = who_going

    return board3d



def start_game(board):
    print(board)
    print(split_dimensions(board))

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            player_move = input("pick a move (ex: e2e4): ")
            try:
                move = chess.Move.from_uci(player_move)
                if move in board.legal_moves:
                    board.push(move)
                    print(board)
                    print(evaluation(board))
                else:
                    print("invalid move")
            except ValueError:
                print("invalid move")
        else:
            computer_move(board)
            """ player_move = input("pick a move (ex: e2e4): ")
            try:
                move = chess.Move.from_uci(player_move)
                if move in board.legal_moves:
                    board.push(move)
                    print(board)
                    print(evaluation(board))
                else:
                    print("invalid move")
            except ValueError:
                print("invalid move") """

def computer_move(board):
    depth = 1
    alpha = float("-inf")
    beta = float("inf")
    move, score = miniMax(board, depth, alpha, beta)
    board.push(move)
    print(board)
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
    
def evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
    result = engine.analyse(board, chess.engine.Limit(time = time_limit))
    score = result['score'].relative.score()
    if board.turn == chess.WHITE:
        return score
    else:
        return -score
    



def main():
    board = chess.Board()
    start_game(board)

main()