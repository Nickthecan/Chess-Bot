import chess
import chess.engine
from chess.engine import Cp
import math

def start_game(board):
    print(board)

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
    depth = 3
    alpha = -math.inf
    beta = math.inf
    miniMax(board, depth, alpha, beta)

def miniMax(board, depth, alpha, beta):
    if depth == 0:
        return evaluation(board)
    
    scores = []
    for move in board.legal_moves:
        board.push(move)
        scores.append(miniMax(board, depth - 1, alpha, beta))
        board.pop()
    
    return max(scores) if board.turn == chess.WHITE else min(scores)
    
def evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci("stockfish\stockfish-windows-x86-64-avx2.exe")
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
        

    
