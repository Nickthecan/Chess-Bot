import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim

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