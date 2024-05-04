import cnn_model
import chess_game
import chess
import chess_engine

def evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
    result = engine.analyse(board, chess.engine.Limit(time = time_limit))
    score = result['score'].white().score()
    return score
    
fen = "8/8/8/8/1QK5/8/k7/8 b - - 0 1"

cp_best_move = {'cp': 311, 'best_move': ""}

#print(chess_game.make_matrix_CNN(fen, cp_best_move['best_move']))

board = chess.Board(fen)
#board.push_uci(cp_best_move['best_move'])
print(evaluation(board))




