from flask import Flask, render_template, jsonify, request
import chess_game
import chess

app = Flask(__name__)

#THIS IS GOING TO BE A CHESS BOARD ALWAYS
current_board = chess.Board()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/update_board_position", methods=["POST"])
def update_board_position():
    global current_board
    data = request.get_json()
    fen = data.get("board_position")
    current_board = chess.Board(fen)
    

@app.route("/get_board_position", methods=["GET"])
def get_board_position():
    global current_board
    fen = current_board.fen()
    return jsonify({"board_position": fen})

@app.route("/submit_move", methods=["POST"])
def submit_move():
    global current_board

    data = request.get_json()
    fen = data.get("fen")
    move = data.get("move")
    current_board = chess.Board(fen)
    chess_game.player_move(current_board, move)

if __name__ == "__main__":
    app.run(debug=True, port=8080)