class GameState():
    def __init__(self):
        self.board = [
            ["blackRook", "blackKnight", "blackBishop", "blackQueen", "blackKing", "blackBishop", "blackKnight", "blackRook"],
            ["blackPawn", "blackPawn", "blackPawn", "blackPawn", "blackPawn", "blackPawn", "blackPawn", "blackPawn"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["whitePawn", "whitePawn", "whitePawn", "whitePawn", "whitePawn", "whitePawn", "whitePawn", "whitePawn"],
            ["whiteRook", "whiteKnight", "whiteBishop", "whiteQueen", "whiteKing", "whiteBishop", "whiteKnight", "whiteRook"]
        ]
        self.white_to_move = True
        self.move_log = []

class Move():
    def __init__(self, start_square, end_square, board):
        self.start_row = start_square[0]
        self.start_column = start_square[1]
        self.end_row = end_square[0]
        self.end_column = end_square[1]
        self.piece_moved = board[self.start_row][self.start_column]
        self.piece_captured = board[self.end_row][self.end_column]
