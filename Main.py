import pygame as p # type: ignore
import chess_engine 

WIDTH = HEIGHT = 480
DIMENSION = 8
SQUARE_SIZE = HEIGHT // DIMENSION
MAX_FPS = 60
IMAGES = {}

#Load images
def load_images():
    pieces = ["blackRook", "blackKnight", "blackBishop", "blackQueen", "blackKing", "blackPawn",
              "whiteRook", "whiteKnight", "whiteBishop", "whiteQueen", "whiteKing", "whitePawn"]
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("assets/"+ piece + ".png"), (SQUARE_SIZE, SQUARE_SIZE))

"""
Main Driver for program, handles user input and updates the graphics
"""
def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = chess_engine.GameState()
    #game_state.board
    print(gs.board)
    load_images()
    running = True
    square_selected = ()
    player_click = []

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos()
                column = location[0] // SQUARE_SIZE
                row = location[1] // SQUARE_SIZE
                #in case player selects the same square again, undo the move
                if (square_selected == (row, column)):
                    square_selected = ()
                    player_click = []
                #select piece
                else:
                    square_selected = (row, column) 
                    #first click, piece selected, second click, square to move the piece to
                    player_click.append(square_selected)
                if len(player_click) == 2:
                    pass

             
        draw_game_state(screen, gs)
        clock.tick(MAX_FPS)
        p.display.flip()

"""
Method that helps draw the graphics for the current game state
"""
def draw_game_state(screen, gs):
    draw_board(screen)
    draw_pieces(screen, gs.board)

"""
Helper method that draws the squares on the board
"""
def draw_board(screen):
    colors = [p.Color(234, 234, 234), p.Color(127, 100, 167)]

    for row in range(DIMENSION):
        for column in range(DIMENSION):
            color = colors[((row + column) % 2)]
            p.draw.rect(screen, color, p.Rect(column*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                
"""
Helper method to draw the pieces on the board using the current game state's board
"""
def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            piece = board[row][column]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(column*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

if __name__ == "__main__":
    main()