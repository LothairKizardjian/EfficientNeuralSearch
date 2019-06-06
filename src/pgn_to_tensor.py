import numpy as np
import chess.pgn
import copy

def fen_to_tensor(fen):    
    pieces_str = "PRNBQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}
    tensor = np.zeros((7,8,8)).astype('int32')

    inputliste = fen.split(' ')
    row = 0
    col = 0
    for i, c in enumerate(inputliste[0]):
        if c in pieces:
            tensor[np.abs(pieces_dict[c])-1,row, col] = np.sign(pieces_dict[c])
            col = col + 1
        elif c == '/':  # new row
            row = row + 1
            col = 0
        elif int(c) in valid_spaces:
            col = col + int(c)
        else:
            raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))

    if inputliste[1] == "w":
        for i in range(8):
            for j in range(8):
                tensor[6,i,j] = 1
    else:
        for i in range(8):
            for j in range(8):
                tensor[6,i,j] = -1
  
    return tensor

def tensors_for_each_move(game):
    tensors = []
    board = game.board()
    fen = board.fen()
    tensors.append(fen_to_tensor(fen))
    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()
        tensors.append(fen_to_tensor(fen))
    return tensors
        
    
chess_games_pgn = open("../PGN_chess_games/chess_games.pgn")
first_game = chess.pgn.read_game(chess_games_pgn)
first_game_tensors = tensors_for_each_move(first_game)

