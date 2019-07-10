import chess
import sys
import os

from model_chess import *
import numpy as np
mod = model_chess.load()
model =  model_chess(model = mod)
board = chess.Board()

def play():
    print(board.unicode())
    while True:
        move = input('move : ')
        if move=='stop':
            break
        legal_moves = board.legal_moves
        move = board.parse_san(move)
        while move not in legal_moves:
            move = input('illegal move, give another move : ')
            move = board.parse_san(move)        
        board.push(move)
        print(board.unicode())
        
        fen = board.fen()
        tensor = []
        tensor.append(fen_to_tensor(fen))
        tensor = np.asarray(tensor)

        model_move = model.get_move(mod.predict(tensor),legal_moves=legal_moves)
        
        board.push(model_move)
        print(board.unicode())

play()
