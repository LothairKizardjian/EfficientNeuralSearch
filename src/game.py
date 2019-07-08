import chess
import sys
import os

from model_chess import *
import numpy as np
model = model_chess.load()
board = chess.Board()

def play():
    print(board.unicode())
    while True:
        move = input()
        if move=='stop':
            break
        board.push_san(move)
        print(board.unicode())
        
        fen = board.fen()
        tensor = []
        tensor.append(fen_to_tensor(fen))
        tensor = np.asarray(tensor)

        model_move = model.predict(tensor)
        print(model_move)
        
        board.push_san(model_move)
        print(board.unicode())

play()
