import chess
import sys
import os

from model_chess import *

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
        tensor = fen_to_tensor(fen)
        
        model_move = loaded_move.predict(tensor)
        board.push_san(model_move)
        print(board.unicode())
