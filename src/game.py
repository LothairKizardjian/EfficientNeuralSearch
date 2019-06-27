import chess
import sys
import os

from pgn_tensors_utils import *

json_file = open('./data/models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

board = chess.Board()
print(board.unicode())

while True:
    move = input()
    board.push_san(move)
    print(board.unicode())

    fen = board.fen()
    tensor = fen_to_tensor(fen)

    model_move = loaded_move.predict(tensor)
    board.push_san(model_move)
    print(board.unicode())
