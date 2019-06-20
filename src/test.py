import pytest
import chess
import numpy as np
import config
import tensors_to_test

board = chess.Board()
fen_starting_board =board.fen()

def test_fen_to_tensor(tensor,fen):
    tensor_to_test = config.fen_to_tensor(fen) 
    assert tensor.shape == tensor_to_test.shape
    for i in range(len(tensor)):
        for j in range(len(tensor[0])):
            for k in range(len(tensor[0][0])):
                assert tensor[i][j][k] == tensor_to_test[i][j][k]
    print('true')

test_fen_to_tensor(tensors_to_test.tensor_starting_board,fen_starting_board)

counter = 0
for i in range(1,tensors_to_test.nb_moves+1):
    board.push(tensors_to_test.moves[i-1])
    fen = board.fen()
    if i%3 == 0 :
        tensor = tensors_to_test.tensors[counter]
        test_fen_to_tensor(tensor,fen)
        counter += 1
    if i==tensors_to_test.nb_moves:
        tensor = tensors_to_test.tensors[counter]
        test_fen_to_tensor(tensor,fen)
print(board)
