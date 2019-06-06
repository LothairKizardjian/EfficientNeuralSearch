import numpy as np
import chess

nb_moves = 19
tensors = []
moves   = []

tensor_starting_board = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,-1,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,-1,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ])])])

move1 = chess.Move(chess.E2,chess.E4)
move2 = chess.Move(chess.B7,chess.B6)
move3 = chess.Move(chess.D2,chess.D4)

tensor_move3 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,-1,-1,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,-1,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,-1,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1])])])

move4 = chess.Move(chess.C8,chess.B7)
move5 = chess.Move(chess.F1,chess.D3)
move6 = chess.Move(chess.D7,chess.D5)

tensor_move6 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,-1,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,-1,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ])])])

move7 = chess.Move(chess.E4,chess.D5)
move8 = chess.Move(chess.B7,chess.D5)
move9 = chess.Move(chess.G1,chess.F3)

tensor_move9 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,-1,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,-1,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1])])])

move10= chess.Move(chess.B8,chess.C6)
move11= chess.Move(chess.B1,chess.C3)
move12= chess.Move(chess.D5,chess.F3)

tensor_move12 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,-1,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,-1,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,-1 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ])])])

move13= chess.Move(chess.D1,chess.F3)
move14= chess.Move(chess.C6,chess.D4)
move15= chess.Move(chess.D3,chess.B5)

tensor_move15 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,-1,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1])])])

move16= chess.Move(chess.C7,chess.C6)
move17= chess.Move(chess.B5,chess.C6)
move18= chess.Move(chess.D4,chess.C6)

tensor_move18 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,0 ,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,-1,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]),
        np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ])])])

move19= chess.Move(chess.F3,chess.C6)

tensor_move19 = np.array([
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([-1,0 ,0 ,0 ,-1,-1,-1,-1]),
        np.array([0 ,-1,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([-1,0 ,0 ,0 ,0 ,0 ,0 ,-1]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ])]),

    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,-1,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,0 ,-1,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,-1,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ])]),
    
    np.array([
        np.array([0 ,0 ,0 ,0 ,-1,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]),
        np.array([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ])]),    
    
    np.array([
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
        np.array([-1,-1,-1,-1,-1,-1,-1,-1])])])

moves.append(move1)
moves.append(move2)
moves.append(move3)
moves.append(move4)
moves.append(move5)
moves.append(move6)
moves.append(move7)
moves.append(move8)
moves.append(move9)
moves.append(move10)
moves.append(move11)
moves.append(move12)
moves.append(move13)
moves.append(move14)
moves.append(move15)
moves.append(move16)
moves.append(move17)
moves.append(move18)
moves.append(move19)

tensors.append(tensor_move3)
tensors.append(tensor_move6)
tensors.append(tensor_move9)
tensors.append(tensor_move12)
tensors.append(tensor_move15)
tensors.append(tensor_move18)
tensors.append(tensor_move19)
