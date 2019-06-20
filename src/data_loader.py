import numpy as np
import chess.pgn
import pgn_tensors_utils
import os

def load_games(pgn,game_nb):    
    print("loading games ...")
    games = []
    game  = chess.pgn.read_game(pgn)
    counter = 0
    
    while game != None and counter < game_nb:
        games.append(game)
        game = chess.pgn.read_game(pgn)
        counter+=1
        
    print("{} games loaded".format(game_nb))
    return games

def load_data(pgn,game_nb):
    '''
    Load the tensors, the according labels and store them in a .npy file
    '''
    tensors_file_exist = os.path.isfile('./data/tensors_numpy.npy')
    labels_file_exist  = os.path.isfile('./data/labels_numpy.npy')
    
    if tensors_file_exist == False or labels_file_exist == False:
        games = load_games(pgn,game_nb)
        print("loading tensors and according labels ...")
        tensors, labels = pgn_tensors_utils.tensors_labels_from_games(games)
        np.save('./data/tensors_numpy.npy', tensors)
        np.save('./data/labels_numpy.npy', labels)
    else:
        tensors = np.load('./data/tensors_numpy.npy')
        labels  = np.load('./data/labels_numpy.npy')

    return tensors, labels

    
