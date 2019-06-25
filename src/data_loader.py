import numpy as np
import chess.pgn
import pgn_tensors_utils
import os

def load_games(pgn_path,game_nb):
    print("loading games ...")
    pgn_file_exist = os.path.isfile(pgn_path)
    if pgn_file_exist == False:
        raise('File {} not found'.format(pgn_path))
    pgn = open(pgn_path)
    games = []
    game  = chess.pgn.read_game(pgn)
    counter = 0
    
    while game != None and counter < game_nb:
        games.append(game)
        game = chess.pgn.read_game(pgn)
        counter+=1
        
    print("{} games loaded".format(game_nb))
    return games

def load_data(pgn_path,game_nb):
    '''
    Load the tensors, the according labels and store them in a .npy file
    '''
    tensors_file_exist = os.path.isfile('./data/tensors_numpy.npy')
    labels_files_exist = os.path.isfile('./data/labels_numpy.npy')
    if tensors_file_exist == False or labels_files_exist == False:
        games = load_games(pgn_path,game_nb)
        print("loading tensors and according labels ...")
        tensors,labels = pgn_tensors_utils.tensors_labels_from_games(games)
        if os.path.isdir('./data') == False:
            os.mkdir('./data')
        np.save('./data/labels_numpy.npy', labels)
        np.save('./data/tensors_numpy.npy', tensors)
    else:
        tensors = np.load('./data/tensors_numpy.npy')
        labels  = np.load('./data/labels_numpy.npy')
    return tensors,labels

def select_random_examples(tensors,labels,nb_ex):
    if len(tensors) != len(labels) :
        raise('Tensors and labels have different length')
    else:
        samples = np.random.choice(len(tensors), size=nb_ex, replace=False)
        x = tensors[samples]
        y = labels[samples]
    return x,y
