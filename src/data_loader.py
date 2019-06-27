import numpy as np
import chess.pgn
import pgn_tensors_utils
import bz2
import os

def load_pgn_from_bz2(bz2_path):
    bz2_file_exist = os.path.isfile(bz2_path)
    if bz2_file_exist == False:
        print('File {} not found'.format(bz2_path))
        return 0
    with open(bz2_path, 'rb') as source, open(bz2_path.replace('.bz2',''), 'wb') as dest:
        dest.write(bz2.decompress(source.read()))
    return 1
    
def load_games_from_pgn_path(pgn_path,game_nb):
    pgn_file_exist = os.path.isfile(pgn_path)
    if pgn_file_exist == False:
        if(load_pgn_from_bz2(pgn_path+'.bz2')==0):  
            print('File {} not found'.format(pgn_path))
            return 0
    pgn = open(pgn_path)
    return load_games_from_pgn(pgn,game_nb)

def load_games_from_pgn(pgn,game_nb):
    name = pgn.name.split('/')[2].replace('.pgn','')
    print('Loading games for pgn {} ...'.format(name))
    games = []
    game  = chess.pgn.read_game(pgn)
    counter = 0
    
    while game != None and counter < game_nb:
        games.append(game)
        game = chess.pgn.read_game(pgn)
        counter+=1
        
    print("{} games loaded".format(counter))
    return games
    

def load_data(pgn_path,game_nb):
    '''
    Load the tensors, the according labels and store them in a .npy file
    '''
    suffixe = pgn_path.split('/')
    suffixe = suffixe[2].replace('.pgn','')
    tensors_file_exist = os.path.isfile('./data/tensors/tensors_numpy_{}.npy'.format(suffixe))
    labels_files_exist = os.path.isfile('./data/tensors/labels_numpy_{}.npy'.format(suffixe))
    if tensors_file_exist == False or labels_files_exist == False:
        games = load_games_from_pgn_path(pgn_path,game_nb)
        print("loading tensors and according labels ...")
        tensors,labels = pgn_tensors_utils.tensors_labels_from_games(games)
        if os.path.isdir('./data') == False:
            os.mkdir('./data')
        np.save('./data/tensors/labels_numpy_{}.npy'.format(suffixe), labels)
        np.save('./data/tensors/tensors_numpy_{}.npy'.format(suffixe), tensors)
    else:
        tensors = np.load('./data/tensors/tensors_numpy_{}.npy'.format(suffixe))
        labels  = np.load('./data/tensors/labels_numpy_{}.npy'.format(suffixe))
    return tensors,labels

def load_data_from_multiple_files(paths,game_nb):
    tensors = []
    labels  = []
    for pgn_path in paths:
        t,l = load_data(pgn_path,game_nb)
        for row in t:
            tensors.append(row)
        for row in l:
            labels.append(row)
    
    return np.asarray(tensors),np.asarray(labels)

def select_random_examples(tensors,labels,nb_ex):
    if len(tensors) != len(labels) :
        raise('Tensors and labels have different length')
    else:
        samples = np.random.choice(len(tensors), size=nb_ex, replace=False)
        x = tensors[samples]
        y = labels[samples]
    return x,y
