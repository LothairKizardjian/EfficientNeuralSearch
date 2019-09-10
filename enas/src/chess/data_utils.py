import os
import sys
import _pickle as pickle
import numpy as np
import tensorflow as tf
import chess.pgn
import pgn_tensors_utils
import bz2

def read_data(data_path, num_valids=20000):
  print("-" * 80)
  print("Reading data")

  nb_games = 200
  #nb_games = sys.maxsize
  boards, labels, results = {}, {}, {}

  train_files = [
    "pgn_games/chess_games_2000.pgn",
    "pgn_games/chess_games_2001.pgn",
    "pgn_games/chess_games_2002.pgn",
    "pgn_games/chess_games_2003.pgn",
    "pgn_games/chess_games_2004.pgn"
  ]
  test_file = [    
    "pgn_games/chess_games_2005.pgn"
  ]
  boards["train"], labels["train"], results["train"] = load_data(data_path, train_files, nb_games)

  num_valids = int(len(boards["train"])*0.1)

  print(num_valids)
  
  if num_valids:
    boards["valid"] = boards["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]
    results["valid"]= results["train"][-num_valids:]

    boards["train"] = boards["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
    results["train"]= results["train"][:-num_valids]
  else:
    boards["valid"], labels["valid"], results["valid"] = None, None, None

  boards["test"], labels["test"], results["test"] = load_data(data_path, test_file, nb_games)
  
  return boards, results

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
  name = pgn.name.split('/')[3].replace('.pgn','')
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
  

def _load_data(data_path,pgn_path,game_nb):
  '''
    Load the tensors, the according labels and store them in a .npy file
    '''
  suffixe = pgn_path.split('/')    
  suffixe = suffixe[1].replace('.pgn','')
  tensors_file_exist = os.path.isfile(data_path+'/tensors/tensors_numpy_{}games_{}.npy'.format(game_nb,suffixe))
  labels_files_exist = os.path.isfile(data_path+'/tensors/labels_numpy_{}games_{}.npy'.format(game_nb,suffixe))

    
  print("Loading data for {}".format(suffixe))
    
  if tensors_file_exist == False or labels_files_exist == False:      
      full_name = os.path.join(data_path,pgn_path)
      games = load_games_from_pgn_path(full_name,game_nb)
      print("loading tensors and according labels ...")
      tensors,labels,results = pgn_tensors_utils.tensors_labels_from_games(games)
      np.save(data_path+'/tensors/labels_numpy_{}games_{}.npy'.format(game_nb,suffixe), labels)
      np.save(data_path+'/tensors/tensors_numpy_{}games_{}.npy'.format(game_nb,suffixe), tensors)
      np.save(data_path+'/tensors/results_numpy_{}games_{}.npy'.format(game_nb,suffixe), results)
  else:
      tensors = np.load(data_path+'/tensors/tensors_numpy_{}games_{}.npy'.format(game_nb,suffixe))
      labels  = np.load(data_path+'/tensors/labels_numpy_{}games_{}.npy'.format(game_nb,suffixe))      
      results = np.load(data_path+'/tensors/results_numpy_{}games_{}.npy'.format(game_nb,suffixe))
  return tensors,labels,results

def load_data(data_path,paths,game_nb):
  print("Loading data ...")
  tensors = []
  labels  = []
  results = []
  for pgn_path in paths:
      t,l,r = _load_data(data_path,pgn_path,game_nb)
      for row in t:
        tensors.append(row)
      for row in l:
        labels.append(row)
      for row in r:
        results.append(row)
        
  tensors = np.asarray(tensors)
  labels  = np.asarray(labels)
  results = np.asarray(results)
  
  tensors = np.concatenate(tensors, axis=0)
  tensors = np.reshape(tensors, [-1, 7, 8, 8])
  tensors = np.transpose(tensors, [0, 2, 3, 1])
  
  return tensors,labels,results

def select_random_examples(tensors,labels,nb_ex):
    if len(tensors) != len(labels) :
        raise('Tensors and labels have different length')
    else:
      samples = np.random.choice(len(tensors), size=nb_ex, replace=False)
      x = tensors[samples]
      y = labels[samples]
    return x,y
