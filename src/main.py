import sys

from data_loader import *
from model_chess import *
#from game  import *

nb_games = sys.maxsize
paths = []

paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('00'))
#paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('01'))

tensors,labels = load_data_from_multiple_files(paths,nb_games)

print(tensors.shape)
print(labels.shape)

model = model_chess(tensors, labels, batch_size=10000, mini_batch_size=25, epoch_nb=10)
model.train(10)


