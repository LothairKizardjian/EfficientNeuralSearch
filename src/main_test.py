from data_loader import*
import time
import bz2
import sys

start_time = time.time()

paths = []
paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('00'))
paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('01'))


t,l = load_data_from_multiple_files(paths,sys.maxsize)

print(t.shape)
print(l.shape)

print("--- %s seconds ---" % (time.time() - start_time))
