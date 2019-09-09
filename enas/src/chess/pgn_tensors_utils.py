import numpy as np
import tensorflow as tf
import chess.pgn
import copy

def fen_to_tensor(fen):    
    pieces_str = "PRNBQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}
    #tensor = np.zeros((7,8,8)).astype('int32')
    tensor = np.zeros((7,8,8)).astype('float32')
    
    inputliste = fen.split(' ')
    row = 0
    col = 0
    for i, c in enumerate(inputliste[0]):
        if c in pieces:
            tensor[np.abs(pieces_dict[c])-1,row, col] = np.sign(pieces_dict[c])
            col = col + 1
        elif c == '/':  # new row
            row = row + 1
            col = 0
        elif int(c) in valid_spaces:
            col = col + int(c)
        else:
            raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))

    if inputliste[1] == "w":
        for i in range(8):
            for j in range(8):
                tensor[6,i,j] = 1
    else:
        for i in range(8):
            for j in range(8):
                tensor[6,i,j] = -1
  
    return tensor

def tensors_for_each_move(game):
    tensors = []
    board   = game.board()
    fen     = board.fen()
    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()
        tensors.append(fen_to_tensor(fen))
    return np.asarray(tensors)

def get_result_logits(game):
    result = game.headers["Result"]
    if result == "1-0": #white victory
        return np.array([1,0,0])
    elif result == "0-1": #black victory
        return np.array([0,1,0])
    else: #draw
        return np.array([0,0,1])
    
def get_result(game):
    result = game.headers["Result"]
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0
    
def tensors_labels_for_each_move(game):
    #returns a tensor corresponding to the board's actual state and the move done according to
    #this state, except for the last one since there's no more moves.

    tensors = []
    labels  = []
    results = []
    board   = game.board()
    uci     = create_uci_labels()

    for move in game.mainline_moves():
        index = uci.index(move.uci())
        labels.append(index)
        board.push(move)
        fen = board.fen()
        results.append(get_result(game))
        tensors.append(fen_to_tensor(fen))

    return np.asarray(tensors).astype('float32'), np.asarray(labels).astype('float32'), np.asarray(results).astype('float32')

def tensors_labels_from_games(games):
    tensors = []
    labels  = []
    results = []
    for game in games:
        tens, labs, res = tensors_labels_for_each_move(game)
        if len(tens)==len(labs):
            for i in range(len(tens)):
                tensors.append(tens[i])
                labels.append(labs[i])
                results.append(res[i])
    return np.asarray(tensors),np.asarray(labels).astype('float32'),np.asarray(results).astype('float32')

def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    This returns all the possible 'Queen moves' and 'Knight move' for each square plus the promotion
    of a pawn to either a rook, knight, bishop or queen from rank 7 or higher
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array
    
def flipped_uci_labels():
    """
    Seems to somehow transform the labels used for describing the universal chess interface format, putting
    them into a returned list.
    :return:
    """
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]

def one_hot_encoded_label(uci_label,uci_labels):
    label = np.zeros(len(uci_labels))

    for i in range(len(uci_labels)):
        if uci_label == uci_labels[i]:
            label[i] = 1
            break
    return label
        
