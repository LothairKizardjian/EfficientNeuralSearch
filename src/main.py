from data_loader import load_data
from pgn_tensors_utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import LabelBinarizer
from pgn_tensors_utils import create_uci_labels

nb_games = 5000
pgn_path = '../PGN_chess_games/chess_games.pgn'
tensors,labels = load_data(pgn_path,nb_games)

x_train = tensors[0:len(tensors)-int(len(tensors)/3),:]
y_train = labels[0:len(labels)-int(len(labels)/3)]

x_test = tensors[len(tensors)-int(len(tensors)/3):,:]
y_test = labels[len(labels)-int(len(labels)/3):]

#One-hot encoding
uci_labels = create_uci_labels()
y_train = np.asarray([one_hot_encoded_label(y,uci_labels) for y in y_train])
y_test  = np.asarray([one_hot_encoded_label(y,uci_labels) for y in y_test])


#creating the model
input_boards = Input(shape=(7,8,8))

x = Reshape((7,8,8))(input_boards) # batch_size  x board_x x board_y x 1
h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='same', use_bias=False)(x)))  # batch_size  x board_x x board_y x num_channels
h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='same', use_bias=False)(h_conv1)))  # batch_size  x board_x x board_y x num_channels
h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv2))) # batch_size  x (board_x-2) x (board_y-2) x num_channels
h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv3))) # batch_size  x (board_x-4) x (board_y-4) x num_channels
h_conv4_flat = Flatten()(h_conv4)       
s_fc1 = Dropout(0.3)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
s_fc2 = Dropout(0.3)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
output = Dense(len(create_uci_labels()), activation='softmax', name='output')(s_fc2)   # batch_size x self.action_size

model = Model(inputs=input_boards, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)



