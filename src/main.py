import sys

from data_loader import *
from pgn_tensors_utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import LabelBinarizer
from pgn_tensors_utils import create_uci_labels

DEVICE_NAME = '/device:GPU:0'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

nb_games = sys.maxsize

paths = []

paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('00'))
#paths.append("../PGN_chess_games/chess_games_20{}.pgn".format('01'))
print('1')
uci_labels = create_uci_labels()
print('2')
tensors,labels = load_data_from_multiple_files(paths,nb_games)
print(tensors.shape)
print(labels.shape)
#One_hot encoding
print('3')
labels = np.asarray([one_hot_encoded_label(y,uci_labels) for y in labels])


#creating the model

with tf.device(DEVICE_NAME):
    input_boards = Input(shape=(7,8,8))
    x = Reshape((7,8,8))(input_boards) 
    h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='same', use_bias=False)(x)))  
    h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='same', use_bias=False)(h_conv1)))  
    h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv2))) 
    h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(512, 3, padding='valid', use_bias=False)(h_conv3)))
    h_conv4_flat = Flatten()(h_conv4)       
    s_fc1 = Dropout(0.3)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  
    s_fc2 = Dropout(0.3)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))
    output = Dense(len(create_uci_labels()), activation='softmax', name='output')(s_fc2)  
    
    model = Model(inputs=input_boards, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


#training
with  tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    nb_train = 100
    nb_ex    = int(len(tensors)/nb_train)
    for i in range(nb_train):
        print('Training pool {}/{} ...'.format(i+1,nb_train))
        x,y = select_random_examples(tensors,labels,nb_ex)
        print(x.shape)
        print(y.shape)
        x_train = x[0:len(x)-int(len(x)/5),:]
        y_train = y[0:len(y)-int(len(y)/5)]
        
        x_test = x[len(x)-int(len(x)/5):,:]
        y_test = y[len(y)-int(len(y)/5):]
        
        model.fit(x_train, y_train,batch_size=50, validation_data=(x_test, y_test), epochs=15)

model_json = model.to_json()
with open("./data/models/model.json",'r'):
    json_file.write(model_json)
model.save_weihts("./data/models/model.h5")
print("Saved model to disk")
