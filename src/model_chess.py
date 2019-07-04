import sys
import os
import tensorflow as tf

from data_loader import *
from pgn_tensors_utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from pgn_tensors_utils import create_uci_labels

DEVICE_NAME = "/device:GPU:0"

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.90
sess = tf.Session(config=config)

class model_chess():
    def __init__(self,
                 tensors,
                 labels,
                 batch_size,
                 mini_batch_size,
                 epoch_nb
    ):
        print("Building model ...")
        self.tensors         = tensors
        self.labels          = labels 
        self.batch_size      = batch_size
        self.mini_batch_size = mini_batch_size
        self.epoch_nb        = epoch_nb
        self.uci_labels = create_uci_labels()
        
        
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
        self.model = model
            
    def sample(self,sample_size,train_size):
        x, y = select_random_examples(self.tensors,self.labels,sample_size)
        y = np.asarray([one_hot_encoded_label(y,self.uci_labels) for y in y])
        
        x_train = x[0:len(x)-int(len(x)/train_size),:]
        y_train = y[0:len(y)-int(len(y)/train_size)]
        
        x_test = x[len(x)-int(len(x)/train_size):,:]
        y_test = y[len(y)-int(len(y)/train_size):]
        return x_train, y_train, x_test, y_test
    
    def train(self,pool_size):
        print('Training ...')
        for i in range(pool_size):
            print('Pool {}/{} :'.format(i,pool_size))
            x_train,y_train,x_test,y_test = self.sample(self.batch_size,5)
            self.model.fit(x_train,y_train,batch_size=self.mini_batch_size,validation_data=(x_test,y_test),epochs=self.epoch_nb)
        self.save()
 
    def save(self):
        model_json = self.model.to_json()
        with open("./data/models/model.json",'r'):
            json_file.write(model_json)
        self.model.save_weights("./data/models/model.h5")
        print("Saved model to disk")

    @staticmethod
    def load(self):
        json_file = open("./data/models/model.json",'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        self.model = loaded_model

        return loaded_model
