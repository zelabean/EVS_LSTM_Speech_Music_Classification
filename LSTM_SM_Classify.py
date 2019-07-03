from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Embedding,LSTM
import numpy as np
import math
import tensorflow as tf
import pandas as pd
import json

#################################################### Hyperparameters ###################################################
step = 20
learning_rate = 0.00006
epoch = 500
batch = 2048
loss_function = 'binary_crossentropy'
activation_function = 'sigmoid'
optim = tf.keras.optimizers.Adam(lr=learning_rate)
########################################################################################################################
####################################################     Define    #####################################################
mode = 2
index = ['f_01','f_02','f_03','f_04','f_05','f_06','f_07','f_08','f_09','f_10','f_11','f_12','label']
feature_index =  ['f_01','f_02','f_03','f_04','f_05','f_06','f_07','f_08','f_09','f_10','f_11','f_12']
label_index = ['label']
########################################################################################################################
####################################################   Functions   #####################################################

def cutting(dataframe, step):
    n = math.floor(len(dataframe)/step)
    dataframe = dataframe[0:n*step][:]
    return dataframe

def split_train_test(dataframe, ratio):
    length = int(len(dataframe) * ratio)
    train = dataframe[length:][:]
    test = dataframe[0:length][:]
    return train, test

def feature_label_split(set):
    feature = set[feature_index]
    label = set[label_index]
    feature = np.array(feature)
    label = np.array(label)
    feature = np.reshape(feature, (-1, step, len(feature_index)))
    label = np.reshape(label, (-1, step, len(label_index)))
    return feature, label

########################################################################################################################
##################################################### Data loading #####################################################
print('Loading data with Pandas...')
clean = pd.read_csv('./dataset/clean.txt', sep='/',names =index)
babble = pd.read_csv('./dataset/babble.txt', sep='/',names =index)
car = pd.read_csv('./dataset/car.txt', sep='/',names =index)
factory = pd.read_csv('./dataset/factory.txt', sep='/',names =index)
white = pd.read_csv('./dataset/white.txt', sep='/',names =index)
music = pd.read_csv('./dataset/others.txt', sep='/',names =index)
classic = pd.read_csv('./dataset/classic.txt', sep='/',names =index)
print('All data loaded.')

print('Dividing each data for train and evaluate')
clean_tr, clean_te = split_train_test(clean, 0.2)
babble_tr, babble_te = split_train_test(babble, 0.2)
car_tr, car_te = split_train_test(car, 0.2)
fac_tr, fac_te = split_train_test(factory, 0.2)
wt_tr, wt_te = split_train_test(white, 0.2)
msc_tr, msc_te = split_train_test(music, 0.2)
cls_tr, cls_te = split_train_test(classic, 0.2)
print('All data divided')
print('Cutting data for reshape...')
clean_te, clean_tr = cutting(clean_te, step),cutting(clean_tr, step)
babble_te, babble_tr = cutting(babble_te, step),cutting(babble_tr, step)
car_te, car_tr = cutting(car_te, step),cutting(car_tr, step)
fac_te, fac_tr = cutting(fac_te, step),cutting(fac_tr, step)
wt_te, wt_tr = cutting(wt_te, step),cutting(wt_tr, step)
msc_te, msc_tr = cutting(msc_te, step),cutting(msc_tr, step)
cls_te, cls_tr = cutting(cls_te, step),cutting(cls_tr, step)
print('Cut data.')
print('Merging data...')
dataframe = clean_tr
dataframe = pd.concat([dataframe,babble_tr])
dataframe = pd.concat([dataframe,car_tr])
dataframe = pd.concat([dataframe,fac_tr])
dataframe = pd.concat([dataframe,wt_tr])
dataframe = pd.concat([dataframe,msc_tr])
dataframe = pd.concat([dataframe,cls_tr])
print('data Merged.')
########################################################################################################################
################################################## Data preprocessing ##################################################
print('Splitting feature & label')
data_x, data_y = feature_label_split(dataframe)
clean_x,clean_y = feature_label_split(clean_te)
babble_x,babble_y = feature_label_split(babble_te)
car_x,car_y = feature_label_split(car_te)
factory_x,factory_y = feature_label_split(fac_te)
white_x,white_y = feature_label_split(wt_te)
music_x,music_y = feature_label_split(msc_te)
classic_x,classic_y = feature_label_split(cls_te)
data_y = data_y - 1
clean_y = clean_y - 1
babble_y = babble_y - 1
car_y = car_y - 1
factory_y = factory_y - 1
white_y = white_y - 1
music_y = music_y - 1
classic_y = classic_y - 1
print('Splitted')
########################################################################################################################
####################################################### Modeling #######################################################
print('Modeling...')
model = Sequential()
model.add(LSTM(200,input_shape=(step,12),return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(LSTM(600,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(LSTM(200,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1,activation=activation_function))
model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
model.summary()
print('Training...')
history = model.fit(data_x, data_y, epochs=epoch, verbose=mode, batch_size=batch)
########################################################################################################################
###################################################### Evaluationg #####################################################
print('Evaluate model')
clean_loss, clean_acc = model.evaluate(clean_x, clean_y)
babble_loss, babble_acc = model.evaluate(babble_x, babble_y)
car_loss, car_acc = model.evaluate(car_x, car_y)
factory_loss, factory_acc = model.evaluate(factory_x, factory_y)
white_loss, white_acc = model.evaluate(white_x, white_y)
music_loss, music_acc = model.evaluate(music_x, music_y)
classic_loss, classic_acc = model.evaluate(classic_x, classic_y)
print("clean : " + str(clean_acc))
print("babble : " + str(babble_acc))
print("car : " + str(car_acc))
print("factory : " + str(factory_acc))
print("white : " + str(white_acc))
print("music : " + str(music_acc))
print("classic : " + str(classic_acc))
########################################################################################################################
###################################################### Save model ######################################################
print('Saving model')
history_dict = history.history
with open('./out/history.json', 'w') as outfile:
    json.dump(str(history_dict), outfile)
model.save('speech_music_classification_lstm_model.h5')
########################################################################################################################

