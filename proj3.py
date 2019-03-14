
# COMP 551 - Project 3

import os
import numpy as np
import pandas
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
clear_session()

# Set Seed
np.random.seed(256)


#--------------------------DATA PREPARATION--------------------------------------

# Extract Data
raw_test_images = pandas.read_pickle('./data/testimages/test_images.pkl')
raw_train_images = pandas.read_pickle('./data/trainimages/train_images.pkl')
raw_train_labels = pandas.read_csv('./data/train_labels.csv')
train_labels = [0] * len(raw_train_labels)
for i in range(len(raw_train_labels)):
    train_labels[i] = raw_train_labels.at[i, 'Category']

print("raw_train_images.shape: ", raw_train_images.shape)
print("raw_train_labels.shape: ", raw_train_labels.shape)

# Get validation split
train_images, val_images, y_train_label, y_val_label = train_test_split(raw_train_images, train_labels, test_size=0.2, shuffle=False)

# Reshape for CNN
x_train_input_temp = train_images.reshape(32000,64,64,1).astype('float32')
x_val_input_temp = val_images.reshape(8000,64,64,1).astype('float32')
x_test_input_temp = raw_test_images.reshape(10000,64,64,1).astype('float32')

print ('x_train:', x_train_input_temp.shape)
print ('x_test:', x_val_input_temp.shape)

# Normalize data
x_train_input = x_train_input_temp / 255
x_val_input = x_val_input_temp / 255
x_test_input = x_test_input_temp / 255

y_train_label = to_categorical(y_train_label)
y_val_label = to_categorical(y_val_label)


#--------------------------NETWORK DESIGN--------------------------------------

'''
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
'''

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(64,64,1),activation='relu'))
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#--------------------------MODEL EXECUTION--------------------------------------

batch_size = 512
epochs = 50

#model.fit(x_train_input, y_train_label, validation_data=(x_val_input, y_val_label), epochs=3)

checkpointer = ModelCheckpoint(filepath = "weights_at_epoch_{epoch:02d}.hdf5", verbose = 1, save_best_only = False, period = 10)

model.fit(x = x_train_input, y = y_train_label, validation_data=(x_val_input, y_val_label), epochs = epochs, verbose = 1, batch_size = batch_size, callbacks = [checkpointer])

prediction = model.predict_classes(x_test_input)

prediction = pandas.DataFrame(prediction, columns=['Category']).to_csv('test_prediction.csv')







