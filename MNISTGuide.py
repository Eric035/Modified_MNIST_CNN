from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(64,64,1),activation='relu'))#16
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu'))#16
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))#0.25
# model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',activation='relu'))#16
# model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))#36
#model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))#duo not improve
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))#36
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))#0.25

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
print(model.summary())



# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),       
#                  activation='relu',padding='same',
#                  input_shape=(64,64,1)))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# print(model.summary())

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import csv
# print(os.listdir("./input"))
np.random.seed(10)

import pandas as pd
train_images_raw = pd.read_pickle('./input/train_images.pkl')
train_labels_raw = pd.read_csv('./input/train_labels.csv')
test_images_raw = pd.read_pickle('./input/test_images.pkl')
print("train_images_raw.shape: ",train_images_raw.shape)

print("train_labels_raw.shape: ",train_labels_raw.shape)

t_lables=[0]*len(train_labels_raw)


#print(train_labels_raw)
# 	t_lables[i]=train_labels_raw[i][1]

for i in range(len(train_labels_raw)):
    t_lables[i]=train_labels_raw.at[i, 'Category']
#print(t_lables)

train_images, test_images, y_train_label, y_test_label = train_test_split(train_images_raw, t_lables, test_size=0.2, shuffle=False)


x_Train4D =train_images.reshape(32000,64,64,1).astype('float32')
x_Test4D = test_images.reshape(8000,64,64,1).astype('float32')

x_Real_Test=test_images_raw.reshape(10000,64,64,1).astype('float32')

#print(x_Train4D)
print ('x_train:',x_Train4D.shape)
print ('x_test:',x_Test4D.shape)

x_Train4D_normalize = x_Train4D/ 255
x_Test4D_normalize = x_Test4D/ 255
x_Real_Test_normalize=x_Real_Test/ 255
#print(type(x_Train_normalize))
#print(y_train_lable[:5])
#print('y_train_lable: ',y_train_label)

#print('-------------------------------------')

y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

#print(y_TrainOneHot[:3])


batch_size = 1024
epochs = 50

checkpointer_1 = ModelCheckpoint(filepath="weights-{epoch:02d}.hdf5", verbose=1, save_best_only=False, period=10)##


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=False)
#model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])


train_history = model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[checkpointer_1])##
model.save_weights('cnn_drop4096_weights.h5')###



import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x_Test4D_normalize,y_TestOneHot)
print()
print('loss=',scores[0])
print('accuracy=',scores[1])

prediction = model.predict_classes(x_Real_Test_normalize)

print(prediction)


prediction = pd.DataFrame(prediction, columns=['Category']).to_csv('prediction.csv')

# seq_num = range(len(x_Real_Test))
# with open('predic2.csv','w+') as predict_writer:
#     predict_writer.writelines('Id,Label\n')
#     for test_num in seq_num:          
#         prediction = prediction[0].tolist()
#         label = reader.num_list[prediction.index(max(prediction))]
#         predict_writer.writelines(str(test_num+1) + ',' + str(label) + '\n')


# with open('cnm.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows([['Id','Category']])

 
#     for i in range(10000):
#          writer.writerows([[i,prediction[i]]])




