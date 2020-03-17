from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from pandas import Series, DataFrame
from numpy.random import randn
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix


from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten


import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

total_num = 30
training_data_num=3500
neuron_num=1024
raw_ppg_len=400
raw_ppg_width=1

four_feature = '_training_data_ppg.txt'
dtrend = '_PPGDtrend.txt'
dtrend_fourier = '_PPGFilter_Fourier_1000.txt'
file_type = dtrend

input_data, output_data = [], []
check = 0
check_num = 0

for num in range(total_num):
    file_string = str(num+1) + file_type
    #print(file_string+'\n')
    Training_Data = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid'+'\/' +file_string, header = None, delim_whitespace = True)
    Training_Data = np.array(Training_Data)
    check_num += len(Training_Data)
    if check ==0:
        input_data = Training_Data
        input_data = np.array(input_data)
        
    if check != 0:
        #input_data = np.append( input_data, input_data)
        input_data = np.vstack((input_data, Training_Data))
    check = 9999
    
    for tmp in range(len(Training_Data)):
        output_data.append((num))
    

# Convert training data to numpy array
x = np.array(input_data)
y = np.array(output_data)

training_data_num = int( round(len(output_data)*0.8) )
print('Num of data :')
print(training_data_num)


# Shuffle the training data
randomize = np.arange(len(x))
np.random.shuffle(randomize)

#x=x/2000

#normalizw
#x=x/100
#minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
#x=minmax_scale.fit_transform(x);

x = x[randomize]
y = y[randomize]

init_y=y;
init_y=init_y[training_data_num:]
print('Totak num is: ')
print(len(y))

#print(y)
y= to_categorical(y)
#print(y)
#print(len(y))

x = x.astype('float32')
y = y.astype('float32')



# Split traing data for training and testing
x_train, y_train = x[:training_data_num], y[:training_data_num]
x_test, y_test = x[training_data_num:], y[training_data_num:]

'''
# Build model
model = Sequential()
model.add(Dense(input_dim=3,units=100,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
''' 
print(len(x_train))
print('X_Train:', x_train.shape[0])
print('X_Train:', x_train.shape)
print(x_train)
x_train=x_train.reshape(x_train.shape[0], raw_ppg_len, raw_ppg_width ,1)
print('X_Train:', x_train.shape)
print(x_train)
x_test=x_test.reshape(x_test.shape[0], raw_ppg_len, raw_ppg_width ,1)
model = Sequential()
        #將模型疊起
model.add(Conv2D(filters=16,kernel_size=(5,1),padding='same',input_shape=(raw_ppg_len,raw_ppg_width,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Conv2D(filters=36,kernel_size=(5 ,1),padding='same',input_shape=(raw_ppg_len,raw_ppg_width,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(total_num, activation='softmax'))
model.summary()



#model.compile(loss = 'mean_squared_error', optimizer = adam, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

history = model.fit(x_train, y_train,validation_split=0.2, batch_size = 1, epochs = 50, verbose=2)

print(history.history.keys())

score = model.evaluate(x_test, y_test)



print('\nTotal loss on testing set:', score[0])
print('Accuracy of testing set:', score[1])

#confusion matrix




# summarize history for accuracy

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

prediction = model.predict_classes(x_test)
init_data_frame=pd.crosstab(init_y, prediction, rownames=['label'], colnames=['predict'])
data_frame=pd.crosstab(init_y, prediction, rownames=['label'], colnames=['predict'],normalize=1)
print(init_data_frame)
sns.heatmap(data_frame, cmap='YlOrRd', annot=True)


