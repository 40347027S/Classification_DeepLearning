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

#Training_Data_One = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Lin_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Two = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Wu_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Three = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Su_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Four = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Li_5_cycle.txt', header = None, delim_whitespace = True)


Training_Data_One = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\1_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Two = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\2_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Three = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\3_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Four = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\4_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Five = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\5_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Six = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\6_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
Training_Data_Seven = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\Bid\7_PPGFilter_Fourier_1000.txt', header = None, delim_whitespace = True)
training_data_num=3500
neuron_num=1024
raw_ppg_len=100
raw_ppg_width=6

Training_Data_One=np.array(Training_Data_One)
Training_Data_One=Training_Data_One.tolist()
Training_Data_Two=np.array(Training_Data_Two)
Training_Data_Two=Training_Data_Two.tolist()
Training_Data_Three=np.array(Training_Data_Three)
Training_Data_Three=Training_Data_Three.tolist()
Training_Data_Four=np.array(Training_Data_Four)
Training_Data_Four=Training_Data_Four.tolist()

Training_Data_Five=np.array(Training_Data_Five)
Training_Data_Five=Training_Data_Five.tolist()
Training_Data_Six=np.array(Training_Data_Six)
Training_Data_Six=Training_Data_Six.tolist()
Training_Data_Seven=np.array(Training_Data_Seven)
Training_Data_Seven=Training_Data_Seven.tolist()


#print(Training_Data_One)
#Training_Data_Two=Training_Data_Two[:len(Training_Data_One)]
#Training_Data_Three=Training_Data_Three[:len(Training_Data_One)]
#Training_Data_Four=Training_Data_Four[:len(Training_Data_One)]



# Merge the training data  
input_data, output_data = [], []
input_data =Training_Data_One+Training_Data_Two+Training_Data_Three+Training_Data_Four+Training_Data_Five+Training_Data_Six+Training_Data_Seven

print(len(input_data))
for num in range(len(Training_Data_One)):
    output_data.append(0)
for num in range(len(Training_Data_Two)):
    output_data.append(1)
for num in range(len(Training_Data_Three)):
    output_data.append(2)
for num in range(len(Training_Data_Four)):
    output_data.append(3)
for num in range(len(Training_Data_Five)):
    output_data.append(4)
for num in range(len(Training_Data_Six)):
    output_data.append(5)
for num in range(len(Training_Data_Seven)):
    output_data.append(6)



# Convert training data to numpy array
x = np.array(input_data)
y = np.array(output_data)


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
model.add(Dense(7, activation='softmax'))
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
