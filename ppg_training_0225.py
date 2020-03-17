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


import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

#Training_Data_Lin = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Lin_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Wu = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Wu_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Su = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Su_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Li = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Li_5_cycle.txt', header = None, delim_whitespace = True)


Training_Data_Lin = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Lin_Norm.txt', header = None, delim_whitespace = True)
Training_Data_Wu = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Wu_Norm.txt', header = None, delim_whitespace = True)
Training_Data_Su = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Liu_Norm.txt', header = None, delim_whitespace = True)
Training_Data_Li = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Kuo_Norm.txt', header = None, delim_whitespace = True)
training_data_num=1500
neuron_num=300

Training_Data_Lin=np.array(Training_Data_Lin)
Training_Data_Lin=Training_Data_Lin.tolist()
Training_Data_Wu=np.array(Training_Data_Wu)
Training_Data_Wu=Training_Data_Wu.tolist()
Training_Data_Su=np.array(Training_Data_Su)
Training_Data_Su=Training_Data_Su.tolist()
Training_Data_Li=np.array(Training_Data_Li)
Training_Data_Li=Training_Data_Li.tolist()


#print(Training_Data_Lin)
#Training_Data_Wu=Training_Data_Wu[:len(Training_Data_Lin)]
#Training_Data_Su=Training_Data_Su[:len(Training_Data_Lin)]
#Training_Data_Li=Training_Data_Li[:len(Training_Data_Lin)]



# Merge the training data  
input_data, output_data = [], []
input_data =Training_Data_Lin+Training_Data_Wu+Training_Data_Su+Training_Data_Li

print(len(input_data))
for num in range(len(Training_Data_Lin)):
    output_data.append(0)
for num in range(len(Training_Data_Wu)):
    output_data.append(1)
for num in range(len(Training_Data_Su)):
    output_data.append(2)
for num in range(len(Training_Data_Li)):
    output_data.append(3)



# Convert training data to numpy array
x = np.array(input_data)
y = np.array(output_data)


# Shuffle the training data
randomize = np.arange(len(x))
np.random.shuffle(randomize)

#normalizw
#x=x/100
#minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
#x=minmax_scale.fit_transform(x);

x = x[randomize]
y = y[randomize]
init_y=y;
init_y=init_y[training_data_num:]

print(y)
y= to_categorical(y)
print(y)
print(len(y))

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
model = Sequential()
        #將模型疊起
model.add(Dense(input_dim=4 ,units=neuron_num,activation='relu'))
model.add(Dense(units=neuron_num,activation='relu'))

model.add(Dense(units=neuron_num,activation='relu'))

model.add(Dense(units=4,activation='softmax'))
model.summary()


#adam = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss = 'mean_squared_error', optimizer = adam, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 1, epochs = 50, validation_data = (x_test, y_test))
print(history.history.keys())

score = model.evaluate(x_test, y_test)



print('\nTotal loss on testing set:', score[0])
print('Accuracy of testing set:', score[1])

#confusion matrix
'''
print(prediction)
print(len(prediction))
print(init_y)
print(len(init_y))
'''




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
model.save('my_model.h5')