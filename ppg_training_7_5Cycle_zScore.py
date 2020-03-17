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

#Training_Data_One = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Lin_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Two = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Wu_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Three = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Su_5_cycle.txt', header = None, delim_whitespace = True)
#Training_Data_Four = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\4_feature\PPG_Training_Li_5_cycle.txt', header = None, delim_whitespace = True)


Training_Data_One = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Lin_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Two = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Wu_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Three = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Liu_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Four = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Kuo_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Five = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Wang_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Six = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Li_5Cycle_zScore.txt', header = None, delim_whitespace = True)
Training_Data_Seven = pd.read_csv(r'C:\Users\Robinson\Documents\MATLAB\Data\20200225TrainData\Su_5Cycle_zScore.txt', header = None, delim_whitespace = True)
#training_data_num=11000
neuron_num=300

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


training_data_num = int( round(len(output_data)*0.8) )
print('Num of data :')
print(training_data_num)


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

#print(y)
y= to_categorical(y)
#print(y)
print('Total num is: ')
print(len(y))

x = x.astype('float32')
y = y.astype('float32')



# Split traing data for training and testing
x_train, y_train = x[:training_data_num], y[:training_data_num]
x_test, y_test = x[training_data_num:], y[training_data_num:]

'''
#PCA------------------------------
pca = PCA(n_components=70)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)
print('训练集数据的原始维度是：{}'.format(x_train.shape))
print('PCA降维后训练集数据是：{}'.format(x_train_reduced.shape))
x_train=x_train_pca
x_test=x_test_pca
'''

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
model.add(Dense(input_dim=20 ,units=neuron_num,activation='relu'))
model.add(Dense(units=neuron_num,activation='relu'))

model.add(Dense(units=neuron_num,activation='relu'))

model.add(Dense(units=7,activation='softmax'))
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