# -*- coding: utf-8 -*-

# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import Series
import pandas as pd
import numpy as np

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
 
train = dataset
# reshape into X=t and Y=t+1
look_back = 30
trainX, trainY = create_dataset(train, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2)
# make predictions


tmpdata = scaler.inverse_transform(dataset)
    
for i in range(5):
    testX = trainX[-10:]
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    testPredict = model.predict(testX)
     
    predata = scaler.inverse_transform(testPredict)

    tmpdata = np.vstack((tmpdata,predata))
    
    trainX = np.vstack((trainX,testX))
    
 
lastdata = tmpdata
# plot baseline and predictions
plt.plot(lastdata)
 
#plt.savefig('predict_shift1.jpeg')
plt.show()
