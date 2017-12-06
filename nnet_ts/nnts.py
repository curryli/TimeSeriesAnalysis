#pip install nnet-ts
from nnet_ts import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd

time_series = np.array(pd.read_csv("AirPassengers.csv")["x"])
neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
neural_net.fit(time_series, lag = 40, epochs = 10000)
neural_net.predict_ahead(n_ahead = 80)
import matplotlib.pyplot as plt
plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("Box & Jenkins AirPassenger data")
plt.xlabel("Observation ordered index")
plt.ylabel("No. of passengers")
plt.legend()
plt.show()