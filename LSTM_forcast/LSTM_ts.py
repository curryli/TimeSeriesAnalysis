#pip install nnet-ts
#from nnet_ts import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
from TimeSeries_LSTM import TimeSeries_LSTM

time_series = np.array(pd.read_csv("JLF_agg.csv")["settle_trans_at-count"])
neural_net = TimeSeries_LSTM()
neural_net.fit(time_series, lag = 20, epochs = 2000)
neural_net.predict_ahead(n_ahead = 10)
import matplotlib.pyplot as plt
plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("Box & Jenkins AirPassenger data")
plt.xlabel("Observation ordered index")
plt.ylabel("No. of passengers")
plt.legend()
plt.savefig("JLF_nnets_50.jpg")
#plt.show()