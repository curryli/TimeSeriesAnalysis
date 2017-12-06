import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
from keras.layers import Dense, Dropout
from keras.layers.core import Dense, Dropout, Activation,Flatten, Reshape
from keras.layers import Embedding, Masking
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class TimeSeries_LSTM(object):
	def __init__(self,  optimizer = 'Adam', loss = 'mean_squared_error'):
		self.optimizer = optimizer
		self.loss = loss

	def fit(self, timeseries, lag = 20, epochs = 1000, verbose = 2):
		self.timeseries = np.array(timeseries, dtype = "float64") # Apply log transformation por variance stationarity
		self.lag = lag
		self.n = len(timeseries)
		if self.lag >= self.n:
			raise ValueError("Lag is higher than length of the timeseries")
		self.X = np.zeros((self.n - self.lag, self.lag), dtype = "float64")
		self.y = np.log(self.timeseries[self.lag:])
		self.epochs = epochs
		self.scaler = StandardScaler()
		self.verbose = verbose

		logging.info("Building regressor matrix")
		# Building X matrix
		for i in range(0, self.n - lag):
			self.X[i, :] = self.timeseries[range(i, i + lag)]

		logging.info("Scaling data")
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)


		ts = 5
		size_data = self.X.shape[1]
		data_dim = size_data / ts

		logging.info("Checking network consistency")
		# Neural net architecture
		self.nn = Sequential()
		print size_data, ts, data_dim

		self.nn.add(Reshape((ts, data_dim), input_shape=(size_data,)))

		self.nn.add(LSTM(8, input_shape=(ts, data_dim), activation='sigmoid',return_sequences=True))
		#self.nn.add(Dropout(0.5))
		self.nn.add(LSTM(4, activation='sigmoid' ))
		#self.nn.add(Dropout(0.5))

		# Add final node
		self.nn.add(Dense(1))
		#self.nn.add(Activation('linear'))
		self.nn.compile(loss = self.loss, optimizer = self.optimizer)

		logging.info("Training neural net")
		# Train neural net
		self.nn.fit(self.X, self.y, nb_epoch = self.epochs, batch_size=1, verbose = self.verbose)

	def predict_ahead(self, n_ahead = 1):
		# Store predictions and predict iteratively
		self.predictions = np.zeros(n_ahead)

		for i in range(n_ahead):
			self.current_x = self.timeseries[-self.lag:]
			self.current_x = self.current_x.reshape((1, self.lag))
			self.current_x = self.scaler.transform(self.current_x)
			self.next_pred = self.nn.predict(self.current_x)
			self.predictions[i] = np.exp(self.next_pred[0, 0])
			self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0,:])), axis = 0)

		return self.predictions