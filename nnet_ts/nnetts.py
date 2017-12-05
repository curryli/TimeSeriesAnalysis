#pip install nnet-ts
#from nnet_ts import *
from TimeSeriesNnet import TimeSeriesNnet

time_series = np.array(pd.read_csv("JLF_agg.csv")["settle_trans_at-count"])
neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
neural_net.fit(time_series, lag = 20, epochs = 10000)
neural_net.predict_ahead(n_ahead = 10)
import matplotlib.pyplot as plt
plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("Box & Jenkins AirPassenger data")
plt.xlabel("Observation ordered index")
plt.ylabel("No. of passengers")
plt.legend()
plt.show()