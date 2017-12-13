# -*- coding: utf-8 -*-
# pip install nnet-ts
# from nnet_ts import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
from TimeSeriesNnet import TimeSeriesNnet
import os
from pandas import DataFrame, Series

# 遍历指定目录，显示目录下的所有文件名
df_save = DataFrame(columns=['division_nm', 'mchnt_brand', 'settle_dt', 'total_trans_cnt', 'cloudpay_trans_cnt'])
dates = pd.date_range(str("20171208"), periods=5, freq='D')

total_losses = {}
cloud_losses = {}
pathDir = os.listdir("splitted/")
for subname in pathDir:
    child = os.path.join('%s%s' % ("splitted/", subname))
    testfile = child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
    print testfile
    temp_df = pd.read_csv(testfile)
    division_nm = temp_df[['division_nm']].iloc[0, 0]
    mchnt_brand = temp_df[['mchnt_brand']].iloc[0, 0]
    print division_nm, mchnt_brand

######################################
    time_series = np.array(temp_df["total_trans_cnt"])
    neural_net = TimeSeriesNnet(hidden_layers=[20, 15, 5], activation_functions=['sigmoid', 'sigmoid', 'sigmoid'])
    neural_net.fit(time_series, lag=20, epochs=10000)
    total_losses[subname] = neural_net.evaluate()


######################################
    time_series = np.array(temp_df["cloudpay_trans_cnt"])
    neural_net = TimeSeriesNnet(hidden_layers=[20, 15, 5], activation_functions=['sigmoid', 'sigmoid', 'sigmoid'])
    neural_net.fit(time_series, lag=20, epochs=10000)
    cloud_losses[subname] = neural_net.evaluate()

######################################
import json
f = open("total_losses.json", "w")
json.dump(total_losses, f, ensure_ascii=False)
f.close()

f = open("cloud_losses.json", "w")
json.dump(cloud_losses, f, ensure_ascii=False)
f.close()



