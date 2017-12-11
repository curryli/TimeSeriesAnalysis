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

pathDir = os.listdir("split_test/")
for subname in pathDir:
    child = os.path.join('%s%s' % ("split_test/", subname))
    testfile = child.decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
    print testfile
    temp_df = pd.read_csv(testfile)
    division_nm = temp_df[['division_nm']].iloc[0, 0]
    mchnt_brand = temp_df[['mchnt_brand']].iloc[0, 0]
    print division_nm, mchnt_brand


######################################
    time_series = np.array(temp_df["total_trans_cnt"])
    neural_net = TimeSeriesNnet(hidden_layers=[20, 15, 5], activation_functions=['sigmoid', 'sigmoid', 'sigmoid'])
    neural_net.fit(time_series, lag=20, epochs=100)
    forecast_total = neural_net.predict_ahead(n_ahead=5)
    forecast_total_se = Series(forecast_total, index=[0, 1, 2, 3, 4])
######################################
    time_series = np.array(temp_df["cloudpay_trans_cnt"])
    neural_net = TimeSeriesNnet(hidden_layers=[20, 15, 5], activation_functions=['sigmoid', 'sigmoid', 'sigmoid'])
    neural_net.fit(time_series, lag=20, epochs=100)
    forecast_cloud = neural_net.predict_ahead(n_ahead=5)
    forecast_cloud_se = Series(forecast_cloud, index=[0, 1, 2, 3, 4])
######################################


    tmp_save = DataFrame(index=[0, 1, 2, 3, 4], columns=['division_nm', 'mchnt_brand'])
    tmp_save['division_nm'] = division_nm
    tmp_save['mchnt_brand'] = mchnt_brand
    tmp_save['settle_dt'] = dates
    tmp_save['total_trans_cnt'] = forecast_total_se
    tmp_save['cloudpay_trans_cnt'] = forecast_cloud_se


    df_save = pd.concat([df_save, tmp_save[['division_nm', 'mchnt_brand', 'settle_dt', 'total_trans_cnt', 'cloudpay_trans_cnt']]], axis=0)
df_save.to_csv("df_save.csv", index=False)



