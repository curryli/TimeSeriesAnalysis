# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import os                          #python miscellaneous OS system tool
from collections import Counter
import time, datetime
from sklearn import preprocessing
from dateutil import parser


def rounding(str):
    return round(float(str))

if __name__ == '__main__':
    Trans_df = pd.read_csv("df_save.csv", sep=",", low_memory=False, error_bad_lines=False)
    Trans_df["total_trans_cnt"] = Trans_df['total_trans_cnt'].map(lambda x: rounding(str(x)))
    Trans_df["cloudpay_trans_cnt"] = Trans_df['cloudpay_trans_cnt'].map(lambda x: rounding(str(x)))
    Trans_df.to_csv("forecast_rounded.csv", index=False)
