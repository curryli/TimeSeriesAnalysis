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

########################################group 函数########################################

def cnt_20(arr):
    cnt_pairs = Counter(arr)
    return cnt_pairs[20]
###################################################

###########################非group函数################################
def replace_Not_num(x):
    if ((type(x)==int) | (type(x)==float) | (type(x)==long)):
        return x
    else:
        try:
            result = long(x)
        except ValueError:
            result = -1
        return result

#####################################

if __name__ == '__main__':
    Trans_df = pd.read_csv("KKBL.csv", sep=",", low_memory=False, error_bad_lines=False)
    Trans_df = Trans_df.fillna(-1)
####################################################################################################
    grouped_df = Trans_df.groupby([Trans_df['settle_dt']], group_keys=True)

    group_keys = []
    for name, group in grouped_df:
        group_keys.append(name)
##############################################groupby 之后agg################################################
    agg_dict = {}
    agg_dict["settle_trans_at"] = ['count','sum']

    agg_stat_df = grouped_df.agg(agg_dict)
    agg_stat_df.columns = agg_stat_df.columns.map('{0[0]}-{0[1]}'.format)
    agg_stat_df.reset_index(level=0, inplace=True)

###############################################
    agg_stat_df.to_csv("KKBL_agg.csv",index=False)



