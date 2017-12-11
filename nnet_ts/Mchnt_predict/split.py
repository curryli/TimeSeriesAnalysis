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




if __name__ == '__main__':
    Trans_df = pd.read_csv("Oridata.csv", sep=",", low_memory=False, error_bad_lines=False)
####################################################################################################
    grouped_df = Trans_df.groupby([Trans_df['mchnt_brand']], group_keys=True)

    group_keys = []
    i = 0
    for name, group in grouped_df:
        i = i+1
        savename = "splitted/" + name + ".csv"
        group.to_csv(savename, index=False)
        print i
