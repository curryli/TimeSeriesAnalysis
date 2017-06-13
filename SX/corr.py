#-*- coding: utf-8 -*-
#餐饮销量数据相关性分析
import pandas as pd
 
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('data.csv', parse_dates=True, index_col='date',date_parser=dateparse)
                     
d1=data.corr() #相关系数矩阵，即给出了任意两款菜式之间的相关系数  df.corr('spearman')  默认pearson
print d1

d3=data[u'SW_large'].corr(data[u'Z']) #计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数
print d3