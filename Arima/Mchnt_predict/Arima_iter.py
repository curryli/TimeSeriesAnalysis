# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas import Series

if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    data = pd.read_csv('KKBL_agg.csv', parse_dates=True, index_col='settle_dt',date_parser=dateparse)

    ts = data['settle_trans_at-count'].astype(np.float)

    new_start = 20171201
    for i in range(20):

        #简单点一般最大值取2就够了
        pmax=2
        qmax=2

        #bic矩阵
        bic_matrix = []
        for p in range(pmax+1):
          tmp = []
          for q in range(qmax+1):
            try:  #存在部分报错，所以用try来跳过报错。
              tmp.append(ARIMA(ts, (p,1,q)).fit().bic)
            except:
              tmp.append(None)
          bic_matrix.append(tmp)

        #从中可以找出最小值
        bic_matrix = pd.DataFrame(bic_matrix)
        p,q = bic_matrix.stack().idxmin()
        print(u'\n\n BIC最小的p值和q值为：%s、%s' %(p,q))
        print('\n\n下面对比ts_diff与arima模型拟合出来的结果')
        model = ARIMA(ts, order=(p, 1, q))
        results_ARIMA = model.fit(disp=-1)

        forecast_dta = results_ARIMA.forecast(1)[0]   #forecast  的值是ts  不是差分
        dates = pd.date_range(str(new_start), periods=1, freq='D')
        new_ts = Series(forecast_dta, index=dates)
        ts = pd.concat([ts, new_ts], axis=0)

        new_start = new_start +1



    print ts
    plt.plot(ts)
    # plt.plot(new_ts, color='red')
    plt.show()

