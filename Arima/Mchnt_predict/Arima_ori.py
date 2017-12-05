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
    data = pd.read_csv('JLF_agg.csv', parse_dates=True, index_col='settle_dt',date_parser=dateparse)

    ts = data['settle_trans_at-count'].astype(np.float)


    #简单点一般最大值取2就够了
    pmax=5
    qmax=5

    #aic矩阵
    aic_matrix = []
    for p in range(pmax+1):
      tmp = []
      for q in range(qmax+1):
        try:  #存在部分报错，所以用try来跳过报错。
          tmp.append(ARIMA(ts, (p,1,q)).fit().aic)
        except:
          tmp.append(None)
      aic_matrix.append(tmp)

    #从中可以找出最小值
    aic_matrix = pd.DataFrame(aic_matrix)
    print  aic_matrix
    #先用stack展平，然后用idxmin找出最小值位置。
    p,q = aic_matrix.stack().idxmin()
    print(u'\n\n AIC最小的p值和q值为：%s、%s' %(p,q))


    model = ARIMA(ts, order=(p, 1, q))
    results_ARIMA = model.fit(disp=False)

    #forecast_dta = results_ARIMA.forecast(10)[0]   #forecast  的值是ts  不是差分
    # dates = pd.date_range('20171201', '20171210')
    # #dates = pd.date_range('20171201', periods=10, freq='D')
    #
    # new_ts = Series(forecast_dta, index=dates)
    # print new_ts
    # plt.plot(ts)
    # plt.plot(new_ts, color='red')
    # plt.show()

    forecast_dta = results_ARIMA.predict('20171201', '20171210', dynamic=True)
    print forecast_dta
    plt.plot(ts)
    plt.plot(forecast_dta, color='red')
    plt.show()

