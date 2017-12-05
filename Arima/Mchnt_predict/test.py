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
# from statsmodels.tsa.arima_model import ARIMAResults
from pandas import Series

if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    data = pd.read_csv('JLJR_agg.csv', parse_dates=True, index_col='settle_dt', date_parser=dateparse)

    ts = data['settle_trans_at-count'].astype(np.float)

    model = ARIMA(ts, order=(3, 1, 8))
    results_ARIMA = model.fit(disp=-1,  method='css')

    forecast_dta = results_ARIMA.forecast(10)[0]   #forecast  的值是ts  不是差分
    dates = pd.date_range('20171201', '20171210')
    #dates = pd.date_range('20171201', periods=10, freq='D')

    new_ts = Series(forecast_dta, index=dates)
    print new_ts
    plt.plot(ts)
    plt.plot(new_ts, color='red')
    plt.show()

    # forecast_dta = results_ARIMA.predict('20171201', '20171210', dynamic=True)
    # print forecast_dta
    # plt.plot(ts)
    # plt.plot(forecast_dta, color='red')
    # plt.show()
