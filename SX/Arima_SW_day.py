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
#from statsmodels.tsa.arima_model import ARIMAResults

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
    


rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('SW_all.csv', parse_dates=True, index_col='date',date_parser=dateparse)


#ts = data[u'Z']
#ts = np.log(data[u'SW_large'])
ts = data[u'SW_large']

moving_avg = pd.rolling_mean(ts,12)
#plt.plot(ts)
#plt.plot(moving_avg, color='red')
#plt.show()
 
expwighted_avg = pd.ewma(ts, halflife=12)
#plt.plot(ts)
#plt.plot(expwighted_avg, color='red')
#plt.show()

#ts_diff = ts - expwighted_avg
#ts_diff = ts - moving_avg
ts_diff = ts.diff()#.diff()#.diff()

ts_diff.dropna(inplace=True)


plt.title('ts_diff') 
#plt.plot(ts)
plt.plot(ts_diff, color='red')
plt.show()


lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')


print('\n\n下面对ts_diff序列进行ADF检验') 
ts_diff.dropna(inplace=True)
test_stationarity(ts_diff)
 
print('\n\n下面对ts_diff序列白噪声检验')
P_result = acorr_ljungbox(ts_diff, lags=1)[1][0]
print(P_result)
if(P_result<0.05):
    print('\n\nts_diff为平稳非白噪声序列,检验通过')
else:
    print('\n\nts_diff为白噪声序列,检验失败')
    sys.exit(0)
    

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
print  bic_matrix
#先用stack展平，然后用idxmin找出最小值位置。
p,q = bic_matrix.stack().idxmin() 
print(u'\n\n BIC最小的p值和q值为：%s、%s' %(p,q))


print('\n\n下面对比ts_diff与arima模型拟合出来的结果') 
#model = ARIMA(ts, order=(p, 1, q))  


model = ARIMA(ts, order=(7, 1, 7))  

results_ARIMA = model.fit(disp=-1)    #fit  返回ARIMAResults 类型。 ARIMAResults类型不直接面对用户 
plt.plot(ts_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_diff)**2))
plt.show()





predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA = pd.Series(ts.ix[0], index=ts.index)
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print predictions_ARIMA_log.head()
 
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()


#尝试使用pyflux
 
##下面这个时间必须要在index_col='date' 里面
#predict_dta = results_ARIMA.predict('2016-02-01','2016-02-09', dynamic=True)  #predict只能预测已有的日期的数据
#predict_dta = results_ARIMA.predict()
predict_dta = results_ARIMA.forecast(5)  #forecast可以预测end之后后面5天的数据

 
print(predict_dta)