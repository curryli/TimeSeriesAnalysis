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

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('data.csv', parse_dates=True, index_col='date',date_parser=dateparse)


ts = data[u'Z']
#ts = np.log(data[u'SW_large'])
#ts = data[u'SW_large']


ts_diff = ts.diff()#.diff().diff()
ts_diff.dropna(inplace=True)


plt.title('original ts') 
plt.plot(ts)
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
model = ARIMA(ts, order=(p, 1, q))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_diff)**2))
plt.show()

 
predata = results_ARIMA.predict('2011-01-01', '2017-09-01', dynamic=False)
print "predata is :"
print predata 
plt.plot(ts)
plt.plot(predata,"red")
plt.show()
 
forecast_dta = results_ARIMA.forecast(30)[0]   #forecast  的值是ts  不是差分
print forecast_dta
 
 
plt.plot(forecast_dta)
plt.show()