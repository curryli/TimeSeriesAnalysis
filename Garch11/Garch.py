# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 13:59:50 2015
Last modified on Mon Jan 05 10:51:01 2014
@author: congxiu
"""
import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

"""
#Example uses 1 year's data to fit the model, and then make prediction on the
#next 3 months

daily_return = pd.read_csv('./IF_daily_return.csv', parse_dates = 'dt',
                           index_col = 'dt')['daily_return']
variance = pd.rolling_var(daily_return, 60).dropna()
garch = Garch11(daily_return[-300 : -60], 0.000006, 0.05, 0.9)
garch.fit()
print "parameters alpha0(omega), alpha1, beta1, mu, sigma are", garch.params()

sq_resid_re = (daily_return[-61:-1] - garch.params()[3]) ** 2
data = pd.DataFrame(data = [variance[-61:-1], sq_resid_re]).transpose()
data.columns = ['sq_resid_re', 'daily_return']
pred_var = data.apply(lambda x:garch.predict(x[1], x[0]), axis = 1)
hist_var = variance[-60:]
pred_var.index = hist_var.index

pred_var.plot()
hist_var.plot()
plt.legend(['predicted', 'historical'])
"""



class Garch11(object):
    """
    An implementation of GARCH(1, 1) model
    """
        
    def __init__(self, series, alpha0 = 0.0001, alpha1 = 0.2, beta1 = 0.3):
        try:
            if alpha0 < 0 or alpha1 < 0 or beta1 < 0 or \
            alpha1 + beta1 >= 1:
                raise ValueError("Irregular coefficients")
            else:
                self.alpha0 = alpha0
                self.alpha1 = alpha1
                self.beta1 = beta1
        except ValueError:
            print "Initial condition leads to unstationary series or negative variance."
            print "Resetting everything to default"
            self.alpha0 = 0.1
            self.alpha1 = 0.2
            self.beta1 = 0.3
        finally:
            self.mu = series.mean()
            self.sigma = series.std()
            self.size = len(series)
            self.data = series
    
    def params(self):
        """
        Returns a list of parameters
        """
        return [self.alpha0, self.alpha1, self.beta1, self.mu, self.sigma]
        
    @staticmethod
    def likelihood(param, data):
        """
        Returns negative log likelihood.
        """
        size = len(data)
        sq_resi_return = (data - param[3]) ** 2
        variance = pd.Series(param[4] ** 2, index = [sq_resi_return.index.values[0]])
        variance = variance.append(param[0] + param[1] * sq_resi_return.shift(1).dropna())
        
        for idx in range(1, size):
            variance.iloc[idx] = variance.iloc[idx] + param[2] *\
            variance.iloc[idx - 1] 
        
        likelihood = -(size * np.log(2 * math.pi) + np.log(variance).sum()
        + (sq_resi_return / variance).sum()) / 2
        
        return -likelihood
        
    def fit(self):  
        """
        Fit the model
        """
        cons = ({'type' : 'ineq',
                 'fun' : lambda x : np.array([x[0] - 0.000001])},
                {'type' : 'ineq',
                 'fun' : lambda x : np.array([x[1] - 0.000001])},
                {'type' : 'ineq',
                 'fun' : lambda x : np.array([x[2] - 0.000001])},
                {'type' : 'ineq',
                 'fun' : lambda x : np.array([- x[1] - x[2] + 0.999999])},
                {'type' : 'ineq',
                 'fun' : lambda x : np.array([x[4] - 0.000001])}
                 )
                 
        res = sp.optimize.minimize(self.likelihood, np.array(self.params()),
                             args = (self.data,), constraints = cons, 
                             method = 'COBYLA',
                             options = {'disp' : True})
        
        #res = sp.optimize.minimize(self.likelihood, np.array(self.params()),
        #                           args = (self.data,))
        
        self.alpha0 = res.x[0]
        self.alpha1 = res.x[1]
        self.beta1 = res.x[2]
        self.mu = res.x[3]
        self.sigma = res.x[4]
        
    def predict(self, curr_var, sq_resid_return):
        """
        Make prediction based on fitted model
        
        Returns the next period's unconditional variance
        """
        new_var = self.alpha0 + self.alpha1 * sq_resid_return + self.beta1 * curr_var
        
        return new_var