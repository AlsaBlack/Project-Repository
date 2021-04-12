# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:22:25 2020

@author: h
"""


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels as sm
import math 
from scipy import stats

path=r"C:\Users\h\Documents\学习\数量金融自学读物\300ETF.csv"
data=pd.read_csv(path)
data.index=data['date']
data=pd.Series(data['close'])
#plt.plot(data)

data = data.map(lambda x:math.log(x))
data = data.diff(1)[1:]
# plt.plot(data)
# plt.show()

from datetime import datetime
start=data.index[1]
end=data.index[-1]
breakpoint=str(pd.Timestamp("2018-12-31"))
in_sample=data[start:breakpoint]
out_sample=data[breakpoint:end]

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
acf=acf(data)
pacf=pacf(data)
acf_fig=plt.plot(range(10),acf[:10])
pacf_fig=plt.plot(pacf[:10])

from statsmodels.tsa import stattools
res=sm.tsa.stattools.arma_order_select_ic(
    data,
    ic=['aic','bic'],
    trend='nc',
    max_ar=4,
    max_ma=4
    )
print(res.aic_min_order)
print(res.bic_min_order)

mod_aic=sm.tsa.arima_model.ARMA(in_sample,order=res.aic_min_order)
mod_bic=sm.tsa.arima_model.ARMA(in_sample,order=res.bic_min_order)

ARMAmodel = []
for i in range(1,5):
  for j in range(0,3):
    ARMAmodel.append(sm.tsa.arima_model.ARMA(
        in_sample,
        order=(i,j)
        ))

mod_res1 = mod_aic.fit(trend="nc",disp=-1)
mod_res1.summary2()

import matplotlib.pyplot as plt
resid1=mod_res1.resid
plt.plot(resid1)

resid1_acf=sm.tsa.stattools.acf(resid1)
plt.plot(resid1_acf)

stats.probplot(resid1, dist="norm", plot=plt)
plt.show()

mod_res2 = mod_aic.fit(trend="nc",disp=-1)
mod_res2.summary2()
resid2=mod_res2.resid
plt.plot(resid2)
resid2_acf=sm.tsa.stattools.acf(resid2)
plt.plot(resid2_acf)
stats.probplot(resid2, dist="norm", plot=plt)
plt.show()

pre1=mod_res1.predict(end=len(out_sample)-1)#为什么要-1
pre1.index=out_sample.index
pre2=mod_res1.predict(end=len(out_sample)-1)#为什么要-1
pre2.index=out_sample.index

plt.plot(pre1[:100])
plt.plot(pre2)
plt.plot(out_sample)
plt.show()



