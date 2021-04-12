# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:43:16 2020

@author: h

"""
#导入包
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels as sm
import math 
from scipy import stats

#导入数据
path=r"C:\Users\h\Documents\学习\数量金融自学读物\300ETF.csv"
data=pd.read_csv(path)
data.index=data['date']
data=pd.Series(data['close'])
#plt.plot(data)

#数据平稳性检验
from statsmodels.tsa import stattools
test=sm.tsa.stattools.adfuller(data) 
print(test)

#数据差分
data = data.map(lambda x:math.log(x))
data = data.diff(1)[1:]
# plt.plot(data)
# plt.show()

#数据平稳性检验
test=sm.tsa.stattools.adfuller(data) 
print(test)

#分割数据
from datetime import datetime
start=data.index[1]
end=data.index[-1]
breakpoint=str(pd.Timestamp("2018-12-31"))
in_sample=data[start:breakpoint]
out_sample=data[breakpoint:end]

#判断截尾性质,得出滞后项
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
acf=acf(in_sample)
pacf=pacf(in_sample)
fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
acf_fig = plot_acf(in_sample,lags = 20,ax=ax1)
pacf_fig=plot_pacf(in_sample,lags = 20,ax=ax1)
# acf_fig=plt.plot(range(10),acf[:10])
# pacf_fig=plt.plot(pacf[:10])

#建立模型
order=(4,0)
model =sm.tsa.arima_model.ARMA(in_sample,order).fit()
model.summary()#ar模型

#是否具有arch效应
at =model.resid
at2 = np.square(at)
plt.figure(figsize=(10,6))
plt.subplot(211)
plt.plot(at,label = 'at')
plt.legend()
plt.subplot(212)
plt.plot(at2,label='at^2')
plt.legend(loc=0)

#对at2序列进行混成检验： 原假设H0:序列没有相关性，备择假设H1:序列具有相关性
m = 25 # 我们检验25个自相关系数
acf,q,p = sm.tsa.stattools.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
out = np.c_[range(1,26), acf[1:], q, p]
output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
output = output.set_index('lag')
output

#确定arch模型滞后阶数
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
acf=acf(at2)
pacf=pacf(at2)
fig = plt.figure(figsize=(20,5))
ax1=fig.add_subplot(111)
pacf_fig=plot_pacf(at2,lags = 20,ax=ax1)

#建立arch模型
from arch import arch_model
am = arch_model(in_sample,mean='AR',lags=8,vol='ARCH',p=5) 
res = am.fit()
#查看参数
res.summary()
res.params
#预测1
res.hedgehog_plot()
# # #预测2forecast可真难用啊
# pre = res.forecast(horizon=len(out_sample),start=1)
# plt.figure(figsize=(10,4))
# plt.plot(out_sample,label='realValue')
# pre.variance[breakpoint:].plot()
# plt.plot(np.zeros(10),label='zero')
# plt.legend(loc=0)

#不用predict了
# pre1=mod_res1.predict(end=len(out_sample)-1)#为什么要-1
# pre1.index=out_sample.index
# pre2=mod_res1.predict(end=len(out_sample)-1)#为什么要-1
# pre2.index=out_sample.index

# plt.plot(pre1[:100])
# plt.plot(pre2)
# plt.plot(out_sample)
# plt.show()



#seaborn也不用了
# import seaborn
# seaborn.set_style('darkgrid')
# seaborn.mpl.rcParams['figure.figsize'] = (10.0, 6.0)
# seaborn.mpl.rcParams['savefig.dpi'] = 90
# seaborn.mpl.rcParams['font.family'] = 'sans-serif'
# seaborn.mpl.rcParams['font.size'] = 14

# import datetime as dt
# st = dt.datetime(2014, 7, 23)
# en = dt.datetime(2017, 12,31)

# from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
# lb_test(data,lags=None,boxpierce=False)
# plt.plot(lb_test[0])
# plt.plot(lb_test[1])
# plt.show()

#ARCH Model
from arch import arch_model
mod_ARCH = arch_model(in_sample)
res_ARCH = mod_ARCH.fit(update_freq=1)
print(res_ARCH.summary())
fig = res_ARCH.plot(annualize='D') 

#GJR Model 
mod_GJR = arch_model(in_sample, p=1, o=1, q=1)
res_GJR = mod_GJR.fit(update_freq=1, disp='final')
print(res_GJR.summary())
fig = res_GJR.plot(annualize='D')
 
#TARCH/ZARCH Model
mod_TARCH = arch_model(insample, p=1, o=1, q=1, power=1.0)
res_TARCH = mod_TARCH.fit(update_freq=1)
print(res_TARCH.summary())
fig = res_TARCH.plot(annualize='D')


#ARCH with Student's T Errors
am_a = arch_model(insample, dist='StudentsT')
res_a = am_a.fit(update_freq=1)
print('ARCH with StudentT\'s loglikelihood:'+str(res_a.loglikelihood))

#GJR ARCH with Student's T Errors
am_g = arch_model(insample, p=1, o=1, q=1, dist='StudentsT')
res_g = am_g.fit(update_freq=1)
print('TARCH with StudentT\'s loglikelihood:'+str(res_g.loglikelihood))
#print(res.summary())

#TARCH with Student's T Errors
am_t = arch_model(data, p=1, o=1, q=1, power=1.0, dist='StudentsT')
res_t = am_t.fit(update_freq=1)
print('TARCH with StudentT\'s loglikelihood:'+str(res_t.loglikelihood))
#print(res.summary())

import collections
lls = pd.Series(
    collections.OrderedDict((('TARCH', res_TARCH.loglikelihood),
                 ('ARCH', res_ARCH.loglikelihood), 
                 ('GJR',res_GJR.loglikelihood)
                 )))
print(lls)
lls1 = pd.Series(
    collections.OrderedDict((('TARCH with StudentT\'s', res_t.loglikelihood),
                 ('ARCH with StudentT\'s', res_a.loglikelihood), 
                 ('GJR with StudentT\'s',res_g.loglikelihood)
                 )))
print(lls1)
#选择ARCH
res_ARCH.params
#比较
# res_ARCH.plot()
# plt.plot(data)
#预测
# res_ARCH.hedgehog_plot()

sim_mod = arch_model(in_sample)
sim_data = sim_mod.simulate(res.params, 403)
sim_data.index=out_sample.index

plt.plot(sim_data.data+sim_data.volatility,label='r')
plt.plot(out_sample,label='b')

# pre1=mod_ARCH.predict(end=len(out_sample)-1)#为什么要-1
# pre1.index=out_sample.index

# ini = res_ARCH.resid[-10:]
# a = np.array(res_ARCH.params[1:9])
# w = a[::-1] # 系数
# for i in range(10):
#     new = test[i] - (res_ARCH.params[0] + w.dot(ini[-8:]))
#     ini = np.append(ini,new)
# print( len(ini))
# at_pre = ini[-10:]
# at_pre2 = at_pre**2
# at_pre2

# #评价
# def MSE(ARMAmodel):
#   mod_res = ARMAmodel.fit(trend="nc",disp=-1)
#   pre = mod_res.predict(end=len(out_sample)-1)
#   pre.index = out_sample.index
#   print(sum((pre - out_sample)**2)/ len(pre))
# print(MSE(mod_aic)) 
# print(MSE(mod_bic))