# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 09:49:16 2018

@author: liuxin
"""

import pandas as pd
import pandas.io.sql as sql
import numpy as np
import pymysql
from sqlalchemy import create_engine

con=pymysql.connect('10.46.228.175','root','xunzhaoshengbei','quant',charset = 'utf8')
data0=sql.read_sql("select code,date,pre_close,volume,return_1 from yucezhe_qfq_finaldata1 limit 6000",con)

days=[5,10,20,30,60,90,120]

stocks = data0['code'].unique().tolist()

##计算value 
'''
1. calPV * calValue
股票的涨跌幅与交易量并不成正比。
以下跌为例，有时候，股票下跌巨大，但是交易量很小，这时候实际上受损失的人很少，数额也较少，对于这种情况，我们想缩小下跌的效应
同样的，有时候，股票下跌幅度不大，但是交易量很大，这时候我们想放大下跌的效应
首先选取不同的时间窗口，以10天为例,此时day=10,假设选取3.1-3.10（日期）
为了计算相对值，我们首先对交易量volume进行处理。先选取10日内volume的中位数，然后每个volume除以中位数，作为权重,记为v_per
令第一日的return为0，value0 = v_per*(1+return0)
此后，value_t = value_t-1 *(1+return_t)*v_per
则，对于3.10来说，value_10就是3.1-3.10这个窗口的最后一个value_t
同样，对于3.11来说，value_10就是3.2-3.11的最后一个value_t
对于每一个窗口，我们只要最后一个value_t,赋值给该窗口的最后一个日期
这个value_t的经济意义，我们可以认为是从3.1开始，这支股票的理论价值涨幅（可以跟经济学中的价值相联系）+1 （注意是加了1的）
我们用real_return计算该窗口真实涨幅
用p-v_day 表示在窗口为day时，(real_return+1 )-value_t(这里的t就是day)


2. calMarketInsight
如上所述，股票的涨幅与交易量并不成正比。
有时候下跌幅度巨大，但是交易量很少，说明市场对该股票看法一致。短时间内，大家都认为这会跌，可能在很少的交易量时，价格就跌的很厉害
有时候下跌幅度很小，但是交易量很大，说明市场对该股票看法不一致，股票下跌是多方博弈的结果。
那么，我们对交易量同样做类似如上处理（volume/volume在窗口内的中位数）算出相对交易量，但是略有不同。好难说，不详细说了，可以看代码
接下来计算涨幅与相对交易量的比值*100(*100不是百分比，只是为了放大数值，否则很小)
第一种情况，比值很大，看法一致。类似，第二种情况，比值很小，看法不一致。
总结，比值越大，看法越一致。
缺陷：当return为0时，比值为0，其实此时交易量是不同的，但是与实际偏离不大，因为很可能价格不变是多方博弈的结果，而不是市场一致看涨或看跌
'''

def calValue(x,data,day):
    x = data[data.index.isin(x)]
    day=len(x)
    midname = 'mid_%s'%day
    v_per = x['volume']/x[midname].values[-1]
    values=[]
    values.append(v_per.values[0])
    for i in range(1,day):
        values.append(values[-1]*(1+x['return_1'].values[i])*v_per.values[i])
    return values[-1]
    
def calPV(data,day):

    a=pd.rolling_apply(pd.DataFrame(data.index),day,lambda x: calValue(x,data,day))
    data[valuename]=a.values
    real_rtn = 'real_rtn_%s'%day
    data[real_rtn] = (data['pre_close']-data['pre_close'].shift(day-1))/data['pre_close'].shift(day-1)
    pv = 'P-V_%s'%day
    data[pv]=data[real_rtn]-data[valuename]+1
    return data
 
def calMarketInsight(data,day):
    midname = 'mid_%s'%day
    insightname = 'MktInsight_%s'%day
    v_per = data['volume']/data[midname]
    data[insightname] = data['return_1']/v_per*100
    return data
    
    
    
    
dataresult=pd.DataFrame()   
for stock in stocks:
    data = data0[data0['code']==stock].sort_values('date')
    for day in days:
        midname = 'mid_%s'%day
        valuename = 'value_%s'%day
        data[midname]=pd.rolling_median(data['volume'],day)
        data = calPV(data,day)
        data = calMarketInsight(data,day)

    dataresult = dataresult.append(data)





