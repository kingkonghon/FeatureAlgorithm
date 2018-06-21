# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 09:49:16 2018

@author: liuxin
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
import os
import sys
from sqlalchemy import create_engine
from multiprocessing import Pool

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

sourceTableQuote = 'STOCK_FORWARD_ADJ_QUOTE'

Days=[5,10,20,30,60,90,120]

targetTableName = 'STOCK_VALUE'
weightField = 'amount'

##计算value
def calPV(dataO, weight, day, am_median_name, ret_name):
    # data['weight'] = data[weightField] / data[am_median_name]
    cum_weight = weight.rolling(window=day).apply(lambda x: np.prod(x))  #cumulative product of weights
    pv = dataO[ret_name] * cum_weight  # return * cum weight

    return pv


def calMarketInsight(dataO, weight, am_median_name):
    # data['weight'] = data[weightField]/data[am_median_name]
    insight = dataO['ret_1D'] / weight * 100
    return insight


# def calFeatures(dataO):
    # stocks = dataO['code'].unique().tolist()
    #
    # dataresult = pd.DataFrame()
    # for stock in stocks:
    #     subset = dataO[dataO.code == stock].sort_values('date')
    #     subset['ret_1D'] = subset['close'] / subset['close'].shift(1) - 1
    #     tmp_result = pd.DataFrame(subset[['date', 'code']])
    #     for day in days:
    #         am_median_name = 'am_median_%dD' % day
    #         ret_name = 'ret_%dD' % day
    #         pv_name = 'PV_%dD' % day
    #         mrk_insight_name = 'Market_Insight_%dD' % day
    #         subset.loc[:, am_median_name] = subset[weightField].rolling(window=day+1).median() # shift(n)  --> rolling(n+1)
    #         subset.loc[:, ret_name] = subset['close'] / subset['close'].shift(day) - 1
    #         tmp_result.loc[:, pv_name] = calPV(subset, day, am_median_name, ret_name)
    #         tmp_result.loc[:, mrk_insight_name] = calMarketInsight(subset, am_median_name)
    #     dataresult = dataresult.append(tmp_result)
    # return dataresult

def calFeatures(subset):
    subset['ret_1D'] = subset['close'] / subset['close'].shift(1) - 1
    tmp_result = pd.DataFrame(subset[['date', 'code']])
    for day in Days:
        am_median_name = 'am_median_%dD' % day
        ret_name = 'ret_%dD' % day
        pv_name = 'PV_%dD' % day
        mrk_insight_name = 'Market_Insight_%dD' % day
        subset.loc[:, am_median_name] = subset[weightField].rolling(window=day+1).median() # shift(n)  --> rolling(n+1)
        tmp_weight = subset[weightField] / subset[am_median_name]
        subset.loc[:, ret_name] = subset['close'] / subset['close'].shift(day) - 1
        tmp_result.loc[:, pv_name] = calPV(subset, tmp_weight, day, am_median_name, ret_name)
        tmp_result.loc[:, mrk_insight_name] = calMarketInsight(subset, tmp_weight, am_median_name)

    return tmp_result

def calIncrmFeatures(subset, start_date):
    subset.loc[:, 'ret_1D'] = subset['close'] / subset['close'].shift(1) - 1

    tmp_new_data = subset.loc[subset['date'] > start_date]
    tmp_old_data = subset.loc[subset['date'] <= start_date]
    new_data_num = tmp_new_data.shape[0]

    if new_data_num > 0:
        tmp_result = tmp_new_data[['date', 'code']]
        for day in Days:
            tmp_lag_data = tmp_old_data.iloc[-(2*day + 1):] # include data for both median and cum product
            tmp_full_data = tmp_lag_data.append(tmp_new_data)

            am_median_name = 'am_median_%dD' % day
            ret_name = 'ret_%dD' % day
            pv_name = 'PV_%dD' % day
            mrk_insight_name = 'Market_Insight_%dD' % day
            tmp_full_data.loc[:, am_median_name] = tmp_full_data[weightField].rolling(window=day+1).median() # shift(n)  --> rolling(n+1)
            tmp_weight = tmp_full_data[weightField] / tmp_full_data[am_median_name]
            tmp_full_data.loc[:, ret_name] = tmp_full_data['close'] / tmp_full_data['close'].shift(day) - 1

            # PV
            pv = calPV(tmp_full_data, tmp_weight, day, am_median_name, ret_name)
            tmp_result.loc[:, pv_name] = pv.iloc[-new_data_num:]

            # market insight
            mrk_ins = calMarketInsight(subset, tmp_weight, am_median_name)
            tmp_result.loc[:, mrk_insight_name] = mrk_ins.iloc[-new_data_num:]
    else:
        tmp_cols = ['date', 'code']
        for day in Days:
            tmp_cols.append('PV_%dD' % day)
            tmp_cols.append('Market_Insight_%dD' % day)
        tmp_result = pd.DataFrame([], columns=tmp_cols)

    return tmp_result


def calFull(chunk_size=50000):
    # connect db
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    sql_statement = "select `code`,date, close, amount from %s" % sourceTableQuote
    dataO = pd.read_sql(sql_statement, quant_engine)

    stocks = dataO['code'].unique().tolist()

    pool = Pool(processes=3)

    multi_pro_results = []
    for (i, stock) in enumerate(stocks):
        subset = dataO[dataO.code == stock].sort_values('date')

        # tmp_result = calFeatures(subset)
        tmp_result = pool.apply_async(calFeatures, (subset,))
        multi_pro_results.append(tmp_result)

    dataresult = pd.DataFrame()
    for i in range(len(multi_pro_results)):
        tmp_result = multi_pro_results[i].get()
        dataresult = dataresult.append(tmp_result)
    # dataresult = dataresult.append(tmp_result)

    # reconnect db
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    write_method = 'replace'
    for i in range(int(dataresult.shape[0] / chunk_size) + 1):
        chunk_data = dataresult.iloc[i * chunk_size: (i + 1) * chunk_size]
        chunk_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
        write_method = 'append'

def calIncrm():
    # connect db
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # find target table date and date to fetch data
    sql_statement = "select max(date) from %s" % targetTableName
    target_max_date = pd.read_sql(sql_statement, quant_engine)
    if not target_max_date.empty:
        target_max_date = target_max_date.iloc[0,0]

    target_fetch_date = datetime.strptime(target_max_date, '%Y-%m-%d') - timedelta(days=365) # lags
    target_fetch_date = datetime.strftime(target_fetch_date, '%Y-%m-%d')
    # fetch data
    sql_statement = "select `code`,date, close, amount from %s where date >= '%s'" % (sourceTableQuote, target_fetch_date)
    dataO = pd.read_sql(sql_statement, quant_engine)

    # calculate feature stock by stock
    stocks = dataO['code'].unique().tolist()

    # multiprocess
    pool = Pool(processes=4)

    dataresult = pd.DataFrame()
    all_processes = []
    for stock in stocks:
        subset = dataO[dataO.code == stock].sort_values('date')

        tmp_process = pool.apply_async(calIncrmFeatures, (subset, target_max_date))
        all_processes.append(tmp_process)
        # tmp_result = calIncrmFeatures(subset, target_max_date)
        # dataresult = dataresult.append(tmp_result)

    for tmp_process in all_processes:
        tmp_result = tmp_process.get()
        dataresult = dataresult.append(tmp_result)

    # dataresult = calFeatures(dataO)
    if not dataresult.empty:
        dataresult = dataresult.sort_values('date')
        # reconnect to db
        quant_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
        dataresult.to_sql(targetTableName, quant_engine, index=False, if_exists='append')


def airflowCallable():
    calIncrm()

if __name__ == '__main__':
    # calFull()
    t_start = time.clock()
    # calIncrm()
    airflowCallable()
    print(time.clock() - t_start)