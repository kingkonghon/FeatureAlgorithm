#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:12:48 2018

@author: lipchiz
"""
import os
import sys
import pandas as pd
import pandas.io.sql as sql
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from multiprocessing import Pool
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# parameters
days = [5, 10, 20, 60, 90, 120, 252]
halfLife = 63

# con = pymysql.connect('10.46.228.175', 'alg', 'Alg#824', 'quant', charset='utf8')

sourceStockQuoteTableName = 'STOCK_FORWARD_ADJ_QUOTE'
sourceIndexQuoteTableName = 'HS300_QUOTE'

targetTableName = 'STOCK_BETA_NEW'

timestampField = 'time_stamp'

def BETA(data, day, halflife, weight):
    # calculate asset return & hs300 return
    # asset_return = (np.asarray(data.close_asset[1:]) - np.asarray(data.close_asset[:-1])) / np.asarray(
    #     data.close_asset[:-1])   #**********************************
    asset_return = (data.close_asset / data.close_asset.shift(1) - 1).values
    # asset_return = asset_return.tolist()
    # asset_return.insert(0, np.nan)

    # hs300_return = (np.asarray(data.close_hs300[1:]) - np.asarray(data.close_hs300[:-1])) / np.asarray(
    #     data.close_hs300[:-1])  #**********************************
    hs300_return = (data.close_hs300 / data.close_hs300.shift(1) - 1).values
    # hs300_return = hs300_return.tolist()
    # hs300_return.insert(0, np.nan)

    # data['asset_return'], data['hs300_return'] = asset_return, hs300_return
    # data.reset_index(drop=True, inplace=True)

    # beta = np.repeat(np.nan, day).tolist()
    beta = np.full(data.shape[0], np.nan) #*****************************
    for i in np.arange(day, len(data)):
        # tmp_subset = []
        # tmp_subset = data.copy().iloc[i - day + 1:i + 1, :]
        subset_asset_ret = asset_return[i - day + 1 : i + 1]
        subset_hs300_ret = hs300_return[i - day + 1: i + 1]
        # asset_return = np.asarray(tmp_subset['asset_return'])
        # hs300_return = np.asarray(tmp_subset['hs300_return'])
        # cov = np.sum((asset_return - np.mean(asset_return)) * (hs300_return - np.mean(hs300_return)) * weight)
        # var = np.sum((hs300_return - np.mean(hs300_return)) ** 2 * weight)
        cov = np.sum((subset_asset_ret - np.mean(subset_asset_ret)) * (subset_hs300_ret - np.mean(subset_hs300_ret)) * weight)
        var = np.sum((subset_hs300_ret - np.mean(subset_hs300_ret)) ** 2 * weight)
        tmp_beta = cov / var
        # beta.append(tmp_beta)
        beta[i] = tmp_beta #*****************************
    return beta


def calFull():
    # read data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    data = sql.read_sql("select `code`,date,close from %s " % sourceStockQuoteTableName, con_quant)

    # calculation
    codes = data1['code'].unique().tolist()
    dataresult = pd.DataFrame()

    # multiprocess pool
    pool = Pool(processes=20)

    for code in codes:
        tmp_data = data1[data1['code'] == code]
        tmp_data = tmp_data.sort_values('date', ascending=True)

        # buffer to store multiprocess result
        multi_process_result = []
        is_skip = np.full(len(days), False)

        # run multi-process
        for i, day in enumerate(days):
            # xnam = 'beta_{n}'.format(n=day)
            if len(tmp_data) < day:
                # tmp_data[xnam] = np.nan
                is_skip[i] = True
            else:
                # tmp_data[xnam] = BETA(data=tmp_data, day=day, halflife=63, weight=all_weights[day])
                tmp_result = pool.apply_async(BETA, (tmp_data, day, halfLife, all_weights[day]))
                multi_process_result.append(tmp_result)

        # get result from multi process
        for i, day in enumerate(days):
            xnam = 'beta_{n}'.format(n=day)
            if is_skip[i]:
                tmp_data[xnam] = np.nan
            else:
                tmp_data[xnam] = multi_process_result[i].get()

        # combine result from different code
        dataresult = dataresult.append(tmp_data)

    # drop columns
    dataresult = dataresult[final_columns]
    chunk_size = 500000

    # add time stamp
    dataresult.loc[:, timestampField] = datetime.now()

    # dump data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    write_method = 'replace'
    for i in range(int(dataresult.shape[0] / chunk_size) + 1):
        chunk_data = dataresult.iloc[i*chunk_size : (i+1)*chunk_size]
        chunk_data.to_sql(targetTableName, con_quant, index=False, if_exists=write_method)
        write_method = 'append'

# interface for multiprocess
def calMultiBetas(tmp_data, target_max_date, all_weights):
    tmp_data = tmp_data.sort_values('date', ascending=True)
    tmp_data.reset_index(drop=True, inplace=True)
    tmp_result = tmp_data.loc[tmp_data['date'] > target_max_date, ['date', 'code']]
    if not tmp_result.empty:
        tmp_new_data = tmp_data.loc[tmp_data['date'] > target_max_date]
        tmp_data_num = tmp_new_data.shape[0]
        for day in days:
            xnam = 'beta_{n}'.format(n=day)

            tmp_lag_data = tmp_data.loc[tmp_data['date'] <= target_max_date].iloc[-day:]
            tmp_tot_data = tmp_lag_data.append(tmp_new_data)

            if tmp_tot_data.shape[0] <= day:  # not enough data point
                tmp_data.loc[:, xnam] = np.nan
            else:
                tmp_result.loc[:, xnam] = BETA(data=tmp_tot_data, day=day, halflife=63, weight=all_weights[day])[
                                          -tmp_data_num:]
    return  tmp_result

def callIncrm():
    # *************************************
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    sql_sate = "select max(`date`) from %s" % targetTableName
    target_max_date = pd.read_sql(sql_sate, con_quant)
    if not target_max_date.empty:
        target_max_date = target_max_date.iloc[0,0]
    incrm_start_date = datetime.strptime(target_max_date, '%Y-%m-%d') - timedelta(days=365)
    incrm_start_date = datetime.strftime(incrm_start_date, '%Y-%m-%d')
    sql_asset = "select `code`,date,`close` from %s where date>='%s'" % (sourceStockQuoteTableName, incrm_start_date)
    # sql_olddata = "select * from %s where date>='%s'" % (targetTableName, incrm_start_date)
    # *************************************

    # year_now = datetime.now().year
    # sql_asset = "select `code`,date,`close` from %s where date>='" % sourceStockQuoteTableName \
    #             + str(year_now - 1) + "-01-01'"
    # sql_olddata = "select * from %s where date>='" % targetTableName + str(year_now - 1) + "-01-01'"

    # parameters
    data0 = sql.read_sql(sql_asset, con_quant)
    hs300 = sql.read_sql("select date,close from HS300_QUOTE ", con_quant)
    # data_old = sql.read_sql(sql_olddata, con_quant)

    # data merged
    data1 = pd.merge(data0, hs300, on='date', how='inner', suffixes=('_asset', '_hs300'))
    data1 = data1.sort_values(['code', 'date'], ascending=True)
    data1['code'] = data1['code'].astype(str)

    # time decay weights
    all_weights = {}
    final_columns = ['date', 'code']
    for day in days:
        all_weights[day] = np.array(
            sorted(np.power(.5, np.arange(day) / np.float(halfLife)), reverse=False))
        final_columns.append('beta_{n}'.format(n=day))

    # pool = Pool(processes=2)
    # pool_results = []

    # calculation
    codes = data1['code'].unique().tolist()
    dataresult = pd.DataFrame()
    for code in codes:
        tmp_data = data1[data1['code'] == code]

        # tmp_result = pool.apply_async(calMultiBetas, (tmp_data, target_max_date, all_weights))
        # pool_results.append(tmp_result)

        tmp_result = calMultiBetas(tmp_data, target_max_date, all_weights)

        dataresult = dataresult.append(tmp_result)

    # dataresult = pd.DataFrame()
    # for res in pool_results:
    #     dataresult = dataresult.append(res.get())

    # close multiprocessing pool
    # pool.close()

    if not dataresult.empty:
        dataresult = dataresult[final_columns] # ***********************
        # data_old = data_old[final_columns] # ***********************
        # dataresult.append(data_old)
        dataresult = dataresult.drop_duplicates(subset=['date', 'code'], keep=False)

        # add time stamp
        dataresult.loc[:, timestampField] = datetime.now()

        # dump data
        dataresult.to_sql(targetTableName, con_quant, if_exists='append', index=False) #**********************


def airflowCallable():
    callIncrm()


if __name__ == '__main__':
    # 全量
    calFull()
    t_start = time.clock()
    # 增量
    # callIncrm()
    elasped = time.clock() - t_start
    print(elasped)