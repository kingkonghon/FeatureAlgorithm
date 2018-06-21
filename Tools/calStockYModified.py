#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import pandas.io.sql as sql
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from multiprocessing import Pool
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# parameters
YPairs = {
    2: [0.01, 0.02, 0.03, 0.04, 0.05],
    5: [0.02, 0.03, 0.05, 0.07, 0.10],
    10: [0.03, 0.05, 0.07, 0.10, 0.15],
    20: [0.04, 0.07, 0.10, 0.15, 0.20],
    30: [0.05, 0.10, 0.15, 0.20, 0.25]
}

Y_skim_rate = 0.3

sourceStockQuoteTableName = 'STOCK_FORWARD_ADJ_QUOTE'

targetTableName = 'STOCK_HIGH_Y_MODIFIED'

timestampField = 'time_stamp'

def calStockY(data, y_pairs, tot_y_num):
    data_result = data[['date', 'code']].copy()

    for day, all_rets in y_pairs.items():
        future_high = data['high'].rolling(day).max().shift(-day)  # rolling is 1 row less than shift, so it is the high of the next n days
        high_percent = future_high / data['close'] - 1
        for ret in all_rets:
            tmp_label = 'Y_%dD_%dPCT' % (day, int(ret*100))
            data_result.loc[:, tmp_label] = (high_percent > ret).astype('float')
            data_result.loc[high_percent.isnull(), 'Y_%dD_%dPCT' % (day, int(ret*100))] = np.nan
            # skim Y that is very close to 1
            Y_0_high_percent = high_percent[data_result[tmp_label] == 0]
            if Y_0_high_percent.size > 0:
                Y_0_threshold = Y_0_high_percent.quantile(1 - Y_skim_rate)
                data_result.loc[(data_result[tmp_label] == 0) & (high_percent >= Y_0_threshold), tmp_label] = np.nan


    return data_result


def calFull():
    # read data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    data0 = sql.read_sql("select `code`,date,close,high from %s" % sourceStockQuoteTableName, con_quant)

    # calculation
    codes = data0['code'].unique().tolist()
    data_result = pd.DataFrame()

    tot_y_num = 0
    for tmp_val in YPairs.values():
        tot_y_num += len(tmp_val)

    # ====== calculation =======
    # multiprocess pool
    pool = Pool(processes=4)

    # buffer to store multiprocess result
    multi_process_result = []
    for code in codes:
        tmp_data = data0[data0['code'] == code]
        tmp_data = tmp_data.sort_values('date', ascending=True)

        # run multi-process
        tmp_result = pool.apply_async(calStockY, (tmp_data, YPairs, tot_y_num))
        multi_process_result.append(tmp_result)

    # get result from multi process
    for tmp_res in multi_process_result:
        data_result = data_result.append(tmp_res.get())

    # drop columns
    # data_result = data_result[final_columns]
    chunk_size = 500000

    # add time stamp
    data_result.loc[:, timestampField] = datetime.now()

    # dump data
    loop_num = int(data_result.shape[0] / chunk_size)
    if data_result.shape[0] > loop_num * chunk_size:
        loop_num = loop_num + 1

    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    write_method = 'replace'
    for i in range(loop_num):
        chunk_data = data_result.iloc[i*chunk_size : (i+1)*chunk_size]
        chunk_data.to_sql(targetTableName, con_quant, index=False, if_exists=write_method)
        write_method = 'append'


def callIncrm(start_date, end_date):
    # read data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    data0 = sql.read_sql("select `code`,date,close,high from %s where date between '%s' and '%s'" % (
        sourceStockQuoteTableName, start_date, end_date), con_quant)

    # calculation
    codes = data0['code'].unique().tolist()
    data_result = pd.DataFrame()

    tot_y_num = 0
    for tmp_val in YPairs.values():
        tot_y_num += len(tmp_val)

    # ====== calculation =======
    # multiprocess pool
    pool = Pool(processes=4)

    # buffer to store multiprocess result
    multi_process_result = []
    for code in codes:
        tmp_data = data0[data0['code'] == code]
        tmp_data = tmp_data.sort_values('date', ascending=True)

        # run multi-process
        tmp_result = pool.apply_async(calStockY, (tmp_data, YPairs, tot_y_num))
        multi_process_result.append(tmp_result)

    # get result from multi process
    for tmp_res in multi_process_result:
        data_result = data_result.append(tmp_res.get())

    # drop columns
    # data_result = data_result[final_columns]
    chunk_size = 500000

    # add time stamp
    data_result.loc[:, timestampField] = datetime.now()

    # dump data
    loop_num = int(data_result.shape[0] / chunk_size)
    if data_result.shape[0] > loop_num * chunk_size:
        loop_num = loop_num + 1

    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    write_method = 'replace'
    for i in range(loop_num):
        chunk_data = data_result.iloc[i * chunk_size: (i + 1) * chunk_size]
        chunk_data.to_sql(targetTableName, con_quant, index=False, if_exists=write_method)
        write_method = 'append'


if __name__ == '__main__':
    # 全量
    # calFull()
    # 增量
    start_date = '2017-01-01'
    end_date = '2018-05-31'
    callIncrm(start_date, end_date)