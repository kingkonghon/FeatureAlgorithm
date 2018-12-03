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
# import matplotlib.pyplot as plt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# parameters
days = [5, 10, 20, 60, 90, 120, 252]
halfLife = 63

# con = pymysql.connect('10.46.228.175', 'alg', 'Alg#824', 'quant', charset='utf8')

sourceIndexQuoteTableName = 'HS300_QUOTE'

targetTableName = 'DERI_HS300_PEAK_TROUGH'

timestampField = 'time_stamp'

def findNextPeak(chg_pct, chg_threshold, hs300):
    tmp_rets = hs300['close'] / hs300['close'].iloc[0] - 1
    tmp_max_rets = tmp_rets.expanding().max()
    tmp_drawdown = (1 + tmp_rets) / (1 + tmp_max_rets) - 1
    tmp_drawdown_pct = (- tmp_drawdown) / tmp_max_rets
    tmp_turn_point = tmp_drawdown.loc[(tmp_drawdown_pct > chg_pct) & ( (- tmp_drawdown) > chg_threshold)]

    if not tmp_turn_point.empty:
        tmp_turn_point_idx = tmp_turn_point.index[0]

        tmp_peak_idx = tmp_rets.loc[:tmp_turn_point_idx].idxmax()

        r1 = hs300.loc[tmp_peak_idx, 'close'] / hs300['close'].iloc[0] - 1
        r2 = hs300.loc[tmp_turn_point_idx, 'close'] / hs300.loc[tmp_peak_idx, 'close'] - 1
        # print('peak:', r1, r2, (-r2) / r1)

        if (r1 <= chg_threshold) or ((-r2) <= chg_threshold) or ((-r2) <= r1 * chg_pct):
            if hs300.index[0] != 0:
                print('peak error!')
                raise ValueError

        return tmp_peak_idx
    else:
        return

def findNextTrough(chg_pct, chg_threshold, hs300):
    tmp_rets = hs300['close'] / hs300['close'].iloc[0] - 1
    tmp_min_rets = tmp_rets.expanding().min()
    tmp_drawup = (1 + tmp_rets) / (1 + tmp_min_rets) - 1
    tmp_drawup_pct = tmp_drawup / (-tmp_min_rets)
    tmp_turn_point = tmp_drawup.loc[(tmp_drawup_pct > chg_pct) & (tmp_drawup > chg_threshold)]

    if not tmp_turn_point.empty:
        tmp_turn_point_idx = tmp_turn_point.index[0]

        tmp_trough_idx = tmp_rets.loc[:tmp_turn_point_idx].idxmin()

        r1 = hs300.loc[tmp_trough_idx, 'close'] / hs300['close'].iloc[0] - 1
        r2 = hs300.loc[tmp_turn_point_idx, 'close'] / hs300.loc[tmp_trough_idx, 'close'] - 1
        # print('trough:', r1, r2, r2/(-r1))

        if (r2 <= chg_threshold) or (r2 <= (-r1) * chg_pct):
            if hs300.index[0] != 0:
                print('peak error!')
                raise ValueError

        return  tmp_trough_idx
    else:
        return

def calFull(chg_pct, chg_threshold, start_date, end_date):
    # read data
    db_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    con_quant = db_quant.connect()

    sql_statement = "select date,`close` from %s where date between '%s' and '%s'" % (sourceIndexQuoteTableName,
                                                                                      start_date, end_date)
    hs300 = sql.read_sql(sql_statement, con_quant)

    # =============== calculation
    hs300 = hs300.reset_index()
    hs300 = hs300.drop('index', axis=1)   # reset index
    peak_trough_label = 'PeakTrough'
    hs300.loc[:, peak_trough_label] = 0

    # first peak/trough
    tmp_next_peak_idx = findNextPeak(chg_pct, chg_threshold, hs300)
    tmp_next_trough_idx =findNextTrough(chg_pct, chg_threshold, hs300)

    if (tmp_next_peak_idx is None) or (tmp_next_trough_idx is None): # cannot find at least one peak or one trough, return None
        return

    if tmp_next_peak_idx < tmp_next_trough_idx:  # the first extreme point is peak
        hs300.loc[tmp_next_peak_idx, peak_trough_label] = 1
        next_find_peak = False  #   next step is to find trough point
    elif tmp_next_peak_idx > tmp_next_trough_idx:  # the first extreme point is trough
        hs300.loc[tmp_next_trough_idx, peak_trough_label] = -1
        next_find_peak = True   #   next step is to find peak point
    else:
        print('find first peak trough point error')
        raise ValueError

    # from the second peak/trough forward
    while True:
        if next_find_peak:
            tmp_price_serie = hs300.loc[tmp_next_trough_idx:]   # find the next peak from the last trough point
            tmp_next_peak_idx = findNextPeak(chg_pct, chg_threshold, tmp_price_serie)
            if tmp_next_peak_idx is None:
                hs300.loc[hs300.index[-1], peak_trough_label] = 1
                break
            else:
                hs300.loc[tmp_next_peak_idx, peak_trough_label] = 1
                next_find_peak = False
        else:
            tmp_price_serie = hs300.loc[tmp_next_peak_idx:]    # find the next trough from the last peak point
            tmp_next_trough_idx = findNextTrough(chg_pct, chg_threshold, tmp_price_serie)
            if tmp_next_trough_idx is None:
                hs300.loc[hs300.index[-1], peak_trough_label] = -1
                break
            else:
                hs300.loc[tmp_next_trough_idx, peak_trough_label] = -1
                next_find_peak = True

    hs300.loc[:, peak_trough_label] = hs300[peak_trough_label].replace({0: np.nan})
    hs300.loc[:, peak_trough_label] = hs300[peak_trough_label].fillna(method='bfill')

    # # plot
    # hs300.loc[hs300[peak_trough_label].isnull(), peak_trough_label] = 0
    # dt_dates = pd.to_datetime(hs300['date'], format='%Y-%m-%d')
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(dt_dates, hs300['close'])
    #
    # ax2 = ax1.twinx()
    # ax2.plot(dt_dates, hs300[peak_trough_label])
    #
    # plt.show()

    con_quant.close()

    hs300.loc[:, 'ret'] = hs300['close'] / hs300['close'].shift(1)
    return hs300[['date','ret', peak_trough_label]]

if __name__ == '__main__':
    # 全量
    # chg_pct = 0.2
    # chg_threshold = 0.15
    chg_pct = 0.2
    chg_threshold = 0.05
    tot_start_date = '2007-01-01'
    tot_end_date = '2018-08-31'
    peak_trough = calFull(chg_pct, chg_threshold, tot_start_date, tot_end_date)

    testing_start_date = '2013-01-01'
    testing_end_date = '2013-03-01'

    peak_trough = peak_trough.loc[(peak_trough['date'] >= testing_start_date) & (peak_trough['date'] <= testing_end_date)]
    print(peak_trough)