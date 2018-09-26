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
import matplotlib.pyplot as plt

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

def assignNotch(x, boundries):
    if np.isnan(x):
        return np.nan

    notch = np.nan
    for i, bd in enumerate(boundries):
        if x <= bd:
            notch = i
            break

    if np.isnan(notch):   # the maximum notch
        notch = len(boundries)

    return notch

def calFull():
    # read data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # sql_statement = "select date,`close` from %s where date between '%s' and '%s'" % (sourceIndexQuoteTableName,
    #                                                                                   start_date, end_date)
    sql_statement = "select date,`close` from %s" % sourceIndexQuoteTableName
    hs300 = sql.read_sql(sql_statement, con_quant)

    # =============== calculation
    hs300 = hs300.reset_index()
    hs300 = hs300.drop('index', axis=1)   # reset index

    hs300.loc[:, 'ret'] = hs300['close'] / hs300['close'].shift(1)

    # notch_num = 6
    # percent = np.array(range(1, notch_num + 1)) * (1. / notch_num)
    # notch_boundary = hs300['close'].quantile(percent, interpolation='midpoint')
    notch_boundary = [0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]

    hs300.loc[:, 'ret_notch'] = hs300['ret'].apply(lambda x: assignNotch(x, notch_boundary))

    return hs300[['date', 'ret_notch']]

if __name__ == '__main__':
    # 全量
    calFull()