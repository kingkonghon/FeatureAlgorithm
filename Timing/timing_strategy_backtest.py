from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

def backTest(start_date='2007-01-01', end_date='2020-12-31'):
    sourceIndexQuoteTableName = 'HS300_QUOTE'

    # read data
    con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    sql_statement = "select date,`close` from %s" % sourceIndexQuoteTableName
    hs300 = pd.read_sql(sql_statement, con_quant)

    hs300.loc[:, 'ret'] = hs300['close'] / hs300['close'].shift(1)

    # read prediction
    folder_path = 'D:/FeatureAlgorithm/Timing/'
    # crf_predict = pd.read_csv(folder_path + 'timing_prediction.csv')
    crf_predict = pd.read_csv(folder_path + 'lgbm_timing_prediction.csv')

    # get prediction p&l
    strategy = hs300[['date', 'ret']]
    strategy = strategy.merge(crf_predict, on='date', how='left')
    strategy.loc[:, 'predict'] = strategy['predict'].shift(1)  # predict tomorrow
    strategy = strategy.loc[~strategy['predict'].isnull()]  # drop nan

    hs300 = hs300.loc[hs300['date'].isin(strategy['date'])]  # trim by date
    hs300 = hs300.loc[(hs300['date'] >= start_date) & (hs300['date'] <= end_date)]
    hs300.loc[:, 'cum_ret'] = hs300['ret'].cumprod()  # index cum retunrns

    strategy.loc[:, 'real_ret'] = strategy.apply(lambda x: x['ret'] if x['predict'] == 1 else 1, axis=1)
    strategy = strategy.loc[(strategy['date'] >= start_date) & (strategy['date'] <= end_date)]
    strategy.loc[:, 'real_cum_ret'] = strategy['real_ret'].cumprod()

    # plot
    # fig, ax1 = plt.subplots()
    dt_dates = pd.to_datetime(hs300['date'], format='%Y-%m-%d').values
    plt.plot(dt_dates, hs300['cum_ret'].values, 'b')

    # ax2 = ax1.twinx()
    plt.plot(dt_dates, strategy['real_cum_ret'].values, 'r')
    plt.legend(['HS300', 'Strategy'])

    plt.show()


if __name__ == '__main__':
    start_date = '2017-01-01'
    end_date = '2018-08-31'
    backTest(start_date, end_date)