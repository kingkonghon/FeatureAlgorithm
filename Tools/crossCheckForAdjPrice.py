import tushare as ts
from sqlalchemy import create_engine
from Utils.DB_config import ConfigSpider
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


hfqTableName = 'EastMoneyHouFuQuan'
oriTableName = ''
Field = 'close'

h5FileNameAdjFactor = 'LZ_CN_STKA_CMFTR_CUM_FACTOR.h5'
h5FileNameIndClosePrice = 'LZ_CN_STKA_QUOTE_TCLOSE.h5'

def getTushareData(code, start_date, end_date):
    hfq_data = ts.get_k_data(code=code, ktype='D', autype='hfq', index=False, start=start_date, end=end_date)
    ori_data = ts.get_k_data(code=code, ktype='D', autype=None, index=False, start=start_date, end=end_date)
    return hfq_data, ori_data

def getDBData(sql_engine, code, start_date, end_date):
    sql_statement = "select `report_time`, `%s` from `%s` where `code` = '%s'" % (Field, hfqTableName, code)
    data = pd.read_sql(sql_statement, sql_engine)
    data = data.rename(columns={'report_time': 'date'})
    data.loc[:, Field] = data[Field].astype('float')
    data = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
    return data

def getH5Data(single_code):
    h5_file = h5py.File(h5FileNameIndClosePrice)
    # dataset
    ds_data = h5_file['data']
    ds_code = h5_file['header']
    ds_tradedates = h5_file['date']

    quote = ds_data[...]
    code = ds_code[...]
    tradedates = ds_tradedates[...]

    # tranform code format
    code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), code)))  # remove the market signs

    single_quote = quote[:, code == single_code]

    h5_file = h5py.File(h5FileNameAdjFactor)
    # dataset
    ds_data = h5_file['data']
    ds_code = h5_file['header']
    ds_tradedates = h5_file['date']

    adj_factor = ds_data[...]
    factor_code = ds_code[...]
    factor_tradedates = ds_tradedates[...]

    # tranform code format
    factor_code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), factor_code)))  # remove the market signs

    single_adj_factor = adj_factor[:, factor_code == single_code]
    single_adj_factor = single_adj_factor[np.in1d(factor_tradedates, tradedates)]

    # transform date format
    tradedates = list(map(lambda x: str(x), tradedates))
    tradedates = np.array(list(map(lambda x: '-'.join([x[0:4], x[4:6], x[6:]]), tradedates)))

    single_quote = single_quote.T[0]
    single_quote = pd.Series(single_quote, index=tradedates)
    single_adj_factor = single_adj_factor.T[0]
    single_adj_factor = pd.Series(single_adj_factor, index=tradedates)

    return single_quote, single_adj_factor


def crossCheckData(ts_hfq_quote, db_hfq_quote, ori_data, wind_ori_quote, wind_adj_factor):
    ts_hfq_close = ts_hfq_quote['close']
    ts_hfq_close.index = ts_hfq_quote['date']
    ori_close = ori_data['close']
    ori_close.index = ori_data['date']
    db_hfq_close = db_hfq_quote['close']
    db_hfq_close.index = db_hfq_quote['date']

    ts_factor = ts_hfq_close / ori_close
    db_factor = db_hfq_close / ori_close

    tot_adj_factor = pd.concat([ts_factor, db_factor, wind_adj_factor], axis=1, join='inner')
    tot_adj_factor.columns = ['ts', 'db', 'wind']
    pass

def plotTrend(ts_hfq_quote, db_hfq_quote, wind_hfq_close):
    ts_hfq_close = ts_hfq_quote['close']
    ts_hfq_close.index = ts_hfq_quote['date']
    db_hfq_close = db_hfq_quote['close']
    db_hfq_close.index = db_hfq_quote['date']

    ts_hfq_cum_ret = ts_hfq_close / ts_hfq_close.iloc[0]
    db_hfq_cum_ret = db_hfq_close / db_hfq_close.iloc[0]
    wind_hfq_cum_ret = wind_hfq_close / wind_hfq_close.iloc[0]

    all_ret = pd.concat([ts_hfq_cum_ret, db_hfq_cum_ret, wind_hfq_cum_ret], axis=1, join='inner')
    all_ret = all_ret.reset_index()
    all_ret.columns = ['date', 'ts_ret', 'db_ret', 'wind_ret']

    dt = all_ret['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    plt.plot(dt, all_ret['ts_ret'], label='ts')
    plt.plot(dt, all_ret['db_ret'], label='em')
    plt.plot(dt, all_ret['wind_ret'], label='wind')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider))
    code = '600000'
    start_date = '2005-01-01'
    end_date = '2018-04-24'

    ts_hfq_quote, ori_quote = getTushareData(code, start_date, end_date)
    db_hfq_quote = getDBData(spider_engine, code, start_date, end_date)
    wind_ori_quote, wind_adj_factor = getH5Data(code)

    wind_hfq_close = wind_ori_quote * wind_adj_factor
    wind_hfq_close = wind_hfq_close.loc[~wind_hfq_close.isnull()]
    plotTrend(ts_hfq_quote, db_hfq_quote, wind_hfq_close)

    crossCheckData(ts_hfq_quote, db_hfq_quote,ori_quote, wind_ori_quote, wind_adj_factor)