import tushare as ts
from sqlalchemy import create_engine
from Utils.DB_config import ConfigSpider2
import pandas as pd
import h5py
import numpy as np


sourceTableName = 'tushare_real_time_market'
Field = 'mktcap'
codeField = 'code'
dateField = 'time_stamp'

h5FileFactor = 'LZ_CN_STKA_VAL_A_TCAP.h5'


def getTushareData(date):
    data = ts.get_stock_basics(date)
    # data = ts.get_today_all()
    return data

def getDBData(sql_engine, code):
    sql_statement = "select `%s`, `%s`, %s from %s where `%s` = '%s'" % (dateField, codeField, Field, sourceTableName, codeField, code)
    data = pd.read_sql(sql_statement, sql_engine)

    return data

def getH5Data():
    h5_file = h5py.File(h5FileFactor)
    # dataset
    ds_data = h5_file['data']
    ds_code = h5_file['header']
    ds_tradedates = h5_file['date']

    factor = ds_data[...]
    code = ds_code[...]
    tradedates = ds_tradedates[...]

    # tranform code format
    code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), code)))  # remove the market signs

    tradedates = list(map(lambda x: str(x), tradedates))
    tradedates = np.array(list(map(lambda x: '-'.join([x[0:4], x[4:6], x[6:]]), tradedates)))

    return factor, code, tradedates


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

if __name__ == '__main__':
    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))
    start_date = '2005-01-01'
    end_date = '2018-04-25'

    date = '2018-05-03'
    code = '000002'

    ts_factor = getTushareData(date)
    db_factor = getDBData(spider_engine, code)
    wind_factor, code, trade_dates = getH5Data()

    crossCheckData(ts_hfq_quote, db_hfq_quote,ori_quote, wind_ori_quote, wind_adj_factor)