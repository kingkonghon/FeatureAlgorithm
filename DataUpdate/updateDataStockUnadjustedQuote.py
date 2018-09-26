# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
# import h5py
import numpy as np
# import tushare as ts
import os
import sys
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider, ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'EastMoneyBuFuQuan'
sourceFields = ['report_time', 'code', 'open', 'high', 'low', 'close', 'VOL', 'amount', 'turnover']
sourceTimeStamp = 'report_time'
sourceCode = 'code'

# TARGET
targetTableName = 'STOCK_UNADJUSTED_QUOTE'
targetFields = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'] # after combining main and supplement data
chgDataTypeCol = ['open', 'high', 'low', 'close', 'volume']
targetTimeStamp = 'date'

targetNewTimeStamp = 'time_stamp'


def updateFull(quant_engine, spider_engine, chunk_size, start_date='2007-01-01'):
    # get distinct code
    sql_statement = "select distinct `%s` from %s where %s >= '%s'" % (sourceCode, sourceTableName, sourceTimeStamp, start_date)
    tot_codes = pd.read_sql(sql_statement, spider_engine)
    tot_codes = tot_codes.values.T[0]

    # drop B share codes
    tmp_idx = list(map(lambda x: x[0] != '9', tot_codes))
    tot_codes = tot_codes[tmp_idx]

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    write_method = 'replace'
    loop_num = int(tot_codes.size / chunk_size)
    if tot_codes.size > loop_num * chunk_size:
        loop_num += 1
    for i in range(loop_num):
        tmp_code = tot_codes[i*chunk_size:(i+1)*chunk_size]
        tmp_code_str = list(map(lambda x:"'%s'"%x, tmp_code))
        tmp_code_str = ','.join(tmp_code_str)

        sql_statement = "select %s from %s where (`%s` > '%s') and (`%s` != 'null') and (`%s` in (%s))" % (tmp_fields, sourceTableName, sourceTimeStamp,
                        start_date, sourceTimeStamp, sourceCode, tmp_code_str)
        chunk_data = pd.read_sql(sql_statement, spider_engine)

        # rename columns
        rename_dict = {}
        for field in zip(sourceFields, targetFields):
            rename_dict[field[0]] = field[1]
        chunk_data = chunk_data.rename(columns=rename_dict)

        # drop duplicates
        chunk_data = chunk_data.drop_duplicates(['date', 'code'])

        # change data type
        chunk_data.loc[:, 'amount'] = chunk_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (
                float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
        chunk_data.loc[:, 'turnover'] = chunk_data['turnover'].apply(lambda x: x if x != '-' else 0)
        chunk_data.loc[:, 'turnover'] = chunk_data['turnover'].apply(lambda x: float(x) if x != 'null' else np.nan)
        chunk_data = chgDFDataType(chunk_data, chgDataTypeCol, 'float')

        # add time stamp
        chunk_data[targetNewTimeStamp] = datetime.now()

        # write data to db
        chunk_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
        write_method = 'append'

    pass

def updateIncrm(quant_engine, spider_engine):
    # get lastest tradedate
    sql_statement = "select max(`%s`) from %s where `%s` != 'null'" % (targetTimeStamp, targetTableName, targetTimeStamp)
    latest_date = pd.read_sql(sql_statement, quant_engine).iloc[0,0]

    # get incremental data
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from %s where (%s > '%s') and (%s != 'null')" % (tmp_fields, sourceTableName, sourceTimeStamp,
                                                                                latest_date, sourceTimeStamp)
    incrm_data = pd.read_sql(sql_statement, spider_engine)

    print('data from spider:', incrm_data.shape)

    # rename data
    rename_dict = {}
    for field in zip(sourceFields, targetFields):
        rename_dict[field[0]] = field[1]
    incrm_data = incrm_data.rename(columns=rename_dict)

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    # drop B shares
    incrm_data = incrm_data.loc[incrm_data['code'].apply(lambda x: x[0] != '9')]

    # change data type
    incrm_data.loc[:, 'amount'] = incrm_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (
        float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
    incrm_data.loc[:, 'turnover'] = incrm_data['turnover'].apply(lambda x: x if x != '-' else 0)
    incrm_data.loc[:, 'turnover'] = incrm_data['turnover'].apply(lambda x: float(x) if x != 'null' else np.nan)
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

    # add time stamp
    incrm_data[targetNewTimeStamp] = datetime.now()

    print('data to write:', incrm_data.shape)

    # write data to db
    if not incrm_data.empty:
        incrm_data.to_sql(targetTableName, quant_engine, index=False, if_exists='append')
    pass

def airflowCallable():
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    chunk_size = 10

    # updateFull(quant_engine, spider_engine, chunk_size)
    # supplementForAdjQuote(quant_engine)
    updateIncrm(quant_engine, spider_engine)


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    chunk_size = 10
    t_start = time.clock()

    # updateFull(quant_engine, spider_engine, chunk_size)

    # updateIncrm(quant_engine, spider_engine)

    print(time.clock() - t_start)