# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'EastMoneyHouFuQuan'
sourceFields = ['report_time', 'code']
sourceTimeStamp = 'report_time'

# TARGET
targetTableName = 'STOCK_FLAG'
targetFields = ['date', 'code']
targetFieldsNew = ['ST_FLAG', 'CSI500_FLAG', 'HS300_FLAG']  # data download from tushare
targetTimeStamp = 'date'

targetNewTimeStamp = 'time_stamp'


def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):

    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, quant_engine) # quant schema
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where (`%s` > '%s') and (%s != 'null')" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp, sourceTimeStamp)
    incrm_data = pd.read_sql(sql_statement, spider_engine) # spider schema

    if incrm_data.empty:
        return

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    if incrm_data.empty:
        return

    # start to download data
    for tmp_col in targetFieldsNew:
        incrm_data.loc[:, tmp_col] = 0  # initialized

    # st flag
    tmp_stocks = ts.get_st_classified()
    incrm_data.loc[incrm_data['code'].isin(tmp_stocks['code']), 'ST_FLAG'] = 1

    # CSI500 flag
    tmp_stocks = ts.get_zz500s()
    incrm_data.loc[incrm_data['code'].isin(tmp_stocks['code']), 'CSI500_FLAG'] = 1

    # HS300 flag
    tmp_stocks = ts.get_hs300s()
    incrm_data.loc[incrm_data['code'].isin(tmp_stocks['code']), 'HS300_FLAG'] = 1

    # add time stamp & write data to target
    if not incrm_data.empty:
        incrm_data[targetNewTimeStamp] = datetime.now()  # add time stamp
        incrm_data.to_sql(targetTableName, quant_engine, index=False, if_exists='append')

def airflowCallable():
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # updateFull(quant_engine, spider_engine)
    updateIncrm(quant_engine, spider_engine)


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # updateFull(quant_engine, spider_engine)
    updateIncrm(quant_engine, spider_engine)
