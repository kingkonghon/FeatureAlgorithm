
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'tushare_real_time_market'
sourceFields = ['time_stamp', 'code', 'pb', 'mktcap', 'nmc']
sourceTimeStamp = 'time_stamp'

# TARGET
targetTableName = 'STOCK_FUNDAMENTAL_BASIC'
targetFields = ['date', 'code', 'PB', 'TOT_MRK_CAP', 'FREE_MRK_CAP']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = ['PB', 'TOT_MRK_CAP', 'FREE_MRK_CAP']

# Trade Calendar
calendarTableName = 'TRADE_CALENDAR'
calendarField = 'date'


def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, quant_engine)
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp)
    incrm_data = pd.read_sql(sql_statement, spider_engine)

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # change date format
    incrm_data.loc[:, 'date'] = incrm_data['date'].apply(lambda x: x[:10])
    incrm_data = incrm_data.loc[incrm_data['date'] > target_max_timestamp]

    # use trade calendar to drop duplicates
    sql_statement = "select `%s` from %s" % (calendarField, calendarTableName)
    calendar = pd.read_sql(sql_statement, quant_engine).values.T[0]
    incrm_data = incrm_data.loc[incrm_data['date'].isin(calendar)]

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    if not incrm_data.empty:
        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # add time stamp
        incrm_data[targetNewTimeStamp] = datetime.now()

        # write data tot target
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
