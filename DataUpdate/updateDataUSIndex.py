# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'BaiduStockIndex'
sourceFields = ['time_stamp', 'open', 'high', 'low', 'close', 'volume', 'netChangeRatio']
sourceIndexCode = ['.DJI', '.INX', '.IXIC']
sourceTimeStamp = 'time_stamp'
sourceCodeField = 'stockCode'

# TARGET
targetTableName = ['DOW_JONES_QUOTE', 'SP500_QUOTE', 'NASDAQ_COMPOSITE_QUOTE']
targetFields = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
chgDataTypeCol = ['open', 'high', 'low', 'close', 'volume', 'change']
targetTimeStamp = 'date'

targetNewTimeStamp = 'time_stamp'

# TRADE CALENDAR (use trade calendar to drop duplicate dates, because spider also creep data in holiday)
tradeCalendarTableName = 'TRADE_CALENDAR'
calendarField = 'date'


def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    # get trade calendar
    sql_statement = 'select `%s` from %s' % (calendarField, tradeCalendarTableName)
    trade_calendar = pd.read_sql(sql_statement, quant_engine)
    trade_calendar = trade_calendar.values.T[0]

    # get target latest date
    for (table_name, index_code) in zip(targetTableName, sourceIndexCode):
        sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, table_name)
        target_max_timestamp = pd.read_sql(sql_statement, quant_engine) # quant schema
        target_max_timestamp = target_max_timestamp.iloc[0, 0]

        # fetch data from source
        tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
        tmp_fields = ','.join(tmp_fields)
        sql_statement = "select %s from `%s` where `%s` > '%s' and `%s` = '%s'" % (
            tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp, sourceCodeField, index_code)
        incrm_data = pd.read_sql(sql_statement, spider_engine) # spider schema

        if incrm_data.empty:
            return

        # rename columns
        incrm_data = renameDF(incrm_data, sourceFields, targetFields)

        # change date format
        tmp_date = incrm_data['date'].apply(lambda x: datetime.strptime(x[:11], r'%Y年%m月%d日')) - timedelta(days=1)
        incrm_data.loc[:, 'date'] = tmp_date.apply(lambda x:datetime.strftime(x, '%Y-%m-%d'))
        incrm_data = incrm_data.loc[incrm_data['date'] > target_max_timestamp]

        # drop duplicates
        incrm_data = incrm_data.drop_duplicates(['date'])
        incrm_data = incrm_data.loc[incrm_data['date'].isin(trade_calendar)] # holiday also creep data

        if incrm_data.empty:
            return

        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')
        incrm_data.loc[:, 'change'] = incrm_data['change'] / 100

        # add time stamp & write data to target
        if not incrm_data.empty:
            incrm_data[targetNewTimeStamp] = datetime.now()
            incrm_data.to_sql(table_name, quant_engine, index=False, if_exists='append')

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
