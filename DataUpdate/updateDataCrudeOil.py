# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pymysql
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import chgDFDataType

# SOURCE
sourceTableName = 'SinaBulunte'
sourceField = ['reportTime', 'open', 'high', 'low', 'close']
sourceTimestampField = 'reportTime'
sourceNameField = 'stockName'
sourceOilName = 'OIL'


# TARGET
targetTableName = 'BRENT_QUOTE'
chgDataTypeCol = ['open', 'high', 'low', 'close']
targetTimeStampField = 'date'

targetNewTimeStampField = 'time_stamp'



def updateFull(start_date='2007-01-01'):
    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    tmp_fields = map(lambda x:'`%s`' % x, sourceField)
    tmp_fields = ','.join(tmp_fields)

    # get data from file
    sql_statement = "select %s from %s" % (tmp_fields, sourceTableName)
    data_full = pd.read_sql(sql_statement, spider_engine)

    # change column name
    data_full = data_full.rename(columns={sourceTimestampField: targetTimeStampField})

    # change data type
    data_full = chgDFDataType(data_full, chgDataTypeCol, 'float')

    # add time stamp
    data_full.loc[:, targetNewTimeStampField] = datetime.now()

    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    data_full.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')


def updateIncrm():
    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # get lastest tradedate
    sql_statement = "select max(`%s`) from %s" % (targetTimeStampField, targetTableName)
    latest_date = pd.read_sql(sql_statement, quant_engine)
    if not latest_date.empty:
        latest_date = latest_date.iloc[0, 0]

    tmp_fields = map(lambda x: '`%s`' % x, sourceField)
    tmp_fields = ','.join(tmp_fields)

    # get data from file
    sql_statement = "select %s from %s where %s > '%s'" % (tmp_fields, sourceTableName, sourceTimestampField, latest_date)
    data_incrm = pd.read_sql(sql_statement, spider_engine)

    if data_incrm.empty:
        return

    # change column name
    data_incrm = data_incrm.rename(columns={sourceTimestampField: targetTimeStampField})

    # change data type
    data_incrm = chgDFDataType(data_incrm, chgDataTypeCol, 'float')

    # add time stamp
    data_incrm.loc[:, targetNewTimeStampField] = datetime.now()

    data_incrm.to_sql(targetTableName, quant_engine, index=False, if_exists='append')


def airflowCallable():
    updateIncrm()


if __name__ == '__main__':
    # updateFull()
    updateIncrm()
