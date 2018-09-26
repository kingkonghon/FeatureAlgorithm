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
sourceTableName = 'east_financing'
sourceField = ['trade_date', 'today_rzye', 'today_rzyezb', 'rzmre', 'three_rzmre', 'five_rzmre', 'ten_rzmre',
               'today_rzche', 'three_rzche', 'five_rzche', 'ten_rzche',
               'today_rzjmre', 'three_rzjmre', 'five_rzjmre', 'ten_rzjmre',
               'rqye', 'rqyl', 'rqmcl', 'three_rqmcl', 'five_rqmcl', 'ten_rqmcl',
               'rqchl', 'three_rqchl', 'five_rqchl', 'ten_rqchl',
               'rqjmcl', 'three_rqjmcl', 'five_rqjmcl', 'ten_rqjmcl',
               'rzrqye', 'rzrqyecz']
sourceTimestampField = 'trade_date'


# TARGET
targetTableName = 'FINANCE_CASH_AND_SECURITY'
targetField = ['date', 'OUTSTANDING_CASH_BALANCE', 'OUTSTANDING_CASH_TO_FREE_CAP', 'FINANCE_CASH_BUY_AMOUNT', 'FINANCE_CASH_BUY_AMOUNT_3D', 'FINANCE_CASH_BUY_AMOUNT_5D', 'FINANCE_CASH_BUY_AMOUNT_10D',
               'FINANCE_CASH_REPAY_AMOUNT', 'FINANCE_CASH_REPAY_AMOUNT_3D', 'FINANCE_CASH_REPAY_AMOUNT_5D', 'FINANCE_CASH_REPAY_AMOUNT_10D',
               'FINANCE_CASH_NET_BUY_AMOUNT', 'FINANCE_CASH_NET_BUY_AMOUNT_3D', 'FINANCE_CASH_NET_BUY_AMOUNT_5D', 'FINANCE_CASH_NET_BUY_AMOUNT_10D',
               'OUTSTANDING_SECURITY_BALANCE', 'OUTSTANDING_SECURITY_NUM', 'FINANCE_SECURITY_SELL_NUM', 'FINANCE_SECURITY_SELL_NUM_3D', 'FINANCE_SECURITY_SELL_NUM_5D', 'FINANCE_SECURITY_SELL_NUM_10D',
               'FINANCE_SECURITY_REPAY_NUM', 'FINANCE_SECURITY_REPAY_NUM_3D', 'FINANCE_SECURITY_REPAY_NUM_5D', 'FINANCE_SECURITY_REPAY_NUM_10D',
               'FINANCE_SECURITY_NET_SELL_NUM', 'FINANCE_SECURITY_NET_SELL_NUM_3D', 'FINANCE_SECURITY_NET_SELL_NUM_5D', 'FINANCE_SECURITY_NET_SELL_NUM_10D',
               'FINANCE_CASH_ADD_SECURITY_AMOUNT', 'FINANCE_CASH_MINUS_SECURITY_AMOUNT']
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
    tmp_rename_dict = dict(zip(sourceField, targetField))
    data_full = data_full.rename(columns=tmp_rename_dict)
    data_full.loc[:, targetTimeStampField] = data_full[targetTimeStampField].apply(lambda x: x[:10])

    # change data type
    tmp_fields = targetField.copy()
    tmp_fields.remove(targetTimeStampField)
    data_full = chgDFDataType(data_full, tmp_fields, 'float')

    # add time stamp
    data_full.loc[:, targetNewTimeStampField] = datetime.now()

    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    data_full.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')


def updateIncrm():
    pass


def airflowCallable():
    updateIncrm()


if __name__ == '__main__':
    updateFull()
    # updateIncrm()
