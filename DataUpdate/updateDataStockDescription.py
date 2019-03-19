# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'stock_list'
sourceFields = ['code', 'name', 'industry', 'area', 'list_date']
sourceTimeStamp = 'update_time'

# TARGET
targetTableName = 'STOCK_DESCRIPTION'
targetFields = ['code', 'name', 'industry', 'area', 'list_date']
targetNewTimeStamp = 'time_stamp'


def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s`" % (tmp_fields, sourceTableName)
    full_data = pd.read_sql(sql_statement, spider_engine)

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # change list date format
    full_data = full_data.loc[full_data['list_date'] != '0']
    full_data.loc[:, 'list_date'] = full_data['list_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))

    # drop duplicates
    full_data = full_data.drop_duplicates('code')

    # add time stamp
    full_data[targetNewTimeStamp] = datetime.now()

    # write data tot target
    if not full_data.empty:
        full_data.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')

def updateIncrm(quant_engine, spider_engine):
    pass


def airflowCallable():
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    updateFull(quant_engine, spider_engine)
    # updateIncrm(quant_engine, spider_engine)

if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    updateFull(quant_engine, spider_engine)
    # updateIncrm(quant_engine, spider_engine)
