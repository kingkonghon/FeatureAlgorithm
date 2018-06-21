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
sourceTableName = 'shibor'
sourceFields = ['ReportDate', 'TimeLimitO_N', 'TimeLimit1W', 'TimeLimit2W', 'TimeLimit1M', 'TimeLimit3M', 'TimeLimit6M',
                'TimeLimit9M', 'TimeLimit1Y']
sourceTimeStamp = 'ReportDate'

# TARGET
targetTableName = 'SHIBOR_NEW'
targetFields = ['date', 'O_N', '1D', '2D', '1M', '3M', '6M', '9M', '1Y']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = ['O_N', '1D', '2D', '1M', '3M', '6M', '9M', '1Y']


def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s`" % (tmp_fields, sourceTableName)
    full_data = pd.read_sql(sql_statement, spider_engine)

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # 爬虫库周末也有运行，会爬到重复的数据，所以需要用drop_duplicates
    full_data = full_data.drop_duplicates(targetTimeStamp)

    # change data type
    full_data = chgDFDataType(full_data, chgDataTypeCol, 'float')

    # change datetime format
    full_data.loc[:, targetTimeStamp] = full_data[targetTimeStamp].apply(lambda x: x[:10])
    full_data = full_data.sort_values(targetTimeStamp)

    # add time stamp
    full_data[targetNewTimeStamp] = datetime.now()

    # write data tot target
    if not full_data.empty:
        full_data.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')

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

    # 爬虫库周末也有运行，会爬到重复的数据，所以需要用drop_duplicates
    incrm_data = incrm_data.drop_duplicates(targetTimeStamp)

    # change data type
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

    # change datetime format
    incrm_data.loc[:, targetTimeStamp] = incrm_data[targetTimeStamp].apply(lambda x: x[:10])
    incrm_data = incrm_data.loc[incrm_data[targetTimeStamp] > target_max_timestamp]
    incrm_data = incrm_data.sort_values(targetTimeStamp)

    # add time stamp
    incrm_data[targetNewTimeStamp] = datetime.now()

    # write data tot target
    if not incrm_data.empty:
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

    updateFull(quant_engine, spider_engine)
    # updateIncrm(quant_engine, spider_engine)