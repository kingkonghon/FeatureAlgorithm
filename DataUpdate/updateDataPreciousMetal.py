
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
sourceTableName = 'guijinshu'
sourceFields = ['report_date', 'contract', 'opening_price', 'top_bid_price', 'lowest_price', 'closing_price',
                'change_amount', 'price_change_ratio', 'VWAP', 'VOL', 'AMOUNT']
sourceTimeStamp = 'report_date'

# TARGET
targetTableName = 'PRECIOUS_METAL'
targetFields = ['date', 'code', 'open', 'high', 'low', 'close', 'change', 'change_ratio', 'VWAP', 'volume', 'amount']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = ['change', 'VWAP', 'volume', 'amount']

def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s`" % (tmp_fields, sourceTableName)
    full_data = pd.read_sql(sql_statement, spider_engine)

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # change data type
    full_data = full_data.replace('--', 'nan')
    full_data.loc[:, 'change_ratio'] = full_data['change_ratio'].apply(lambda x: float(x.strip('%')) / 100.)
    full_data.loc[:, 'volume'] = full_data['volume'].apply(lambda x:x.replace(',', ''))
    full_data.loc[:, 'amount'] = full_data['amount'].apply(lambda x: x.replace(',', ''))
    full_data = chgDFDataType(full_data, chgDataTypeCol, 'float')

    # drop duplicates
    full_data = full_data.drop_duplicates(['code', 'date'])

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

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['code', 'date'])

    if not incrm_data.empty:
        # change data type
        incrm_data = incrm_data.replace('--', 'nan')
        incrm_data.loc[:, 'change_ratio'] = incrm_data['change_ratio'].apply(lambda x: float(x.strip('%')) / 100.)
        incrm_data.loc[:, 'volume'] = incrm_data['volume'].apply(lambda x: x.replace(',', ''))
        incrm_data.loc[:, 'amount'] = incrm_data['amount'].apply(lambda x: x.replace(',', ''))
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # sort by date
        incrm_data = incrm_data.sort_values(targetTimeStamp)

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
