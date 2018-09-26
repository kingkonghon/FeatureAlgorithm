
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
sourceTableName = 'shangzhengjijinzhishu'
sourceFields = ['report_time', 'open_price', 'top_price', 'low_price', 'close_price', 'VOL',
                'Transaction_Amount']
sourceCodeField = 'stock'
sourceStockCode = ' 000011'  # space before code !!
sourceTimeStamp = 'report_time'

# TARGET
targetTableName = 'FUND_INDEX'
targetFields = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = [ 'open', 'high', 'low', 'close', 'volume', 'amount']

def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where %s = '%s'" % (tmp_fields, sourceTableName, sourceCodeField, sourceStockCode)
    full_data = pd.read_sql(sql_statement, spider_engine)

    # drop duplicates & sort value
    full_data = full_data.drop_duplicates(sourceTimeStamp)
    full_data = full_data.sort_values(sourceTimeStamp)

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # change data type
    full_data = chgDFDataType(full_data, chgDataTypeCol, 'float')

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
    sql_statement = "select %s from `%s` where (`%s` > '%s') and (%s = '%s')" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp, sourceCodeField, sourceStockCode)
    incrm_data = pd.read_sql(sql_statement, spider_engine)

    # drop duplicates and sort values
    incrm_data = incrm_data.drop_duplicates(sourceTimeStamp)
    incrm_data = incrm_data.sort_values(sourceTimeStamp)

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # change data type
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

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

    # updateFull(quant_engine, spider_engine)
    updateIncrm(quant_engine, spider_engine)
