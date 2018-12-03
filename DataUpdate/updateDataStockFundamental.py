
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
from Utils.ProcessFunc import renameDF, chgDFDataType, writeDB

# SOURCE TUSAHRE (OLD)
sourceTableName = 'tushare_real_time_market'
sourceFields = ['time_stamp', 'code', 'pb', 'mktcap', 'nmc']
sourceTimeStamp = 'time_stamp'

# SOURCE TENCENT
sourceTCTableName = 'TencentStock_daily_data'
sourceTCFields = ['time_stamp', 'code', 'pb', 'mktcap', 'nmc']
sourceTCTimeStamp = 'time_stamp'

# SOURCE TUSHARE (NEW)
sourceTSNewTableName = 'STOCK_BASIC_TUSHARE'
sourceTSNewFields = ['date', 'code', 'pb', 'TOT_MRK_CAP', 'FREE_MRK_CAP']
sourceTSNewTimeStamp = 'date'

# TARGET
targetTableName = 'STOCK_FUNDAMENTAL_BASIC'
targetFields = ['date', 'code', 'PB', 'TOT_MRK_CAP', 'FREE_MRK_CAP']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = ['PB', 'TOT_MRK_CAP', 'FREE_MRK_CAP']

# Trade Calendar
calendarTableName = 'TRADE_CALENDAR'
calendarField = 'date'


def updateTCIncrm(sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num):
    tmp_fields = list(map(lambda x: '`%s`' %x, sourceTCFields))
    tmp_fields = ','.join(tmp_fields)

    # read Tencent daily data from db
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (tmp_fields, sourceTCTableName, sourceTCTimeStamp, target_max_timestamp)
    tc_data = pd.read_sql(sql_statement, sql_conn_spider)

    # process raw data
    tc_data.loc[:, 'date'] = tc_data['time_stamp'].apply(lambda x: x[:8])
    tc_data.loc[:, 'date'] = tc_data['date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))  # convert date format

    chgDict = dict(zip(sourceTCFields[2:], targetFields[2:]))
    tc_data = tc_data.rename(chgDict, axis=1)    # change column names

    for tmp_col in targetFields[2:]:
        tc_data.loc[:, tmp_col] = tc_data[tmp_col].replace('', 'nan')  # '' cannot be converted to float
        tc_data.loc[:, tmp_col] = tc_data[tmp_col].astype('float')  # convert data type

    tc_data.loc[:, 'TOT_MRK_CAP'] = tc_data['TOT_MRK_CAP'] * 10000
    tc_data.loc[:, 'FREE_MRK_CAP'] = tc_data['FREE_MRK_CAP'] * 10000

    tc_data = tc_data.drop('time_stamp',axis=1)

    # drop duplicates
    tc_data = tc_data.drop_duplicates(['date','code'])

    tc_data = tc_data[targetFields]

    incrm_data_date_num = tc_data['date'].unique().size

    if incrm_data_date_num == supposed_date_num:
        # add timestamp
        tc_data.loc[:, 'time_stamp'] = datetime.now()

        writeDB(sql_conn_quant, targetTableName, tc_data)

        return True
    else:
        return False

def updateTSNewIncrm(sql_conn_quant, target_max_timestamp, supposed_date_num):
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceTSNewFields))
    tmp_fields = ','.join(tmp_fields)

    # read Tushare(new version) daily data from db
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (
        tmp_fields, sourceTSNewTableName, sourceTSNewTimeStamp, target_max_timestamp)
    tc_data = pd.read_sql(sql_statement, sql_conn_quant)

    # rename columns
    tc_data = renameDF(tc_data, sourceTSNewFields, targetFields)
    tc_data = tc_data.sort_values(sourceTSNewTimeStamp, ascending=True)

    # drop duplicates
    tc_data = tc_data.drop_duplicates(['date', 'code'])

    incrm_data_date_num = tc_data['date'].unique().size

    if incrm_data_date_num == supposed_date_num:
        # add timestamp
        tc_data.loc[:, 'time_stamp'] = datetime.now()

        writeDB(sql_conn_quant, targetTableName, tc_data)

        return True    # data written to db successfully
    else:
        return False    # missing data


def updateTSOldIncrm(calendar, sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp)
    incrm_data = pd.read_sql(sql_statement, sql_conn_spider)

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # change date format
    incrm_data.loc[:, 'date'] = incrm_data['date'].apply(lambda x: x[:10])
    incrm_data = incrm_data.loc[incrm_data['date'] > target_max_timestamp]

    # use trade calendar to drop duplicates
    incrm_data = incrm_data.loc[incrm_data['date'].isin(calendar)]

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    incrm_data_date_num = incrm_data['date'].unique().size

    if incrm_data_date_num == supposed_date_num:  # records from source 2 are complete
        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # add time stamp
        incrm_data[targetNewTimeStamp] = datetime.now()

        # write data tot target
        writeDB(sql_conn_quant, targetTableName, incrm_data)

        return True  # successfully
    else:
        return False  # missing data in this source

    return incrm_data

def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    sql_conn_quant = quant_engine.connect()
    sql_conn_spider = spider_engine.connect()

    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, sql_conn_quant)
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    # get trade calendar, and supposed missing dates
    sql_statement = "select `%s` from %s" % (calendarField, calendarTableName)
    calendar = pd.read_sql(sql_statement, sql_conn_quant).values.T[0]

    today = datetime.now()
    cur_hour = today.hour
    today = datetime.strftime(today, '%Y-%m-%d')

    supposed_date_num = calendar[(calendar > target_max_timestamp) & (calendar <= today)].size  # supposing number of dates for new data
    if cur_hour < 15:  # before market close, day num - 1
        supposed_date_num -= 1

    if supposed_date_num != 0:  # need to update data
        # get data from tushare (new)
        is_successful = updateTSNewIncrm(sql_conn_quant, target_max_timestamp, supposed_date_num)  # get data from tushare(new)

        if not is_successful:  # data from tushare (new) incomplete, get data from tushare (old)
            # get data from tushare (old)
            is_successful = updateTSOldIncrm(calendar, sql_conn_spider, target_max_timestamp, supposed_date_num)

            if not is_successful:  # data from tushare (old) incomplete, get data from tencent
                is_successful = updateTCIncrm(sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num)

    sql_conn_quant.close()
    sql_conn_spider.close()


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
