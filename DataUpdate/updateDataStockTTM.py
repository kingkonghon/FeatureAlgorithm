
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

calendarTableName = 'TRADE_CALENDAR'

# SOURCE
sourceTableName = 'XueQiuStockTTM'
sourceFields = ['tradedate', 'stockCode', 'pettm', 'pcttm']
sourceTimeStamp = 'tradedate'

# SOURCE tushare
sourceTSTableName = 'STOCK_BASIC_TUSHARE'
sourceTSFields = ['date', 'code', 'pe_ttm', 'ps_ttm']
sourceTSTimeStamp = 'date'


# TARGET
targetTableName = 'STOCK_FUNDAMENTAL_TTM'
targetFields = ['date', 'code', 'PE_TTM', 'PS_TTM']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = ['PE_TTM', 'PS_TTM']

# SUPPLEMENT TABLE
fundamentalTushareTableName = 'STOCK_FUNDAMENTAL_DAILY_TUSHARE'
fundamentalBasicTableName = 'STOCK_FUNDAMENTAL_BASIC'
stockQuoteTableName = 'STOCK_UNADJUSTED_QUOTE'
codeField = 'code'
dateField = 'date'
netProfitTTMField = 'parent_net_profits_ttm'
salesTTMField = 'sales_per_share_ttm'
totalCapFields = 'TOT_MRK_CAP'
stockQuoteField = 'close'

def supplementByRawData(quant_engine, target_max_timestamp, data):
    # date & code comebine as unique identity
    ori_data = data.copy()
    identity_field = 'identity'
    ori_data.loc[:, identity_field] = ori_data.apply(lambda x: '%s %s' % (x[dateField], x[codeField]), axis=1)

    # get fundamental ttm
    tmp_fields = [dateField, codeField, netProfitTTMField, salesTTMField, 'fiscal_year', 'fiscal_season']
    tmp_fields = list(map(lambda x: '`%s`' % x, tmp_fields))
    tmp_fields = ','.join(tmp_fields)
    tmp_state = "select %s from %s where %s >= '%s'" % (tmp_fields, fundamentalTushareTableName, dateField, target_max_timestamp)
    stock_data = pd.read_sql(tmp_state, quant_engine)

    # get fundamental basic
    tmp_fields = [dateField, codeField, totalCapFields]
    tmp_fields = list(map(lambda x: '`%s`' % x, tmp_fields))
    tmp_fields = ','.join(tmp_fields)
    tmp_state = "select %s from %s where %s >= '%s'" % (tmp_fields, fundamentalBasicTableName, dateField, target_max_timestamp)
    tmp_data = pd.read_sql(tmp_state, quant_engine)
    stock_data = stock_data.merge(tmp_data, on=[dateField, codeField], how='outer')

    # get stock quote
    tmp_fields = [dateField, codeField, stockQuoteField]
    tmp_fields = list(map(lambda x: '`%s`' % x, tmp_fields))
    tmp_fields = ','.join(tmp_fields)
    tmp_state = "select %s from %s where %s >= '%s'" % (
        tmp_fields, stockQuoteTableName, dateField, target_max_timestamp)
    tmp_data = pd.read_sql(tmp_state, quant_engine)
    stock_data = stock_data.merge(tmp_data, on=[dateField, codeField], how='outer')

    # raw data's identity
    stock_data.loc[:, identity_field] = stock_data.apply(lambda x: '%s %s' % (x[dateField], x[codeField]), axis=1)

    # calculate PE TTM
    stock_data.loc[:, 'PE_TTM'] = stock_data[totalCapFields] / stock_data[netProfitTTMField]
    pe_ttm = stock_data.loc[~stock_data['PE_TTM'].isnull(), [dateField, codeField, 'PE_TTM', identity_field, 'fiscal_year', 'fiscal_season']]

    tmp_ori_identity = ori_data.loc[~ori_data['PE_TTM'].isnull(), identity_field]  # find original data with PE_TTM
    pe_ttm = pe_ttm.loc[~pe_ttm[identity_field].isin(tmp_ori_identity)] # find PE_TTM calculated from raw data not in original data

    # calculate PS_TTM
    stock_data.loc[:, 'PS_TTM'] = stock_data[stockQuoteField] / stock_data[salesTTMField]
    ps_ttm = stock_data.loc[~stock_data['PS_TTM'].isnull(), [dateField, codeField, 'PS_TTM', identity_field]]

    tmp_ori_identity = ori_data.loc[~ori_data['PS_TTM'].isnull(), identity_field]  # find original data with PE_TTM
    ps_ttm = ps_ttm.loc[
        ~ps_ttm[identity_field].isin(tmp_ori_identity)]  # find PE_TTM calculated from raw data not in original data

    supplement_data = pe_ttm.merge(ps_ttm, on=[dateField, codeField], how='outer')

    return supplement_data

def writeDB(sql_conn, table_name, data):
    # check if there are already records in the table
    new_dates = data['date'].unique()

    new_start_date = new_dates.min()
    new_end_date = new_dates.max()

    sql_statement = "select count(1) as num from `%s` where `date` between '%s' and '%s'" % (table_name, new_start_date, new_end_date)
    old_record_num = pd.read_sql(sql_statement, sql_conn)
    old_record_num = old_record_num.iloc[0, 0]

    if old_record_num > 0:
        print('delete old data from %s to %s' % (new_end_date, new_end_date))
        sql_statement = "delete from `%s` where `date` between '%s' and '%s'" % (table_name, new_start_date, new_end_date)
        sql_conn.execute(sql_statement)

    # write new data to db
    data.to_sql(table_name, sql_conn, index=False, if_exists='append')


def updateTSIncrm(sql_conn_quant, target_max_timestamp, supposed_date_num):
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceTSFields))
    tmp_fields = ','.join(tmp_fields)

    sql_statement = "select %s from `%s` where `%s` > '%s'" % (tmp_fields, sourceTSTableName, sourceTSTimeStamp, target_max_timestamp)
    incrm_data = pd.read_sql(sql_statement, sql_conn_quant)

    incrm_data = incrm_data.drop_duplicates(['date', 'code'])  # drop duplicates
    incrm_data = incrm_data.sort_values('date')  # sort by date

    incrm_data_date_num = incrm_data['date'].unique().size

    if incrm_data_date_num != supposed_date_num:
        return False  # signal of missing data from this source
    else:
        incrm_data = renameDF(incrm_data, sourceTSFields, targetFields)  # change column names

        incrm_data.loc[:, 'time_stamp'] = datetime.now()  # add time stamp

        writeDB(sql_conn_quant, targetTableName, incrm_data)

        return True  # signal of successfully written data to database

def updateXueQiuIncrm(sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num):
    target_max_timestamp_format = target_max_timestamp.replace('-', '')  # 2015-01-01 --> 20150101

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    # there maybe some record update later (so use ">="  instead of ">"), fetch data including the latest day in target, drop duplicates later
    sql_statement = "select %s from `%s` where `%s` >= '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp_format)
    incrm_data = pd.read_sql(sql_statement, sql_conn_spider)

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # change date format
    incrm_data.loc[:, 'date'] = incrm_data['date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:8]]))
    incrm_data = incrm_data.loc[incrm_data['date'] >= target_max_timestamp]

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    # fetch latest data in target table
    tmp_fields = list(map(lambda x: '`%s`' % x, targetFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where `%s` >= '%s'" % (
        tmp_fields, targetTableName, targetTimeStamp, target_max_timestamp)
    existing_data = pd.read_sql(sql_statement, sql_conn_quant)

    # combine existing and increment, and drop duplicates --> remain the real increment and missing data
    incrm_data = existing_data.append(incrm_data)
    incrm_data = incrm_data.drop_duplicates(['date', 'code'], keep=False)

    # check if there are missing data from this source
    incrm_data_date_num = incrm_data['date'].unique().size
    if incrm_data_date_num < supposed_date_num:
        return False   # signal of missing data from this source
    else:
        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # # calculate from raw data to fill the missings in spider data
        # sup_data = supplementByRawData(quant_engine, target_max_timestamp, incrm_data)

        # sort by date
        incrm_data = incrm_data.sort_values('date')

        # add time stamp
        incrm_data[targetNewTimeStamp] = datetime.now()

        writeDB(sql_conn_quant, targetTableName, incrm_data)

        return True   # signal of successfully written data to database


def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    sql_conn_quant = quant_engine.connect()
    sql_conn_spider = spider_engine.connect()

    today = datetime.now()
    cur_hour = today.hour
    today = datetime.strftime(today, '%Y-%m-%d')

    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, sql_conn_quant)
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    target_max_timestamp_format = target_max_timestamp.replace('-', '') # 2015-01-01 --> 20150101

    # get supposing dates of data
    sql_statement = "select `date` from `%s`" % calendarTableName
    trade_calendar = pd.read_sql(sql_statement, sql_conn_quant)
    trade_calendar = trade_calendar['date'].values
    supposed_date_num = trade_calendar[(trade_calendar > target_max_timestamp) & (trade_calendar <= today)].size

    if cur_hour < 15:  # before market close, day num - 1
        supposed_date_num -= 1

    if supposed_date_num > 0:  # need to update data
        is_successful = updateTSIncrm(sql_conn_quant, target_max_timestamp, supposed_date_num)

        # if not is_successful:  # update from primary source is not successful, try the backup source  (XiuQiu is not accurate)
        #     updateXueQiuIncrm(sql_conn_quant, sql_conn_spider, target_max_timestamp_format, supposed_date_num)

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
