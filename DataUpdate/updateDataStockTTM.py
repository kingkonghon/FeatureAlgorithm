
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
sourceTableName = 'XueQiuStockTTM'
sourceFields = ['tradedate', 'stockCode', 'pettm', 'pcttm']
sourceTimeStamp = 'tradedate'

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



def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, quant_engine)
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    target_max_timestamp_format = target_max_timestamp.replace('-', '') # 2015-01-01 --> 20150101

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    # there maybe some record update later (so use ">="  instead of ">"), fetch data including the latest day in target, drop duplicates later
    sql_statement = "select %s from `%s` where `%s` >= '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp_format)
    incrm_data = pd.read_sql(sql_statement, spider_engine)

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
    existing_data = pd.read_sql(sql_statement, quant_engine)

    # combine existing and increment, and drop duplicates --> remain the real increment and missing data
    incrm_data = existing_data.append(incrm_data)
    incrm_data = incrm_data.drop_duplicates(['date', 'code'], keep=False)

    if not incrm_data.empty:
        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # # calculate from raw data to fill the missings in spider data
        # sup_data = supplementByRawData(quant_engine, target_max_timestamp, incrm_data)

        # sort by date
        incrm_data = incrm_data.sort_values('date')

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
