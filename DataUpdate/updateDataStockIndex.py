# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant, ConfigSpider2
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'EastMoneyIndex'
sourceFields = ['report_time', 'code', 'open', 'high', 'low', 'close', 'turnover', 'amount']
sourceTimeStamp = 'report_time'

# TARGET
targetTableNameMarketIndex = 'STOCK_MARKET_INDEX_QUOTE'
targetTableNameHS = 'HS300_QUOTE'
targetFields = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
marketToCode = {'SH': '000001', 'SZ':'399001', 'SMEB':'399005', 'GEB': '399006'}
codeHS300 = '000300'
chgDataTypeCol = ['open', 'high', 'low', 'close', 'volume']
targetTimeStamp = 'date'

targetNewTimeStamp = 'time_stamp'

def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s`" % (tmp_fields, sourceTableName)
    full_data = pd.read_sql(sql_statement, spider_engine)  # spider schema

    if full_data.empty:
        return

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # drop duplicates (if any)
    full_data = full_data.drop_duplicates(['date', 'code'])

    # change data type
    full_data.loc[:, 'amount'] = full_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
    full_data = chgDFDataType(full_data, chgDataTypeCol, 'float')

    market_index_data = full_data.loc[full_data['code'].isin(list(marketToCode.values()))]
    hs300_data = full_data.loc[full_data['code'] == '000300']

    # convert code to market
    market_index_data['market'] = ''
    for (tmp_market, tmp_code) in marketToCode.items():
        market_index_data.loc[market_index_data['code'] == tmp_code, 'market'] = tmp_market

    market_index_data = market_index_data.drop('code', axis=1)
    hs300_data = hs300_data.drop('code', axis=1)

    # add time stamp
    market_index_data[targetNewTimeStamp] = datetime.now()
    hs300_data[targetNewTimeStamp] = datetime.now()

    # write data to target
    if not market_index_data.empty:
        market_index_data.to_sql(targetTableNameMarketIndex, quant_engine, index=False, if_exists='replace')
    if not hs300_data.empty:
        hs300_data.to_sql(targetTableNameHS, quant_engine, index=False, if_exists='replace')

    pass

def updateIncrm(quant_engine, spider_engine):
    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableNameMarketIndex)
    target_max_timestamp = pd.read_sql(sql_statement, quant_engine) # quant schema
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableNameHS)
    target_max_timestamp_hs = pd.read_sql(sql_statement, quant_engine)  # quant schema
    target_max_timestamp_hs = target_max_timestamp_hs.iloc[0, 0]

    sql_timestamp = min(target_max_timestamp, target_max_timestamp_hs)

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, sql_timestamp)
    incrm_data = pd.read_sql(sql_statement, spider_engine) # spider schema

    if incrm_data.empty:
        return

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    # change data type
    incrm_data.loc[:, 'amount'] = incrm_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (
        float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

    market_index_data = incrm_data.loc[incrm_data['code'].isin(list(marketToCode.values()))]
    hs300_data = incrm_data.loc[incrm_data['code'] == '000300']

    # trim by date
    market_index_data = market_index_data.loc[market_index_data['date'] > target_max_timestamp]
    hs300_data = hs300_data.loc[hs300_data['date'] > target_max_timestamp_hs]

    # convert code to market
    market_index_data['market'] = ''
    for (tmp_market, tmp_code) in marketToCode.items():
        market_index_data.loc[market_index_data['code'] == tmp_code, 'market'] = tmp_market

    # drop column
    market_index_data = market_index_data.drop('code', axis=1)
    hs300_data = hs300_data.drop('code', axis=1)

    # add time stamp & write data to target
    if not incrm_data.empty:
        market_index_data[targetNewTimeStamp] = datetime.now()
        market_index_data.to_sql(targetTableNameMarketIndex, quant_engine, index=False, if_exists='append')
    if not hs300_data.empty:
        hs300_data[targetNewTimeStamp] = datetime.now()
        hs300_data.to_sql(targetTableNameHS, quant_engine, index=False, if_exists='append')

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
