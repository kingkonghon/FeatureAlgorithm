# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
# import h5py
import numpy as np
# import tushare as ts
import os
import sys
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'EastMoneyHouFuQuan'
sourceFields = ['report_time', 'code', 'open', 'high', 'low', 'close', 'turnover', 'amount']
sourceTimeStamp = 'report_time'
sourceCode = 'code'

# supplement turnover from unadjusted quote
sourceSupplementTableName = 'EastMoneyBuFuQuan'
sourceSupFields = ['report_time', 'code', 'turnover']
sourceTurnoverField = 'turnover'

# TARGET
targetTableName = 'STOCK_FORWARD_ADJ_QUOTE'
targetFields = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'] # after combining main and supplement data
chgDataTypeCol = ['open', 'high', 'low', 'close', 'volume']
targetTimeStamp = 'date'

targetNewTimeStamp = 'time_stamp'

supplementH5FilePath = r'F:\FeatureAlgorithm\Tools\LZ_CN_STKA_QUOTE_TCLOSE.h5'

def updateFull(quant_engine, spider_engine, chunk_size, start_date='2007-01-01'):
    # get distinct code
    sql_statement = "select distinct `%s` from %s where %s >= '%s'" % (sourceCode, sourceTableName, sourceTimeStamp, start_date)
    tot_codes = pd.read_sql(sql_statement, spider_engine)
    tot_codes = tot_codes.values.T[0]

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    tmp_sup_fields = list(map(lambda x: '`%s`' % x, sourceSupFields))
    tmp_sup_fields = ','.join(tmp_sup_fields)
    write_method = 'replace'
    loop_num = int(tot_codes.size / chunk_size)
    if tot_codes.size > loop_num * chunk_size:
        loop_num += 1
    for i in range(loop_num):
        tmp_code = tot_codes[i*chunk_size:(i+1)*chunk_size]
        tmp_code_str = list(map(lambda x:"'%s'"%x, tmp_code))
        tmp_code_str = ','.join(tmp_code_str)

        sql_statement = "select %s from %s where (`%s` > '%s') and (`%s` != 'null') and (`%s` in (%s))" % (tmp_fields, sourceTableName, sourceTimeStamp,
                        start_date, sourceTimeStamp, sourceCode, tmp_code_str)
        chunk_data = pd.read_sql(sql_statement, spider_engine)

        #  ========== supplement ==============
        sql_statement = "select %s from %s where (`%s` > '%s') and (`%s` != 'null')" % (tmp_sup_fields, sourceSupplementTableName, sourceTimeStamp,
                start_date, sourceTimeStamp)
        chunk_sup_data = pd.read_sql(sql_statement, spider_engine)
        # process raw data
        chunk_sup_data = chunk_sup_data.drop_duplicates([sourceCode, sourceTimeStamp])
        chunk_sup_data.loc[:, sourceTurnoverField] = chunk_sup_data[sourceTurnoverField].apply(
            lambda x: x if x != '-' else 0)
        # merge main data set with supplement data set
        chunk_data = chunk_data.merge(chunk_sup_data, how='left', on=[sourceTimeStamp, sourceCode],
                                      suffixes=['', '_sup'])
        chunk_data.loc[:, sourceTurnoverField] = chunk_data[sourceTurnoverField].fillna(0)
        # change column name set
        sourceFields.append(sourceTurnoverField + '_sup')
        # ===========================

        # rename columns
        rename_dict = {}
        for field in zip(sourceFields, targetFields):
            rename_dict[field[0]] = field[1]
        chunk_data = chunk_data.rename(columns=rename_dict)

        # drop duplicates
        chunk_data = chunk_data.drop_duplicates(['date', 'code'])

        # change data type
        chunk_data.loc[:, 'amount'] = chunk_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (
                float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
        chunk_data.loc[:, 'turnover'] = chunk_data['turnover'].apply(lambda x: float(x) if x != 'null' else np.nan)
        chunk_data = chgDFDataType(chunk_data, chgDataTypeCol, 'float')

        # add time stamp
        chunk_data[targetNewTimeStamp] = datetime.now()

        # write data to db
        chunk_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
        write_method = 'append'

    pass

# use tushare to make up the lost data for EastMoney
# def supplementForAdjQuote(quant_engine):
#     # load h5 file
#     h5_file = h5py.File(supplementH5FilePath)
#     # dataset
#     ds_code = h5_file['header']
#     ds_data = h5_file['data']
#     ds_tradedates = h5_file['date']
#     tot_code = ds_code[...]
#     h5_data = ds_data[...]
#     h5_tradedates = ds_tradedates[...]
#
#     # transform the code format
#     tot_code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), tot_code)))  # remove the market signs
#     tmp_idx = np.sum(~np.isnan(h5_data), axis=0) != 0 # find stock that has data
#     tot_code = tot_code[tmp_idx]
#
#     # find code that EastMoney missed
#     sql_statement = "select distinct `code` from `%s`" % targetTableName
#     exist_code = pd.read_sql(sql_statement, quant_engine)
#     missed_code = set(tot_code) - set(exist_code['code'].values)
#
#     # find date range
#     sql_statement = "select max(`date`) from `%s`" % targetTableName
#     end_date = pd.read_sql(sql_statement, quant_engine).iloc[0,0]
#     sql_statement = "select min(`date`) from `%s`" % targetTableName
#     start_date = pd.read_sql(sql_statement, quant_engine).iloc[0,0]
#
#     # prepare h5 data to combine with tushare data (tushare doesn't have `amount`)
#     h5_tradedates = list(map(lambda x: str(x), h5_tradedates))
#     h5_tradedates = np.array(list(map(lambda x: '-'.join([x[0:4], x[4:6], x[6:]]), h5_tradedates)))
#     tmp_idx = (h5_tradedates >= start_date) & (h5_tradedates <= end_date)
#     h5_data = h5_data[tmp_idx, :]
#
#     # get forward adjust quote from tushare
#     for tmp_code in missed_code:
#         tmp_quote = ts.get_k_data(code=tmp_code, ktype='D', autype='hfq', index=False, start=start_date, end=end_date)
#
#         if not tmp_quote.empty:
#             # combine amount (tushare dosen't provide `amount`)
#
#             # add timestamp
#             tmp_quote[targetNewTimeStamp] = datetime.now()
#
#             # write db
#             tmp_quote.to_sql(targetTableName, quant_engine, index=False, if_exists='append')

def updateIncrm(quant_engine, spider_engine):
    # get lastest tradedate
    sql_statement = "select max(`%s`) from %s where `%s` != 'null'" % (targetTimeStamp, targetTableName, targetTimeStamp)
    quant_conn = quant_engine.connect()
    latest_date = pd.read_sql(sql_statement, quant_conn).iloc[0,0]
    quant_conn.close()

    # get incremental data
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    tmp_sup_fields = list(map(lambda x: '`%s`' % x, sourceSupFields))
    tmp_sup_fields = ','.join(tmp_sup_fields)
    sql_statement = "select %s from %s where (%s > '%s') and (%s != 'null')" % (tmp_fields, sourceTableName, sourceTimeStamp,
                                                                                latest_date, sourceTimeStamp)
    spider_conn = spider_engine.connect()
    incrm_data = pd.read_sql(sql_statement, spider_conn)

    #  ========== supplement ==============
    sql_statement = "select %s from %s where (`%s` > '%s') and (`%s` != 'null')" % (tmp_sup_fields, sourceSupplementTableName, sourceTimeStamp,
            latest_date, sourceTimeStamp)
    incrm_sup_data = pd.read_sql(sql_statement, spider_conn)
    spider_conn.close()
    # process raw data
    incrm_sup_data = incrm_sup_data.drop_duplicates([sourceCode, sourceTimeStamp])
    incrm_sup_data.loc[:, sourceTurnoverField] = incrm_sup_data[sourceTurnoverField].apply(lambda x: x if x != '-' else 0)
    # merge main data set with supplement data set
    incrm_data = incrm_data.merge(incrm_sup_data, how='inner', on=[sourceTimeStamp, sourceCode], suffixes=['', '_sup'])
    incrm_data.loc[:, sourceTurnoverField] = incrm_data[sourceTurnoverField].fillna(0)
    # change column name set
    sourceFields.append(sourceTurnoverField + '_sup')
    # ===========================
    print('data from spider:', incrm_data.shape)

    # rename data
    rename_dict = {}
    for field in zip(sourceFields, targetFields):
        rename_dict[field[0]] = field[1]
    incrm_data = incrm_data.rename(columns=rename_dict)

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    # change data type
    incrm_data.loc[:, 'amount'] = incrm_data['amount'].apply(lambda x: float(x[:-1]) * 100000000 if x[-1] == u'亿' else (
        float(x[:-1]) * 10000 if x[-1] == u'万' else float(x)))
    incrm_data.loc[:, 'turnover'] = incrm_data['turnover'].apply(lambda x: float(x) if x != 'null' else np.nan)
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

    # add time stamp
    incrm_data[targetNewTimeStamp] = datetime.now()

    print('data to write:', incrm_data.shape)

    # write data to db
    if not incrm_data.empty:
        quant_conn = quant_engine.connect()
        incrm_data.to_sql(targetTableName, quant_conn, index=False, if_exists='append')
        quant_conn.close()

def airflowCallable():
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    chunk_size = 10

    # updateFull(quant_engine, spider_engine, chunk_size)
    # supplementForAdjQuote(quant_engine)
    updateIncrm(quant_engine, spider_engine)


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    chunk_size = 10
    t_start = time.clock()

    # updateFull(quant_engine, spider_engine, chunk_size)
    # supplementForAdjQuote(quant_engine)

    updateIncrm(quant_engine, spider_engine)

    print(time.clock() - t_start)