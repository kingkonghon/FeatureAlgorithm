# -*- coding:utf-8 -*-

import pandas as pd
import pandas.io.sql as sql
import numpy as np
from sqlalchemy import create_engine
import pymysql
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s  %(module)s:%(lineno)d] %(message)s',
                    filemode='w')

def readDB(sql_statement, db_config):
    con = pymysql.connect(**db_config)
    data = sql.read_sql(sql_statement, con)
    return data


def writeDB(table_name, dataresult, db_config, method):
    yconnect = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    dataresult = dataresult.replace(np.inf, np.nan)
    dataresult = dataresult.replace(-np.inf, np.nan)

    error = []
    chunk_size = 100000
    chunk_list = list(range(0, dataresult.shape[0], chunk_size))
    if chunk_list[-1] < dataresult.shape[0]:
        chunk_list.append(dataresult.shape[0])
    for i in range(len(chunk_list) - 1):
        try:
            temp = dataresult.iloc[chunk_list[i]:chunk_list[i + 1]]
            temp.to_sql(table_name, yconnect, index=False, if_exists=method, chunksize=int(chunk_size / 10))
        except:
            error.append(i)
            logging.error('>>>tableName:%s, location: %d - %d, dump error' % (table_name, chunk_list[i], chunk_list[i+1]))
            continue

    if error == []:
        return

    # error chunk, change to smaller batches
    chunk_size = 10000
    smaller_chunk_list = []
    error2 = []
    for i in error:
        tmp_chunk = list(range(chunk_list[i], chunk_list[i + 1], chunk_size))
        if tmp_chunk[-1] < chunk_list[i+1]:
            tmp_chunk.append(chunk_list[i+1])
        smaller_chunk_list.extend(tmp_chunk)
    for i in range(len(smaller_chunk_list) - 1):
        try:
            temp = dataresult.iloc[smaller_chunk_list[i]:smaller_chunk_list[i + 1]]
            temp.to_sql(table_name, yconnect, if_exists='append', index=True, chunksize= chunk_size / 10)
        except:
            error2.append(i)
            logging.error(
                '>>>tableName:%s, location: %d - %d, dump error' % (table_name, smaller_chunk_list[i], smaller_chunk_list[i + 1]))

def checkIfIncre(db_config, source_table_name, target_table_name, date_field, lags, source_condition, align_flag=True):
    # check if target table exist
    tmp_state = "select count(1) as num from INFORMATION_SCHEMA.TABLES " \
                "where TABLE_SCHEMA='%s' and TABLE_NAME = '%s'" % (db_config['db'], target_table_name)
    tmp_data = readDB(tmp_state, db_config)

    if source_condition != '':
        source_condition_sql = "where " + source_condition
    else:
        source_condition_sql = ''
    last_record_date = np.nan
    start_fetch_date = np.nan
    if tmp_data.loc[0, 'num'] == 0:  # target table not exist
        is_full = 1
    else:  # table already exist
        # compare last update time (***** 待修改，增加一个timestamp来判断)
        tmp_state = "SELECT max(%s) AS timestamp_target, count(1) AS num_target, " \
                    "(SELECT max(%s) FROM `%s` %s) AS timestamp_source, " \
                    "(SELECT count(1) FROM `%s` %s) AS num_source " \
                    "FROM `%s`" % (date_field, date_field, source_table_name, source_condition_sql, source_table_name, source_condition_sql, target_table_name)
        tmp_data = readDB(tmp_state, db_config)

        # last record date on target table
        # tmp_data.loc[0, 'timestamp_target'] = '2017-12-16' # test increment
        last_record_date = tmp_data.loc[0, 'timestamp_target']

        # 开始取数的日期，考虑计算增量所需要的lag天数
        lags_needed = max([max(lags), 252])  # 计算增量所需要的数据条数（包括用于算MA等指标，MACD, BBRANDS； 考虑到EMA的准确性，用一个比较大的lag）
        # 计算开始读数据的时间
        try:
            start_fetch_date = datetime.strptime(last_record_date, '%Y-%m-%d') - timedelta(
                days=lags_needed)
        except TypeError:
            start_fetch_date = last_record_date - timedelta(days=lags_needed)
        start_fetch_date = datetime.strftime(start_fetch_date, '%Y-%m-%d')

        if tmp_data.loc[0, 'timestamp_target'] < tmp_data.loc[0, 'timestamp_source']: # target last record date ealier than source, run incre
            is_full = 0
        elif (tmp_data.loc[0, 'timestamp_target'] == tmp_data.loc[0, 'timestamp_source']) and \
                align_flag and (tmp_data.loc[0, 'num_target'] != tmp_data.loc[0, 'num_source']):  # 最新日期一样，但数据条数不一样，重跑全量
            is_full = 1
        else:
            is_full = -1 # 不需要跑

    return is_full, last_record_date, start_fetch_date


def getTradeDates(sql_engine, date_field, table_name, start_date):
    sql_statement = "SELECT DISTINCT `%s` FROM `%s`;" % (date_field, table_name)
    trade_dates = pd.read_sql(sql_statement, sql_engine).values
    trade_dates = trade_dates.T[0]
    trade_dates = trade_dates[trade_dates >= start_date]
    trade_dates = np.sort(trade_dates)

    return trade_dates

def getDataFromSQL(sql_engine, date_field, code_field, quote_field, table_name, date_range_field):
    sql_statement = 'select `%s`, `%s`, %s from %s where date in (%s)' % (
        date_field, code_field, quote_field, table_name, date_range_field)
    basic_data = pd.read_sql(sql_statement, sql_engine)
    basic_data = basic_data.rename(columns={code_field: 'code', date_field: 'date'})
    basic_data = basic_data.drop_duplicates(['date', 'code'])
    return basic_data

def getIncrmDataFromSQL(sql_engine, date_field, code_field, quote_field, source_table_name, target_table_name,
                        source_time_stamp_field, target_time_stamp_field):
    sql_statement = 'select max(`%s`) from %s' % (target_time_stamp_field, target_table_name)
    target_time_stamp = pd.read_sql(sql_statement, sql_engine)
    if target_time_stamp.size != 0:
        target_time_stamp = target_time_stamp.iloc[0,0]

    sql_statement = "select `%s`, `%s`, %s from %s where `%s` > '%s'" % (
        date_field, code_field, quote_field, source_table_name, source_time_stamp_field, target_time_stamp)
    basic_data = pd.read_sql(sql_statement, sql_engine)

    basic_data = basic_data.rename(columns={code_field: 'code', date_field: 'date'})
    basic_data = basic_data.drop_duplicates(['date', 'code'])
    return basic_data