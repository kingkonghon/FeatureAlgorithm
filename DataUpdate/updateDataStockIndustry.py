# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceTableName = 'sws_research'
sourceFields = ['time_stamp', 'stock', 'industry_name']
sourceTimeStamp = 'time_stamp'

# TARGET
targetTableName = 'STOCK_INDUSTRY'
targetFields = ['date', 'code', 'industry']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'

# CALENDAR (drop duplicates by trade dates)
calendarTableName = 'TRADE_CALENDAR'
calendarDateField = 'date'

# ****** 需要用 TRADE CALENDAR去重；2018-04-25 这个节点数据前后数据会有差异，之前用的是万德的数据，之后是申万官网的数据


def updateFull(quant_engine, spider_engine):
    pass

def updateIncrm(quant_engine, spider_engine):
    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, quant_engine) # quant schema
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    # transform to source time stamp format
    timestamp_str = '%s年%s月%s日' % (target_max_timestamp[:4], target_max_timestamp[5:7], target_max_timestamp[8:10])

    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where `%s` > '%s'" % (
        tmp_fields, sourceTableName, sourceTimeStamp, timestamp_str)
    incrm_data = pd.read_sql(sql_statement, spider_engine) # spider schema

    if incrm_data.empty:
        return

    # rename columns
    incrm_data = renameDF(incrm_data, sourceFields, targetFields)

    # transform time_stamp to date (后一天1点钟更新, 算前一天的数据)
    # tmp_datetime = incrm_data['date'].apply(
    #     lambda x: datetime.strptime(x, u'%Y年%m月%d日 %H:%M:%S') - timedelta(days=1))
    # incrm_data.loc[:, 'date'] = tmp_datetime.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    incrm_data.loc[:, 'date'] = incrm_data['date'].apply(lambda x: '-'.join([x[:4], x[5:7], x[8:10]]))

    incrm_data = incrm_data.loc[incrm_data['date'] > target_max_timestamp]
    if incrm_data.empty:
        return

    # 爬虫库周末也有运行，会爬到重复的数据，所以需要用trade calendar 筛掉无用数据 （周日晚也会更新，那就是周五晚更新的数据可以舍弃）
    sql_statement = 'select `%s` from `%s`' % (calendarDateField, calendarTableName)
    trade_calendar = pd.read_sql(sql_statement, quant_engine).values.T[0] # quant schema
    # trade_calendar = trade_calendar.rename(columns={calendarDateField: 'date'})
    incrm_data = incrm_data.loc[incrm_data['date'].isin(trade_calendar)]

    # # move data 1 trade date earlier
    # tmp_idx = np.where(np.in1d(trade_calendar, incrm_data['date'].values)) - 1
    # actual_date = trade_calendar[tmp_idx]
    # incrm_data.loc[:, 'date'] = actual_date

    # drop duplicates
    incrm_data = incrm_data.drop_duplicates(['date', 'code'])

    # sort by date
    incrm_data = incrm_data.sort_values('date')

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
