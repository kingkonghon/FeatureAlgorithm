from datetime import datetime
from Utils.Algorithms import calWeightedSumIndexQuote
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import writeDB, getTradeDates, getDataFromSQL, checkIfIncre, getIncrmDataFromSQL

ConArea = {
    'tableName':'STOCK_DESCRIPTION',
    'code': 'code',
    'area': 'area'
}
ConQuote ={
    'tableName': 'STOCK_FORWARD_ADJ_QUOTE',
    'code': 'code',
    'date': 'date',
    'fields': ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'],
    'time_stamp': 'time_stamp'
}

ConWeight = {
    'tableName': 'STOCK_FUNDAMENTAL_BASIC',
    'code': 'code',
    'date': 'date',
    'weight': 'FREE_MRK_CAP',
    'time_stamp': 'time_stamp'
}


targetTableName = 'AREA_INDEX_FORWARD_ADJ_QUOTE'
targetDateField = 'date'
targetNewTimeStampField = 'time_stamp'

def calFullArea(db_config, con_area, con_quote, con_weights, chunk_size, start_date='2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    # get area    ************ future data, need to find better solution
    sql_statement = "SELECT `{code}`, `{area}`  FROM `{tableName}`;".format(**con_area)
    stock_area = pd.read_sql(sql_statement, my_engine)
    stock_area = stock_area.rename(columns={con_area['code']: 'code', con_area['area']: 'area'})

    # get total trade dates
    trade_dates = getTradeDates(my_engine, con_quote['date'], con_quote['tableName'], start_date)

    # read and process data by trade dates
    quote_fields = list(map(lambda x:'`%s`'%x, con_quote['fields']))
    quote_fields = ','.join(quote_fields)
    write_sql_method = 'replace'
    for i in range(int(trade_dates.size / chunk_size)+1):
        tmp_trade_dates = trade_dates[i*chunk_size:(i+1)*chunk_size]
        tmp_trade_dates = list(map(lambda x:"'%s'" % x, tmp_trade_dates))
        date_range = ','.join(tmp_trade_dates)

        # get quote data
        basic_data = getDataFromSQL(my_engine,con_quote['date'], con_quote['code'], quote_fields, con_quote['tableName'], date_range)

        # get weights
        weighted_field = '`%s`' % con_weights['weight']
        weights = getDataFromSQL(my_engine, con_weights['date'], con_weights['code'], weighted_field, con_weights['tableName'], date_range)
        weights = weights.rename(columns={con_weights['weight']: 'weight'})

        basic_data = basic_data.merge(weights, on=['date','code'], how='inner')
        basic_data = basic_data.merge(stock_area, on='code', how='inner')

        # calculate weighted sum index quote
        area_index_quote = calWeightedSumIndexQuote(basic_data, con_quote['fields'], 'date', 'area', 'weight')

        # add timestamp
        area_index_quote[targetNewTimeStampField] = datetime.now()

        writeDB(targetTableName, area_index_quote, db_config, write_sql_method)
        write_sql_method = 'append'

def calIncrmArea(db_config, con_area, con_quote, con_weights, chunk_size, start_date='2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    write_sql_method = 'append'

    # get quote data (trim by timestamp)
    quote_fields = list(map(lambda x: '`%s`' % x, con_quote['fields']))
    quote_fields = ','.join(quote_fields)
    basic_data = getIncrmDataFromSQL(my_engine, con_quote['date'], con_quote['code'], quote_fields,
                                     con_quote['tableName'], targetTableName, con_quote['date'],
                                     targetDateField)

    # already the latest data
    if basic_data.empty:
        return

    # get weights (trim by timestamp)
    weight_field = '`%s`' % con_weights['weight']
    weights = getIncrmDataFromSQL(my_engine, con_weights['date'], con_weights['code'], weight_field,
                                  con_weights['tableName'], targetTableName, con_weights['date'],
                                  targetDateField)
    weights = weights.rename(columns={con_weights['weight']: 'weight'})

    # get area    ************ future data, need to find better solution
    sql_statement = "SELECT `{code}`, `{area}`  FROM `{tableName}`;".format(**con_area)
    stock_area = pd.read_sql(sql_statement, my_engine)
    stock_area = stock_area.rename(columns={con_area['code']: 'code', con_area['area']: 'area'})

    basic_data = basic_data.merge(weights, on=['date', 'code'], how='inner')
    basic_data = basic_data.merge(stock_area, on='code', how='inner')

    area_index_quote = calWeightedSumIndexQuote(basic_data, con_quote['fields'], 'date', 'area', 'weight')

    area_index_quote[targetNewTimeStampField] = datetime.now()

    if not area_index_quote.empty:
        writeDB(targetTableName, area_index_quote, db_config, write_sql_method)

def airflowCallable():
    start_date = '2007-01-01'
    chunk_size = 10

    calIncrmArea(ConfigQuant, ConArea, ConQuote, ConWeight, chunk_size, start_date)


if __name__ == '__main__':
    start_date = '2007-01-01'

    is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, ConQuote['tableName'],
                                                               targetTableName, ConQuote['date'], [0], '', False)

    if is_full == 1:
        calFullArea(ConfigQuant, ConArea, ConQuote, ConWeight, 10, start_date)
    elif is_full == 0:
        calIncrmArea(ConfigQuant, ConArea, ConQuote, ConWeight, 10, start_date)
    else:
        pass