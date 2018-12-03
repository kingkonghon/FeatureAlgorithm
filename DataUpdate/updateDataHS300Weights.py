import h5py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant

def loadHS300WeightsFromFile():
    file_path = r'D:\QuantDesk\Factor\LZ_CN_STKA_INDEX_HS300WEIGHT.h5'

    h5_file = h5py.File(file_path, 'r')
    ds_data = h5_file['data']
    ds_code = h5_file['header']
    ds_tradedates = h5_file['date']

    codes = ds_code[...]
    trade_dates = ds_tradedates[...]
    weights = ds_data[...]

    # convert code and date format
    codes = np.array(list(map(lambda x: x.decode('utf-8').split('.')[1], codes)))
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))

    df_weights = pd.DataFrame(weights, columns=codes, index=trade_dates)
    df_weights.index.name = 'date'
    df_weights = df_weights.reset_index()
    df_weights = pd.melt(df_weights, id_vars='date', var_name='code', value_name='weight')
    df_weights = df_weights.loc[df_weights['weight'] > 0]

    return df_weights

def loadHS300WeightsFromDB(start_date='2007-01-01'):
    # ==== read data from spider database
    source_table_name = 'EastMoneyhs300'
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))
    sql_conn = sql_engine.connect()

    source_fields = ['report_time', 'code', 'weigth', 'time_stamp']
    source_date_field = 'report_time'
    tmp_flds = ['`%s`' % x for x in source_fields]
    tmp_flds = ','.join(tmp_flds)

    sql_statement = "select %s from `%s` where `%s` > '%s'" % (tmp_flds, source_table_name, source_date_field, start_date)
    raw_data = pd.read_sql(sql_statement, sql_conn)

    sql_conn.close()

    # ==== process raw data
    # rename column
    new_fields = ['date', 'code', 'weight']
    tmp_rename_dict = {}
    for tmp_old_field, tmp_new_field in zip(source_fields, new_fields):
        tmp_rename_dict[tmp_old_field] = tmp_new_field
    process_data = raw_data.rename(tmp_rename_dict, axis=1)

    # drop duplicates
    process_data = process_data.sort_values('time_stamp')  # sort by time stamp, and keep the one with latest time stamp when drop duplicates
    process_data = process_data.drop_duplicates(['date', 'code'], keep='last')
    process_data = process_data.drop('time_stamp', axis=1)
    process_data = process_data.sort_values(['date', 'code'])

    # change data type
    process_data.loc[:, 'weight'] = process_data['weight'].astype('float')

    return process_data

def updateFull():
    target_table_name = 'HS300_WEIGHTS'
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    weights_from_file = loadHS300WeightsFromFile()
    weights_from_file.loc[:, 'time_stamp'] = datetime.now()

    sql_conn = sql_engine.connect()
    weights_from_file.to_sql(target_table_name, sql_conn, index=False, if_exists='replace')

    sql_conn.close()

def updateIncrm():
    target_table_name = 'HS300_WEIGHTS'
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    sql_statement = "select max(date) from %s" % target_table_name
    sql_conn = sql_engine.connect()
    target_max_date = pd.read_sql(sql_statement, sql_conn)
    target_max_date = target_max_date.iloc[0, 0]
    sql_conn.close()

    weights_from_spider = loadHS300WeightsFromDB(target_max_date)

    if not weights_from_spider.empty:
        weights_from_spider = weights_from_spider.loc[weights_from_spider['date'] > target_max_date]

        weights_from_spider.loc[:, 'time_stamp'] = datetime.now()

        # write sql
        sql_conn = sql_engine.connect()
        weights_from_spider.to_sql(target_table_name, sql_conn, index=False, if_exists='append')

def airflowCallable():
    updateIncrm()

if __name__ == '__main__':
    # weights_from_file = loadHS300WeightsFromFile()
    #
    # weights_from_spider = loadHS300WeightsFromDB()

    # updateFull()

    updateIncrm()

    pass