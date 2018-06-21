from Utils.DB_config import ConfigQuant
from Utils.DBOperation import writeDB, getIncrmDataFromSQL, checkIfIncre
from datetime import  datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

sourceTableName = 'yucezhe_market_overviewdata'
targetTableName = 'STOCK_FORWARD_ADJ_QUOTE'
dateField = 'date'
codeField = 'code'
timeStampField = 'time_stamp'
fieldNames = ['open', 'high', 'low', 'close', 'after_close', 'volume', 'amount', 'turnover']

def calFull(db_config, start_date='2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    # get source table size
    sql_statement = "SELECT DISTINCT(date) FROM `%s`;" % sourceTableName
    trade_dates = pd.read_sql(sql_statement, my_engine).values
    trade_dates = trade_dates.T[0]
    trade_dates = trade_dates[trade_dates > start_date]
    trade_dates = np.sort(trade_dates)

    # read and process data by trade dates
    tmp_fields = [dateField, codeField]
    tmp_fields.extend(fieldNames)
    tmp_fields = ','.join(tmp_fields)
    write_sql_method = 'replace'
    for i in range(int(trade_dates.size / 5)+1):
        tmp_trade_dates = trade_dates[i*5:(i+1)*5]
        tmp_trade_dates = list(map(lambda x:"'%s'" % x, tmp_trade_dates))
        tmp_range = ','.join(tmp_trade_dates)
        sql_statement = 'select %s from %s where date in (%s)' % (tmp_fields, sourceTableName, tmp_range)

        basic_data = pd.read_sql(sql_statement, my_engine)
        basic_data = basic_data.drop_duplicates(['date', 'code'])

        new_data = basic_data[['code', 'date']]
        new_data['adj_factor'] = basic_data['after_close'] / basic_data['close']
        for field in ['open', 'high', 'low', 'close']:
            new_data[field] = basic_data[field] * new_data['adj_factor']

        new_data = new_data.join(basic_data[['volume', 'amount', 'turnover']])

        new_data[timeStampField] = datetime.now()

        writeDB(targetTableName, new_data, db_config, write_sql_method)
        write_sql_method = 'append'

def calIncrm(db_config, start_date='2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    write_sql_method = 'append'

    quote_fields = ','.join(fieldNames)

    basic_data = getIncrmDataFromSQL(my_engine, dateField, codeField, quote_fields,
                                     sourceTableName, targetTableName, timeStampField,
                                     timeStampField)

    if basic_data.empty:
        return

    basic_data = basic_data.drop_duplicates(['date', 'code'])

    new_data = basic_data[['code', 'date']]
    new_data['adj_factor'] = basic_data['after_close'] / basic_data['close']
    for field in ['open', 'high', 'low', 'close']:
        new_data[field] = basic_data[field] * new_data['adj_factor']

    new_data = new_data.join(basic_data[['volume', 'amount', 'turnover']])

    new_data[timeStampField] = datetime.now()

    writeDB(targetTableName, new_data, db_config, write_sql_method)


if __name__ == '__main__':
    is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, sourceTableName,
                                                               targetTableName, dateField, [0], '', False)
    if is_full == 1:
        calFull(ConfigQuant)
    elif is_full == 0:
        calIncrm(ConfigQuant)
    else:
        pass