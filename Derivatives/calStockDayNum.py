import os
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

sourceTableNameBasic = 'STOCK_DESCRIPTION'
sourceTableNameTradeList = 'TRADE_CALENDAR'
sourceTableNameQuote = 'STOCK_FORWARD_ADJ_QUOTE'

targetTableName = 'STOCK_DAY_COUNT'
fieldNameListDayNum = 'LIST_DAYNUM'
fieldNameResumeDayNum = 'RESUME_DAYNUM'

def calListDayNum(stock_list_date, date_list):
    data_result = pd.DataFrame([])
    for i in range(stock_list_date.shape[0]):
        tmp_stock_info = stock_list_date.iloc[i]
        tmp_date_list = date_list[date_list >= tmp_stock_info['list_date']]
        tmp_size = tmp_date_list.size
        tmp_result = pd.DataFrame({'code': np.repeat(tmp_stock_info['code'], tmp_size), 'date': tmp_date_list,
                                   'list_date': np.repeat(tmp_stock_info['list_date'], tmp_size)})

        tmp_diff = pd.to_datetime(tmp_result['date'], format='%Y-%m-%d') - pd.to_datetime(tmp_result['list_date'], format='%Y-%m-%d')
        tmp_result.loc[:, fieldNameListDayNum] = tmp_diff.apply(lambda x:x.days)
        tmp_result = tmp_result.drop('list_date', axis=1)

        data_result = data_result.append(tmp_result)

    return data_result


def calResumeDayNum(quote_record, date_list):
    codes = quote_record['code'].unique()
    dataresult = pd.DataFrame()
    for code in codes:
        tmp_data = quote_record.loc[quote_record['code'] == code]
        tmp_count = pd.DataFrame({'date':date_list[date_list >= tmp_data.iloc[0]['date']]}) # from the first trade date
        # find no trade record sign
        tmp_count.loc[:, 'sign'] = tmp_count['date'].isin(tmp_data['date'].values)
        tmp_count = tmp_count.reset_index(drop=True)
        tmp_count.loc[:, 'count'] = 0
        for i in range(1, tmp_count.shape[0]):
            if tmp_count.loc[i, 'sign']:
                if ~tmp_count.loc[i-1, 'sign']:
                    tmp_count.loc[i, 'count'] = 1

                elif tmp_count.loc[i-1, 'count'] > 0:
                    tmp_count.loc[i, 'count'] = tmp_count.loc[i-1, 'count'] + 1
            else:
                tmp_count.loc[i, 'count'] = 0
        tmp_count.loc[tmp_count['count'] > 30, 'count'] = 0 # after 30 days, stop counting

        tmp_count = tmp_count.drop('sign', axis=1)
        tmp_count = tmp_count.rename(columns={'count': fieldNameResumeDayNum})
        tmp_data = tmp_data.merge(tmp_count, on='date')

        dataresult = dataresult.append(tmp_data)
    return dataresult

def calFull(quant_engine, start_date='2007-01-01', chunk_size=50000):
    # stock list date
    sql_statement = "select `code`, list_date from %s" % sourceTableNameBasic
    stock_list_date = pd.read_sql(sql_statement, quant_engine)

    tmp_df = stock_list_date.duplicated('code')

    # check duplicates
    if tmp_df.sum() > 0:
        print("duplicated stock description")
        raise ValueError

    # get trade record
    sql_statement = "select date, `code` from %s" % sourceTableNameQuote
    quote_record = pd.read_sql(sql_statement, quant_engine)

    last_date = quote_record['date'].max()

    # trim trade calendar by nowdays
    sql_statement = "select * from %s" % sourceTableNameTradeList
    trade_calendar = pd.read_sql(sql_statement, quant_engine).values.T[0]

    date_list = trade_calendar[(trade_calendar <= last_date) & (trade_calendar >= start_date)]

    # get list day num
    list_date_num = calListDayNum(stock_list_date, date_list)
    data_result = quote_record.merge(list_date_num, how='left', on=['date','code'])
    del list_date_num   # free some space

    # get resume day num (maxium 30)
    resume_daynum = calResumeDayNum(quote_record, date_list)

    data_result = data_result.merge(resume_daynum, how='left', on=['date','code'])

    # sort by date
    data_result = data_result.sort_values('date')

    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # dump data
    write_method = 'replace'
    for i in range(int(data_result.size / chunk_size) + 1):
        tmp_data = data_result.iloc[i*chunk_size : (i+1)*chunk_size]
        tmp_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
        write_method = 'append'
    pass


def calIncrm(quant_engine, start_date='2007-01-01'):
    # stock list date
    sql_statement = "select `code`, list_date from %s" % sourceTableNameBasic
    stock_list_date = pd.read_sql(sql_statement, quant_engine)

    tmp_df = stock_list_date.duplicated('code')

    # check duplicates
    if tmp_df.sum() > 0:
        print("duplicated stock description")
        raise ValueError

    # trim trade calendar by nowdays
    sql_statement = "select * from %s" % sourceTableNameTradeList
    trade_calendar = pd.read_sql(sql_statement, quant_engine).values.T[0]

    # latest update date
    sql_statement = "select max(date) from %s" % targetTableName
    target_max_date = pd.read_sql(sql_statement, quant_engine)
    if not target_max_date.empty:
        target_max_date = target_max_date.iloc[0, 0]
    else:
        target_max_date = start_date

    # get quote record
    sql_statement = "select date, `code` from %s where date > '%s'" % (sourceTableNameQuote, target_max_date)
    incrm_quote_record = pd.read_sql(sql_statement, quant_engine)

    if incrm_quote_record.empty:
        return

    last_date = incrm_quote_record['date'].max()

    # incremental list day num is calculated the same as full
    incrm_date_list = trade_calendar[(trade_calendar <= last_date) & (trade_calendar > target_max_date)]

    # get list day num
    list_date_num = calListDayNum(stock_list_date, incrm_date_list)
    data_result = incrm_quote_record.merge(list_date_num, on=['date','code'], how='left')

    # get resume day num (maxium 30)
    # resume_date = calResumeDayNum(quote_record, date_list)

    # last resume day record
    sql_statement = "select `code`, %s as last_daynum from %s where date = '%s'" % (fieldNameResumeDayNum, targetTableName, target_max_date)
    last_resume_daynum = pd.read_sql(sql_statement, quant_engine)

    # last quote record
    sql_statement = "select `code` from %s where date = '%s'" % (sourceTableNameQuote, target_max_date)
    last_quote_record = pd.read_sql(sql_statement, quant_engine)

    # set sign=1 for stocks with trade records
    incrm_quote_record.loc[:, 'sign'] = 1
    last_quote_record.loc[:, 'last_sign'] = 1

    # construct DF with all listed stock
    resume_daynum = pd.DataFrame()
    for tmp_date in incrm_date_list:
        tmp_result = stock_list_date.copy()
        tmp_result = tmp_result.loc[tmp_result['list_date'] <= tmp_date]  # stocks listed before that day

        # get today's quotes
        current_quote_record = incrm_quote_record.loc[incrm_quote_record['date'] == tmp_date, ['code', 'sign']]

        # get today's list day num
        current_list_daynum = list_date_num.loc[list_date_num['date'] == tmp_date, ['code', fieldNameListDayNum]]

        # combine new trade record, last daynum, last trade record
        tmp_result = tmp_result.merge(current_quote_record, on='code', how='left')
        tmp_result = tmp_result.merge(last_resume_daynum, on='code', how='left')
        tmp_result =tmp_result.merge(last_quote_record, on='code', how='left')
        tmp_result = tmp_result.merge(current_list_daynum, on='code', how='left')

        current_daynum = np.zeros(tmp_result.shape[0])
        for i in range(tmp_result.shape[0]):
            tmp_info = tmp_result.iloc[i]
            if tmp_info['sign'] == 1:
                if (tmp_info['last_sign'] != 1) & (tmp_info[fieldNameListDayNum] != 0): # recover from stop
                    current_daynum[i] = 1
                elif tmp_info['last_daynum'] > 0: # continue counting
                    current_daynum[i] = tmp_info['last_daynum'] + 1
        current_daynum[current_daynum > 30] = 0 # after 30 days, stop counting

        tmp_result.loc[:, fieldNameResumeDayNum] = current_daynum
        tmp_result.loc[:, 'date'] = tmp_date
        tmp_result = tmp_result.loc[~tmp_result['sign'].isnull()] # only record those with quotes today

        # select final columns
        tmp_result = tmp_result[['date', 'code', fieldNameResumeDayNum]]

        # append to the buffer
        resume_daynum = resume_daynum.append(tmp_result)

        # update last quote record & last daynum
        last_quote_record = current_quote_record.copy()
        last_quote_record = last_quote_record.rename(columns={'sign': 'last_sign'})
        last_resume_daynum = tmp_result[['code', fieldNameResumeDayNum]].copy()
        last_resume_daynum = last_resume_daynum.rename(columns={fieldNameResumeDayNum: 'last_daynum'})

    data_result = data_result.merge(resume_daynum, how='left', on=['date', 'code'])
    data_result = data_result.loc[data_result['date'] > target_max_date]

    if not data_result.empty:
        # reconnect to db
        quant_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

        data_result.to_sql(targetTableName, quant_engine, index=False, if_exists='append')
    pass

def airflowCallable():
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    calIncrm(quant_engine)

if __name__ == '__main__':
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # calFull(quant_engine)
    # t_start = time.clock()
    calIncrm(quant_engine)
    # print(time.clock() - t_start)