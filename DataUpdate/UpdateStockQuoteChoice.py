# -*- coding:utf-8 -*-
import sys
import os
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant
from Utils.EmQuantAPI import c

targetTableName = 'STOCK_QUOTE_CHOICE'
calendarTableName = 'TRADE_CALENDAR'
adjQuoteTableName = 'STOCK_FORWARD_ADJ_QUOTE_CHOICE'

targetColNames = ['date', 'code', 'open', 'high', 'low', 'close', 'adj_factor', 'volume', 'amount', 'turnover']
choiceFieldName = "Open,High,low,Close,TAFACTOR,Volume,Amount,Turn"
adjColNames = ['open', 'high', 'low', 'close']

def mainCallback(quantdata):
    """
    mainCallback 是主回调函数，可捕捉如下错误
    在start函数第三个参数位传入，该函数只有一个为c.EmQuantData类型的参数quantdata
    :param quantdata:c.EmQuantData
    :return:
    """
    print("mainCallback", str(quantdata))

    # 登录掉线或者 登陆数达到上线（即登录被踢下线） 这时所有的服务都会停止
    if str(quantdata.ErrorCode) == "10001011" or str(quantdata.ErrorCode) == "10001009":
        print("Your account is disconnect. You can force login automatically here if you need.")

    # 行情登录验证失败（每次连接行情服务器时需要登录验证）或者行情流量验证失败时，会取消所有订阅，用户需根据具体情况处理
    elif str(quantdata.ErrorCode) == "10001021" or str(quantdata.ErrorCode) == "10001022":
        print("Your all csq subscribe have stopped.")

    # 行情服务器断线自动重连连续6次失败（1分钟左右）不过重连尝试还会继续进行直到成功为止，遇到这种情况需要确认两边的网络状况
    elif str(quantdata.ErrorCode) == "10002009":
        print("Your all csq subscribe have stopped.")
    else:
        pass

def csdMissedCode(sql_engine, stock_exist_points, start_date, end_date, table_name):
    # total code in the periods
    code_to_load = stock_exist_points.index.tolist()
    # code_to_load = list(map(lambda x: x[:-3], code_to_load))

    # code already in database
    sql_statement = "select distinct `code` from %s where date between '%s' and '%s'" % (table_name, start_date, end_date)
    existing_code = pd.read_sql(sql_statement, sql_engine)
    existing_code = existing_code['code'].tolist()

    missed_code = [x for x in code_to_load if x not in existing_code]

    # missed_code = list(map(lambda x: x + '.SH' if x[0] == '6' else x + '.SZ', missed_code))

    return missed_code


def calRecordNum(trade_dates):
    tot_record_num = 0
    for tmp_date in trade_dates:
        tmp_codes = c.sector("001004", tmp_date)
        tot_record_num += len(tmp_codes.Data) / 2

    return  tot_record_num

def getStockExistInfo(trade_dates):  # csd can get data easily
    stock_record = pd.DataFrame([])
    tot_record_num = 0
    for tmp_date in trade_dates:
        tmp_codes = c.sector("001004", tmp_date)
        tmp_codes = [tmp_codes.Data[i] for i in range(0, len(tmp_codes.Data), 2)] # only get stock codes
        tmp_record = pd.DataFrame({'code': tmp_codes})
        tmp_record.loc[:, 'date'] = tmp_date
        stock_record = stock_record.append(tmp_record)
        tot_record_num += len(tmp_codes)

    stock_record.loc[:, 'exist'] = 1
    exist_stock_flag = stock_record.pivot_table(values='exist', index='date', columns='code', aggfunc=np.sum)

    # get the first and last existing date
    stock_existing_points = pd.DataFrame([], index=exist_stock_flag.columns, columns=['first', 'last'])
    for tmp_code in exist_stock_flag.columns:
        tmp_all_existing_dates = exist_stock_flag.loc[~exist_stock_flag[tmp_code].isnull()].index
        stock_existing_points.loc[tmp_code] = [tmp_all_existing_dates[0], tmp_all_existing_dates[-1]]

    return exist_stock_flag, stock_existing_points, tot_record_num

def updateCSS(max_data_num, start_date='2007-01-01', end_date = ''):
    # get trade calendar
    quant_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    if end_date == '':
        end_date = datetime.strftime(datetime.today() - timedelta(1), '%Y-%m-%d') # yesterday
    sql_statement = "select date from %s where date between '%s' and '%s'" % (calendarTableName, start_date, end_date)
    trade_dates = pd.read_sql(sql_statement, quant_engine)
    trade_dates = trade_dates['date'].values

    # check if table exists, and make sure no duplicate data
    sql_statement = "select TABLE_NAME from INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA='quant' and TABLE_NAME='%s';" % targetTableName
    tmp_result = pd.read_sql(sql_statement, quant_engine)
    if tmp_result.shape[0] > 0: # table already exist
        write_method = 'append' # append data, keep the old data

        tmp_conn = pymysql.connect(**ConfigQuant)
        with tmp_conn.cursor() as tmp_cur:
            sql_statement = "delete from %s where date between '%s' and '%s'" % (targetTableName, start_date, end_date) # avoid duplicated data
            tmp_num = tmp_cur.execute(sql_statement)
            print('delete duplicate record num:', tmp_num)

            sql_statement = "delete from %s where date between '%s' and '%s'" % (adjQuoteTableName, start_date, end_date)  # avoid duplicated data
            tmp_num = tmp_cur.execute(sql_statement)
            print('delete duplicate record num:', tmp_num)
        tmp_conn.commit()
        tmp_conn.close()
    else:
        write_method = 'replace'

    # choice API data
    loginResult = c.start("ForceLogin=1", '', mainCallback) # login
    if (loginResult.ErrorCode != 0):
        print("login in fail")
        exit()

    # get total record num
    tot_record_num = calRecordNum(trade_dates)  # c.sector  not consume data limit
    print('total record num:', tot_record_num)
    if max_data_num < tot_record_num * (len(targetColNames) - 2):
        print('data quota not enough to download all the data')
        raise ValueError

    # loop over trade dates to download stock quotes from API
    choice_data = pd.DataFrame([], columns=targetColNames)
    dump_data_size = 50000 # dump data every n records
    # write_method = 'replace'
    tmp_index = 0
    current_dump_num = 0
    for tmp_date in trade_dates:
        tmp_codes = c.sector("001004", tmp_date)
        tmp_codes = [tmp_codes.Data[i] for i in range(0, len(tmp_codes.Data), 2)]
        tmp_codes = ','.join(tmp_codes)
        tmp_data = c.css(tmp_codes, choiceFieldName, "TradeDate=%s,AdjustFlag=1" % tmp_date) # 1: unadjusted price
        for tmp_c, tmp_quote in tmp_data.Data.items():
            tmp_c = tmp_c[:-3]
            tmp_buffer = [tmp_date, tmp_c]
            tmp_buffer.extend(tmp_quote)
            choice_data.loc[tmp_index] = tmp_buffer
            tmp_index += 1

        tmp_cols = set(targetColNames) - {'date', 'code'}
        for tmp_col in tmp_cols:
            choice_data.loc[:, tmp_col] = choice_data[tmp_col].astype('float') # make sure datetype consistent

        # dump data into database by trunk (or reach the final)
        if (choice_data.shape[0] > dump_data_size) or (tmp_date == trade_dates[-1]):
            choice_data.loc[:, 'time_stamp'] = datetime.now()
            choice_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)

            # Houfuquan quotes
            for tmp_col in adjColNames:
                choice_data.loc[:, tmp_col] = choice_data[tmp_col] * choice_data['adj_factor']
            choice_data = choice_data.drop('adj_factor', axis=1)
            choice_data.to_sql(adjQuoteTableName, quant_engine, index=False, if_exists=write_method)

            current_dump_num += choice_data.shape[0]
            print('finish dump data:', current_dump_num)

            write_method = 'append'
            choice_data = pd.DataFrame([], columns=targetColNames)

    c.stop() # logout

def updateCSD(max_data_num, start_date='2007-01-01', end_date = ''):
    # get trade calendar
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    if end_date == '':
        end_date = datetime.strftime(datetime.today() - timedelta(1), '%Y-%m-%d')  # yesterday
    sql_statement = "select date from %s where date between '%s' and '%s'" % (calendarTableName, start_date, end_date)
    trade_dates = pd.read_sql(sql_statement, quant_engine)
    trade_dates = trade_dates['date'].values

    # check if table exists, and make sure no duplicate data
    sql_statement = "select TABLE_NAME from INFORMATION_SCHEMA.TABLES where TABLE_SCHEMA='quant' and TABLE_NAME='%s';" % targetTableName
    tmp_result = pd.read_sql(sql_statement, quant_engine)
    if tmp_result.shape[0] > 0:  # table already exist
        write_method = 'append'  # append data, keep the old data

        tmp_conn = pymysql.connect(**ConfigQuant)
        with tmp_conn.cursor() as tmp_cur:
            sql_statement = "delete from %s where date between '%s' and '%s'" % (targetTableName, start_date, end_date)  # avoid duplicated data
            tmp_num = tmp_cur.execute(sql_statement)
            print('delete duplicate record num:', tmp_num)

            sql_statement = "delete from %s where date between '%s' and '%s'" % (adjQuoteTableName, start_date, end_date)  # avoid duplicated data
            tmp_num = tmp_cur.execute(sql_statement)
            print('delete duplicate record num:', tmp_num)

        tmp_conn.commit()
        tmp_conn.close()
    else:
        write_method = 'replace'

    # choice API data
    loginResult = c.start("ForceLogin=1", '', mainCallback)  # login
    if (loginResult.ErrorCode != 0):
        print("login in fail")
        exit()

    # get total record num
    stock_exist_flag, stock_exist_points, record_num = getStockExistInfo(trade_dates)  # c.sector  not consume data limit
    print('total record num:', record_num)

    # missed_code = csdMissedCode(quant_engine, stock_exist_points, start_date, end_date, targetTableName)
    # stock_exist_points = stock_exist_points.loc[missed_code]

    if max_data_num < record_num * (len(targetColNames) - 2): # check if there is enough quota to download all the data
        print('data quota not enough to download all the data')
        raise ValueError

    # get data from API
    choice_data = pd.DataFrame([], columns=targetColNames)
    dump_data_size = 50000
    current_dump_num = 0
    for tmp_code in stock_exist_points.index:
        tmp_start_point = stock_exist_points.loc[tmp_code, 'first']
        tmp_end_point = stock_exist_points.loc[tmp_code, 'last']
        tmp_data = c.csd(tmp_code, choiceFieldName, tmp_start_point, tmp_end_point,
                     "period=1,adjustflag=1,curtype=1,pricetype=1,order=1,market=CNSESH")

        # convert data format
        tmp_quote = {}
        tmp_trade_dates = trade_dates[(trade_dates >= tmp_start_point) & (trade_dates <= tmp_end_point)]
        tmp_col_names = [x for x in targetColNames if x not in ['date', 'code']]
        for i, tmp_col in enumerate(tmp_col_names):
            tmp_quote[tmp_col] = tmp_data.Data[tmp_code][i]
        tmp_quote = pd.DataFrame(tmp_quote)
        tmp_quote.loc[:, 'date'] = tmp_trade_dates
        tmp_quote.loc[:, 'code'] = tmp_code[:-3]

        for tmp_col in tmp_col_names:
            tmp_quote.loc[:, tmp_col] = tmp_quote[tmp_col].astype('float') # ensure the datatype is numerical
        tmp_quote = tmp_quote[targetColNames]
        choice_data = choice_data.append(tmp_quote)

        if (choice_data.shape[0] > dump_data_size) or (tmp_code == stock_exist_points.index[-1]): # dump data into database when it is large enough
            choice_data.loc[:, 'time_stamp'] = datetime.now()
            choice_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)

            # Houfuquan quotes
            for tmp_col in adjColNames:
                choice_data.loc[:, tmp_col] = choice_data[tmp_col] * choice_data['adj_factor']
            choice_data = choice_data.drop('adj_factor', axis=1)
            choice_data.to_sql(adjQuoteTableName, quant_engine, index=False, if_exists=write_method)

            current_dump_num += choice_data.shape[0]
            print('finish dump data:', current_dump_num)

            write_method = 'append'
            choice_data = pd.DataFrame([], columns=targetColNames)



if __name__ == '__main__':
    data_quota = 4000000
    start_date = '2010-01-01'
    end_date = '2010-12-31'

    # choice API data
    # updateCSS(max_data_num=data_quota, start_date=start_date, end_date=end_date)

    updateCSD(max_data_num=data_quota, start_date=start_date, end_date=end_date)