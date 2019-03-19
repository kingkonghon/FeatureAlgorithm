from datetime import datetime
from sqlalchemy import create_engine
from SelectedDBConfigStationary import ConfigQuant, ConfigSpider2, StockAppSchema
from RLSellRobot import choose_action
import pandas as pd
import numpy as np
import tushare as ts
import time
from urllib.error import HTTPError
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from socket import timeout
from time import sleep

adminMailConfig = [
    {
        'smtp_server': 'smtp.163.com',
        'smtp_port': 465,
        'user': '13602819622@163.com',
        'password': 'll900515',

        'sender': '13602819622@163.com',
        'receiver': '13602819622@163.com',
    },

    # {
    #     'smtp_server': 'smtp.mxhichina.com',
    #     'smtp_port': 25,
    #     'user': 'jianghan@nuoyuan.com.cn',
    #     'password': 'll@900515',
    #
    #     'sender': 'jianghan@nuoyuan.com.cn',
    #     'receiver': 'quant_atm@aliyun.com',
    # },
]

officalMailConfig = [
    {
        'smtp_server': 'smtp.aliyun.com',
        'smtp_port': 465,
        'user': 'quant_atm@aliyun.com',
        'password': 'Abc12345',

        'sender': 'quant_atm@aliyun.com',
        'receiver': 'quant_atm@aliyun.com',
    },
]

# stockPool1000Path = '/home/quant/predict_live/stock_pool.csv'
stockPool1000Path = r'F:\FeatureAlgorithm\Live\stock_pool.csv'

isBackTest = False


def getTradeCalendar(sql_engine):
    calendarTableName = 'TRADE_CALENDAR'

    # get calendar from db
    sql_statement = "select date from %s" % calendarTableName
    tmp_conn = sql_engine.connect()
    trade_calendar = pd.read_sql(sql_statement, tmp_conn)

    trade_calendar = trade_calendar['date'].unique()

    tmp_conn.close()

    return trade_calendar

def getLatestTradeDay(trade_calendar):
    today = datetime.now()
    today = datetime.strftime(today, '%Y-%m-%d')

    if today in trade_calendar:  # today is trade day
        yesterday = trade_calendar[trade_calendar < today][-1]  # find the latest trade date
    else:
        today = 'non_trade_day'
        yesterday = 'non_trade_day'

    return today, yesterday

def writeDB(sql_enine, schema, table_name, data, add_primary_key=False):
    tmp_conn = sql_enine.connect()

    # check if table exists
    sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = '%s' AND table_name = '%s'" % (schema, table_name)
    tmp_table_exist = pd.read_sql(sql_statement, tmp_conn)
    tmp_table_exist = tmp_table_exist.iloc[0, 0]

    if tmp_table_exist == 0:  # table not exists
        if add_primary_key == True:  # create new table, and set primary key
            data.loc[:, 'id'] = list(range(1, data.shape[0] + 1))  # auto incremental primary key should begin with 1
            data.to_sql(table_name, tmp_conn, index=False, if_exists='replace')
            sql_statement = "alter table `%s` add primary key(id)" % table_name  # set id as primary key
            tmp_result = tmp_conn.execute(sql_statement)
            print('set id as primary key for table %s' % table_name)
            sql_statement = "alter TABLE `%s` change id id BIGINT(20) AUTO_INCREMENT;" % table_name  # set id as auto incremental
            tmp_result = tmp_conn.execute(sql_statement)
            print('set id as auto incremental for table %s' % table_name)
        else:  # only create new table
            data.to_sql(table_name, tmp_conn, index=False, if_exists='replace')
    else:   # table exists
        today = data['date'].unique()
        if today.size != 1:
            print('data not at the same date')
            raise ValueError
        else:
            today = today[0]

        # if there are old data in the table, delete them first before writing new data
        sql_statement = "select count(1) from `%s` where `date`='%s'" % (table_name, today)
        tmp_today_record_num = pd.read_sql(sql_statement, tmp_conn)
        tmp_today_record_num = tmp_today_record_num.iloc[0, 0]
        if tmp_today_record_num > 0:
            print("today's record already in db")
            sql_statement = "delete from `%s` where `date`='%s'" % (table_name, today)  # delete obsolete data
            tmp_conn.execute(sql_statement)
            print("delete old records from %s on %s" % (table_name, today))

        data.to_sql(table_name, tmp_conn, index=False, if_exists='append')
        print('new records  has been written into db')

        tmp_conn.close()


def loadYesterdayPrediction(prediction_table_name, backup_prediction_table_name, sql_engine, date):
    # get preditions of latest trade date from db
    tmp_conn = sql_engine.connect()
    sql_statement = "select * from %s where `date`='%s'" % (prediction_table_name, date)
    predictions = pd.read_sql(sql_statement, tmp_conn)

    if predictions.empty:
        print('no prediction today, try backup prediction instead')

        sql_statement = "select * from %s where `date`='%s'" % (backup_prediction_table_name, date)
        predictions = pd.read_sql(sql_statement, tmp_conn)

        if predictions.empty:
            print('no backup prediction today, error')
            raise ValueError

    tmp_conn.close()

    predictions = predictions.drop(['date', 'time_stamp'], axis=1)

    return predictions

def loadCurrentPosition(position_table_name, sql_engine):
    tmp_conn = sql_engine.connect()

    # check if position recording table exists
    sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = 'quant' AND table_name = '%s'" % position_table_name
    tmp_table_exist = pd.read_sql(sql_statement, tmp_conn)
    tmp_table_exist = tmp_table_exist.iloc[0, 0]

    if tmp_table_exist == 0:  # not exist
        current_position = pd.DataFrame([])
    elif tmp_table_exist == 1:  # table exists, there might be outstanding positions
        sql_statement = "select * from %s" % position_table_name
        current_position = pd.read_sql(sql_statement, tmp_conn)
        current_position = current_position.drop('time_stamp', axis=1)
    else:
        print('table num error')
        raise ValueError

    tmp_conn.close()

    return current_position

def loadHS300Constituents(hs300_table_name, sql_engine, date):
    tmp_conn = sql_engine.connect()

    sql_statement = "select * from %s where `date`='%s'" % (hs300_table_name, date)
    hs300_weights = pd.read_sql(sql_statement, tmp_conn)

    if hs300_weights.empty:
        print('no hs300 weight')
        raise ValueError

    tmp_conn.close()

    hs300_weights = hs300_weights.drop('time_stamp', axis=1)

    return hs300_weights

def loadSpecifiedStockCodes():
    stock_list = pd.read_csv(stockPool1000Path, header=None)
    stock_list = stock_list[0].apply(lambda x: x[:-3]).tolist()

    return stock_list

def getTushareSuspensionFlag(token, today):
    ts_api = ts.pro_api(token)
    today_ts = today.replace('-', '')
    suspension_flag = ts_api.query('suspend', ts_code='', suspend_date=today_ts, resume_date='', fiedls='')

    suspension_flag.loc[:, 'code'] = suspension_flag['ts_code'].apply(lambda x: x[:-3])  # change the format of the code

    suspension_flag = suspension_flag['code'].unique()
    return suspension_flag

def getTushareStockHistoricalQuote(token, date, code_list):
    ts_api = ts.pro_api(token)
    today_ts = date.replace('-', '')
    stock_quote = ts_api.daily(trade_date=today_ts)

    stock_quote.loc[:, 'code'] = stock_quote['ts_code'].apply(lambda x: x[:-3])  # change the format of the code
    stock_quote.loc[:, 'date'] = stock_quote['trade_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))  # change the format of the date

    stock_quote = stock_quote.loc[stock_quote['code'].isin(code_list), ['code', 'open']]  # only need the open price
    return stock_quote


def getTushareStockQuote(code_list):
    # real_time_quote = ts.get_today_all()
    real_time_quote = ts.get_realtime_quotes(code_list)

    real_time_quote = real_time_quote.drop_duplicates('code')
    real_time_quote.loc[:, 'open'] = real_time_quote['open'].astype('float')

    return real_time_quote[['code', 'open']]

def getHistoricalProbability(codes, columns, day_num, trade_calendar, today, his_table_name, sql_engine):
    his_dates = trade_calendar[trade_calendar < today]
    first_date = his_dates[-day_num]
    last_date = his_dates[-1]

    tmp_columns = columns.copy()
    tmp_columns.extend(['date', 'code'])
    tmp_columns_str = list(map(lambda x: "`%s`" % x, tmp_columns)) # write column names into sql statement
    tmp_columns_str = ','.join(tmp_columns_str)

    tmp_codes_str = list(map(lambda x: "'%s'" % x, codes)) # write stock codes into sql statement
    tmp_codes_str = ','.join(tmp_codes_str)

    sql_statement = "select %s from `%s` where `code` in (%s) and `date` >= '%s' and `date` <= '%s'" % (tmp_columns_str, his_table_name, tmp_codes_str, first_date, last_date)
    tmp_conn = sql_engine.connect()
    his_proba = pd.read_sql(sql_statement, tmp_conn)
    tmp_conn.close()

    return his_proba

def getTushareStockRiseFallStop(token, yesterday, stop_limit, real_time_quote):
    # ts_api = ts.pro_api(token)
    # yesterday_format = yesterday.replace('-', '')
    # pre_close = ts_api.daily(trade_date=yesterday_format)
    # pre_close.loc[:, 'code'] = pre_close['ts_code'].apply(lambda x: x[:-3])
    pre_close = getTushareProStockQuote(yesterday, token)

    real_time_quote = real_time_quote.merge(pre_close[['code', 'close']], how='inner', on='code')  # combine today's open with yesterday's close
    real_time_quote = real_time_quote.rename({'close': 'pre_close'}, axis=1)

    # calculate flag for open price reaching rise stop or fall stop
    real_time_quote.loc[:, 'rise_stop'] = (real_time_quote['open'] / real_time_quote['pre_close'] - 1) > stop_limit
    real_time_quote.loc[:, 'fall_stop'] = (real_time_quote['open'] / real_time_quote['pre_close'] - 1) < - stop_limit

    return real_time_quote

def getTushareProStockQuote(date, token):
    ts_api = ts.pro_api(token)
    date_format = date.replace('-', '')  # '2018-01-01' to '20180101'
    stock_quote = ts_api.daily(trade_date=date_format)

    stock_quote.loc[:, 'code'] = stock_quote['ts_code'].apply(lambda x: x[:-3])

    return stock_quote

def getCurrentAvaliableFunds(sql_engine, trade_reocrd_table_name, id_col_name, initial_fund):
    tmp_conn = sql_engine.connect()

    sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = 'quant' AND table_name = '%s'" % trade_reocrd_table_name
    tmp_if_table_exist = pd.read_sql(sql_statement, tmp_conn)
    tmp_if_table_exist = tmp_if_table_exist.iloc[0, 0]

    if tmp_if_table_exist == 0:  # table does not exist
        available_fund = initial_fund
        last_record_id = 0
    elif tmp_if_table_exist == 1:
        sql_statement = "select * from `%s` where `%s` = (select max(record_id) from %s)" % (trade_reocrd_table_name, id_col_name, trade_reocrd_table_name)
        latest_trade_record = pd.read_sql(sql_statement, tmp_conn)

        available_fund = latest_trade_record['balance'].iloc[0]
        last_record_id = latest_trade_record['record_id'].iloc[0]
    else:
        print('get available fund error')
        raise ValueError

    # if not latest_trade_record.empty:
    #     available_fund = latest_trade_record['balance'].iloc[0]
    #     last_record_id = latest_trade_record['record_id'].iloc[0]
    # else:
    #     available_fund = np.nan
    #     last_record_id = np.nan

    tmp_conn.close()

    return available_fund, last_record_id

def loadLatestDailyReturn(sql_conn, performance_table_name, index_table_name):
    ret_col = 'daily_return'
    date_col = 'date'
    index_code = '399300'

    # check if table exists
    sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = 'quant' AND table_name = '%s'" % performance_table_name
    tmp_if_table_exist = pd.read_sql(sql_statement, sql_conn)
    tmp_if_table_exist = tmp_if_table_exist.iloc[0, 0]

    if tmp_if_table_exist == 0:  # table does not exist
        latest_trade_date = 'null'
        latest_ret = 0
    elif tmp_if_table_exist == 1:
        latest_statistics = {}

        # stratgy returns
        # sql_statement = "select `%s`, `%s` from `%s` where `%s` = (select max(`%s`) from %s)" % (
        #     date_col, ret_col, performance_table_name, date_col, date_col, performance_table_name)
        sql_statement = "select `%s`, `%s` from `%s`" % (date_col, ret_col, performance_table_name)
        all_performance = pd.read_sql(sql_statement, sql_conn)

        latest_trade_date = all_performance[date_col].max()
        first_trade_date = all_performance[date_col].min()
        latest_ret = all_performance.loc[all_performance[date_col] == latest_trade_date, 'daily_return'].iloc[0]

        tot_ret = (1 + all_performance[ret_col]).prod() - 1  # cumulative return

        # hs300 return
        sql_statement = "select `date`, `open`, `close` from `%s` where `code` = '%s' and date between '%s' and '%s'" % (index_table_name, index_code, first_trade_date, latest_trade_date)
        index_quotes = pd.read_sql(sql_statement, sql_conn)
        index_quotes = index_quotes.sort_values('date')
        index_tot_ret = index_quotes.loc[index_quotes['date'] == latest_trade_date, 'close'].iloc[0] / index_quotes.loc[index_quotes['date'] == first_trade_date, 'open'].iloc[0] - 1
        index_latest_ret = index_quotes.loc[index_quotes['date'] == latest_trade_date, 'close'].iloc[0] / index_quotes['close'].iloc[-2] - 1

        # alpha return
        tot_alpha_return = tot_ret - index_tot_ret
        latest_alpha_return = latest_ret - index_latest_ret

        # latest_trade_date = all_performance[date_col].iloc[0]
        # latest_ret = latest_performance[ret_col].iloc[0]

        latest_statistics['date'] = latest_trade_date
        latest_statistics['daily_ret'] = latest_ret
        latest_statistics['tot_ret'] = tot_ret
        latest_statistics['daily_alpha_ret'] = latest_alpha_return
        latest_statistics['tot_alpha_ret'] = tot_alpha_return

    return latest_statistics


def strategyBasicPreTrade():
    print(datetime.now())

    # setting
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    # sql_engine_spider = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))
    # config_stock_app = ConfigQuant.copy()
    # config_stock_app['db'] = StockAppSchema
    # sql_engine_stock_app = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**config_stock_app))  # different schema
    predictionTableName = 'LGBM_LIVE_PREDICTION_FINAL'   #  historical prediction records using complete features
    backupPredictionTableName = 'LGBM_LIVE_PREDICTION_TEMP'   # historical prediction records using temporary features
    strategyPositionTableName = 'NAIVE_LGBM_STRATEGY_POS'
    strategyTmpSellListTableName = 'NAIVE_LGBM_STRATEGY_TMP_SELL_LIST'
    strategyTmpBuyListTableName = 'NAIVE_LGBM_STRATEGY_TMP_BUY_LIST'
    strategyRecommendationTableName = 'NAIVE_LGBM_RECOMMENDATION_HIS_RECORDS'   # historical recommendation records (top 20 stocks)
    hs300TableName = 'HS300_WEIGHTS'
    maxPreBuyNum = 20
    maxHoldingPeriod = 20
    weightedProbabilityLabel = 'weighted_proba'
    probaLabel = ['Y_2D', 'Y_5D', 'Y_10D', 'Y_20D']
    buyProbabilityLabel = 'Y_2D'   # label used to select stock
    stockPool = '1000'  # candidate stocks to choose from
    checkDayNum = [2, 5, 10]
    closeProbabilityThreshold = 0.5
    RLMatrixLen = 10   # the length of lag of predicting probabilities

    # load latest trade day
    trade_calendar = getTradeCalendar(sql_engine)
    today, yesterday = getLatestTradeDay(trade_calendar)
    if today == 'non_trade_day':  # today is not trade day
        print('today is not trade day')
        return

    # fill header of email
    mailContent = {
        'from': 'Naive LGBM strategy',
        'to': 'Subscriber',
        'subject': 'Stock recommendation',
    }

    mailPreTradeContent = {
        'from': 'Naive LGBM strategy',
        'to': 'Subscriber',
        'subject': 'Stock Pre-Trade List %s' % today,
    }

    mailContent['content'] = "Today's recommendation stock codes (%s)" % today  # set email content
    mailPreTradeContent['content'] = "Today's proposed trade list:"

    # load predictions
    predictions = loadYesterdayPrediction(predictionTableName, backupPredictionTableName, sql_engine, yesterday)

    # load current position
    current_position = loadCurrentPosition(strategyPositionTableName, sql_engine)

    # ===== decide outstanding position planning to sell
    pre_sell_list = []
    if not current_position.empty:
        # == positions that exceed the maximum holding periods
        tmp_close_pos = current_position.loc[current_position['holding_period'] >= maxHoldingPeriod, 'code'].tolist()
        pre_sell_list.extend(tmp_close_pos)

        # # positions at critical points that need to check if probability suggest closing that position
        # for tmp_day in checkDayNum:
        #     tmp_pos = current_position.loc[(maxHoldingPeriod - current_position['holding_period']) == tmp_day]
        #     tmp_prd = predictions.loc[predictions['code'].isin(tmp_pos['code'].tolist())]
        #     if not tmp_pos.empty:
        #         tmp_label = 'Y_%dD' % tmp_day
        #         tmp_close_pos = tmp_prd.loc[tmp_prd[tmp_label] < closeProbabilityThreshold, ['code', 'holding_period']]  # probability lower than threshold, closed
        #         pre_sell_list = pre_sell_list.append(tmp_close_pos)

        # == load current positions' prediction probability for the past 10 days, input into reinforce learning and output if it needs to be sold
        tmp_codes = current_position['code'].tolist()
        pos_his_proba = getHistoricalProbability(tmp_codes, probaLabel, RLMatrixLen, trade_calendar, today, predictionTableName, sql_engine)  # load historical probability data ( dates < today, take most recent RLMatrixLen days)

        for tmp_c in tmp_codes:  # input into RL one by one
            tmp_his_proba = pos_his_proba.loc[pos_his_proba['code'] == tmp_c, probaLabel].values
            tmp_action = choose_action(tmp_his_proba)
            if tmp_action == 1:  # suggested by robot to sell
                pre_sell_list.append(tmp_c)

    # ======= decide potential orders (recommendation list) that could bought from
    # screen out stocks not in the pool
    if stockPool == 'HS300':   # only select from HS300 constituents
        hs300_weights = loadHS300Constituents(hs300TableName, sql_engine, yesterday)
        predictions = predictions.loc[predictions['code'].isin(hs300_weights['code'].tolist())]
    elif stockPool == '1000':   # only select from specified 1000 stocks
        specified_stock_list = loadSpecifiedStockCodes()
        predictions = predictions.loc[predictions['code'].isin(specified_stock_list)]

    # use specified kind of prediction to select stocks
    predictions.loc[:, weightedProbabilityLabel] = (predictions['Y_2D'] + predictions['Y_5D'] + predictions['Y_10D'] + predictions['Y_20D']) * 0.25  # weighted average of 4 probabilities
    predictions = predictions.sort_values(buyProbabilityLabel, ascending=False) # sort predict probabilities

    pre_buy_list = predictions.iloc[:maxPreBuyNum]  # choose the top probability stocks

    # ===== drop stock in pre-sell list, if it's also in pre-buy list
    tmp_code_list = pre_buy_list['code'].tolist()
    contradict_list = [x for x in pre_sell_list if x in tmp_code_list]
    pre_sell_list = [x for x in pre_sell_list if x not in tmp_code_list]
    if len(contradict_list) > 0:
        print('buy sell contradiction:', contradict_list)

    # ==== send email (recommendation list)
    tmp_code_list = pre_buy_list['code'].tolist()
    tmp_rank_num = list(range(1, pre_buy_list.shape[0]+1))
    tmp_content = ['%d. %s' % (tmp_num, tmp_code) for tmp_num, tmp_code in zip(tmp_rank_num, tmp_code_list)]
    mailContent['content'] += '\n' + '\n'.join(tmp_content)

    for tmp_config in adminMailConfig:
        sendEmail(mailContent, tmp_config)   # send email to different subscribers

    # ==== send email (proposed trade list)
    tmp_code_list = pre_buy_list['code'].tolist()
    tmp_rank_num = list(range(1, pre_buy_list.shape[0] + 1))
    tmp_content = ['%d. %s' % (tmp_num, tmp_code) for tmp_num, tmp_code in zip(tmp_rank_num, tmp_code_list)]
    mailPreTradeContent['content'] += '\nproposed buy list:\n' + '\n'.join(tmp_content)  # pre-buy list

    tmp_code_list = pre_sell_list.copy()
    mailPreTradeContent['content'] += '\n\nproposed sell list:\n' + '\n'.join(tmp_code_list)  # pre-buy list

    for tmp_config in adminMailConfig:
        sendEmail(mailPreTradeContent, tmp_config)  # send email to different subscribers

    for tmp_config in officalMailConfig:
        sendEmail(mailPreTradeContent, tmp_config)  # send email to different subscribers

    # ==== write db (pre buy list & pre sell list)
    tmp_conn = sql_engine.connect()
    # pre buy list (not actually buy, some in the list may suspense, or open price reaching rise limit)
    pre_buy_list.loc[:, 'time_stamp'] = datetime.now()
    pre_buy_list.to_sql(strategyTmpBuyListTableName, tmp_conn, index=False, if_exists='replace')

    # pre sell list
    sql_statement = 'truncate table `%s`' % strategyTmpSellListTableName  # clear up previous records (in case pre-sell list empty)
    tmp_conn.execute(sql_statement)

    if not len(pre_sell_list) == 0:
        pre_sell_list = pd.DataFrame(pre_sell_list, columns=['code'])
        pre_sell_list.loc[:, 'time_stamp'] = datetime.now()
        pre_sell_list.to_sql(strategyTmpSellListTableName, tmp_conn, index=False, if_exists='append')
    tmp_conn.close()

    # recommendation list (historical records of pre buy list, for stock app)
    tmp_conn = sql_engine.connect()  # read stock company name from spider schema
    sql_statement = "select `code`, `name` from STOCK_DESCRIPTION"
    stock_name = pd.read_sql(sql_statement, tmp_conn)
    tmp_conn.close()

    tmp_cols = pre_buy_list.columns.tolist()
    tmp_cols.insert(0, 'date')
    pre_buy_list.loc[:, 'date'] = today
    recommendation_list = pre_buy_list[tmp_cols]  # rearrange column order
    recommendation_list = recommendation_list.merge(stock_name, on='code', how='left')
    # writeDB(sql_engine_stock_app, config_stock_app['db'], strategyRecommendationTableName, recommendation_list, True)

    # recommendation list (backup in quant)
    writeDB(sql_engine, ConfigQuant['db'], strategyRecommendationTableName, recommendation_list, True)

def strategyBasicActualTrade():
    print(datetime.now())

    # setting
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    # config_stock_app = ConfigQuant.copy()
    # config_stock_app['db'] = StockAppSchema
    # sql_engine_stock_app = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**config_stock_app))  # different schema
    strategyPositionTableName = 'NAIVE_LGBM_STRATEGY_POS'   # table recording current positions
    strategyPositionBackupTableName = 'NAIVE_LGBM_STRATEGY_POS_BACKUP'  # table recording historical current positions
    strategyTmpSellListTableName = 'NAIVE_LGBM_STRATEGY_TMP_SELL_LIST'  # table recording list of suggested buy list
    strategyTmpBuyListTableName = 'NAIVE_LGBM_STRATEGY_TMP_BUY_LIST'   # table recording list of suggested sell list
    strategyTradeRecordTableName = 'NAIVE_LGBM_STRATEGY_HIS_RECORDS'  # table recording historical trading records
    strategyActionTableName = 'NAIVE_LGBM_STRATEGY_ACTION_RECORDS'
    strategyPerformanceTableName = 'NAIVE_LGBM_STRATEGY_PERFORMANCE'
    indexQuoteTableName = 'STOCK_INDEX_QUOTE_TUSHARE'
    tushareToken = 'e37ce8e806bbfc9bfcb9ac35e68998c1710e7f0714e8fb5f257cd13c'
    maxPosNum = 5
    probaLabelBuy = 'Y_2D'
    his_id_col_name = 'record_id'
    stopLimit = 0.095
    cur_pos_table_cols = ['code', 'start_date', 'holding_period', 'open_price', 'current_price', 'volume', 'cost', 'commission',
                          'floating_PnL', 'time_stamp']
    trd_rcd_table_cols = ['code', 'start_date', 'end_date', 'holding_period', 'open_price', 'close_price', 'volume', 'cost',
                          'revenue', 'commission', 'PnL', 'balance', 'record_id', 'time_stamp']
    trd_act_table_cols = ['date', 'code', 'action', 'execute_price', 'volume', 'amount', 'commission', 'time_stamp']
    buy_commission_rate = 0.001
    sell_commission_rate = 0.002
    strategyInitialFund = 1000000

    # ==== check if today is trade day
    trade_calendar = getTradeCalendar(sql_engine)
    today, yesterday = getLatestTradeDay(trade_calendar)
    if today == 'non_trade_day':
        print('today is not trade day')
        return

    mailContent = {
        'from': 'Naive LGBM strategy',
        'to': 'Subscriber',
        'subject': 'Strategy Transactions %s' % today,
        'content':''
    }

    # ==== load pre buy/sell orders from db
    # sql_statement = "select * from `%s`" % strategyPositionTableName
    # current_position = pd.read_sql(sql_statement, tmp_conn)
    # current_position = current_position.drop('time_stamp', axis=1) # columns: ...(cur_pos_table_cols)
    current_position = loadCurrentPosition(strategyPositionTableName, sql_engine)  # load current position

    # tmp_conn = sql_engine.connect()
    # sql_statement = "select * from `%s`" % strategyTmpBuyListTableName
    # pre_buy_list = pd.read_sql(sql_statement, tmp_conn)
    # pre_buy_list = pre_buy_list.drop('time_stamp', axis=1) # columns: ['code', 'Y_2D', 'Y_5D', 'Y_10D', 'Y_20D']
    pre_buy_list = loadCurrentPosition(strategyTmpBuyListTableName, sql_engine)

    # sql_statement = "select * from `%s`" % strategyTmpSellListTableName
    # pre_sell_list = pd.read_sql(sql_statement, tmp_conn)
    # pre_sell_list = pre_sell_list.drop('time_stamp', axis=1)
    pre_sell_list = loadCurrentPosition(strategyTmpSellListTableName, sql_engine)

    # tmp_conn.close()

    # ==== get suspension stocks from tushare
    ts_suspension_codes = getTushareSuspensionFlag(tushareToken, today)

    strategy_action = pd.DataFrame([])  # store strategy's buy and sell action
    # =================== decide the real sell list first  ======================
    # check pre-sell list: 1. suspension; 2. not in the pre-buy list; 3.fall stop
    if not pre_sell_list.empty:
        real_sell_list = pre_sell_list.loc[~pre_sell_list['code'].isin(ts_suspension_codes)]  # exclude suspended stocks
    else:
        real_sell_list = pd.DataFrame([])

    if not real_sell_list.empty:
        # closing position screened for suspension (still needs to check if open price reaching fall stop limit)
        close_position = current_position.loc[current_position['code'].isin(real_sell_list['code'].values)]

        # get all stock quotes
        if not isBackTest:
            stock_quote = getTushareStockQuote(close_position['code'].tolist())
        else:
            stock_quote = getTushareStockHistoricalQuote(tushareToken, today, close_position['code'].tolist())

        miss_code = close_position.loc[~close_position['code'].isin(stock_quote['code'])]
        while not miss_code.empty:  # tushare download quotes not including the stocks to be closed, retry
            time.sleep(10)
            if not isBackTest:
                tmp_quote = getTushareStockQuote(miss_code['code'].tolist())
            else:
                tmp_quote = getTushareStockHistoricalQuote(tushareToken, today, close_position['code'].tolist())
            stock_quote = stock_quote.append(tmp_quote)  # add new downloaded quotes into the existing ones
            stock_quote = stock_quote.drop_duplicates('code', keep='first')

            miss_code = close_position.loc[~close_position['code'].isin(stock_quote['code'])]  # check again if there still missing codes
            print('missing quotes for closing positions:', miss_code['code'].values)

        # calculate stop limit flag using today's open price and yesterday's close price
        close_quotes = stock_quote.loc[stock_quote['code'].isin(close_position['code'].values)]
        rise_fall_stop_flag = getTushareStockRiseFallStop(tushareToken,yesterday, stopLimit, close_quotes)
        fall_stop_codes = rise_fall_stop_flag.loc[rise_fall_stop_flag['fall_stop'], 'code'].unique()  # cannot sell stocks reaching fall limit

        # real close position, screened for suspension and fall stop
        close_position = close_position.loc[~close_position['code'].isin(fall_stop_codes)]

        # current position after closed
        new_position = current_position.loc[~current_position['code'].isin(close_position['code'].values)]

        # ==== record close position into db
        if not close_position.empty:
            close_position = close_position.merge(close_quotes, on='code', how='left') # columns: ...(cur_pos_table_cols) + 'close'
            close_position = close_position.rename({'open': 'close_price'}, axis=1)

            # get latest balance and record id
            tmp_balance, tmp_last_record_id = getCurrentAvaliableFunds(sql_engine, strategyTradeRecordTableName, his_id_col_name, strategyInitialFund)

            # add other columns (from cur_pos_table_cols to trd_rcd_table_cols)
            close_position.loc[:, 'end_date'] = today
            close_position.loc[:, 'revenue'] = close_position['close_price'] * close_position['volume']
            close_position.loc[:, 'commission'] = close_position['commission'] + close_position['revenue'] * sell_commission_rate
            close_position.loc[:, 'PnL'] = close_position['revenue'] - close_position['cost'] - close_position['commission']
            close_position.loc[:, 'balance'] = close_position['PnL'].cumsum() + tmp_balance
            close_position.loc[:, his_id_col_name] = tmp_last_record_id + np.array(range(1, close_position.shape[0] + 1))
            close_position.loc[:, 'time_stamp'] = datetime.now()

            close_position = close_position[trd_rcd_table_cols]

            # record sell action
            tmp_df = close_position.copy()
            tmp_df.loc[:, 'action'] = 'sell'
            tmp_df = tmp_df.rename({'close_price': 'execute_price', 'revenue':'amount', 'end_date':'date'}, axis=1)
            tmp_df = tmp_df[trd_act_table_cols]
            strategy_action = strategy_action.append(tmp_df)

            # add content to email (close position)
            mailContent['content'] += 'Sell stock at open (%s):\n' % today
            tmp_close_details = close_position.apply(lambda x: '%s @ %.2f' % (x.code, x.close_price), axis=1).tolist()
            tmp_close_details = '\n'.join(tmp_close_details)
            mailContent['content'] += tmp_close_details
            mailContent['content'] +='\n\n'

            # write closed records to trade record table (schema: quant)
            tmp_conn = sql_engine.connect()

            sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = 'quant' AND table_name = '%s'" % strategyTradeRecordTableName  # check if table exist
            tmp_table_exist = tmp_conn.execute(sql_statement)
            tmp_table_exist = tmp_table_exist.fetchone()[0]

            close_position.to_sql(strategyTradeRecordTableName, tmp_conn, index=False, if_exists='append')

            if tmp_table_exist == 0:  # if table not exist previously, set primary key and auto increment
                sql_statement = "alter table `%s` add primary key(%s)" % (strategyTradeRecordTableName, his_id_col_name)  # set id as primary key
                tmp_result = tmp_conn.execute(sql_statement)
                print("set column %s as primary key for table %s" % (his_id_col_name, strategyTradeRecordTableName))
                sql_statement = "alter TABLE `%s` change %s %s BIGINT(20) AUTO_INCREMENT;" % (strategyTradeRecordTableName, his_id_col_name, his_id_col_name)  # set `record_id` as auto incremental
                tmp_result = tmp_conn.execute(sql_statement)
                print("set column %s auto incremental for table %s" % (his_id_col_name, strategyTradeRecordTableName))
            tmp_conn.close()

    else:
        new_position = current_position.copy()
        stock_quote = pd.DataFrame([])

    # ======================= check if new to open new positions  =======================
    if new_position.shape[0] < maxPosNum:
        #  check pre-buy list: 1. suspension; 2. rise stop; 3.already in current position
        if new_position.empty:
            new_position = pd.DataFrame([], columns=cur_pos_table_cols)

        # screen for suspended stocks
        real_buy_list = pre_buy_list.loc[~pre_buy_list['code'].isin(ts_suspension_codes)]

        # get all stock quotes
        if not stock_quote.empty:
            miss_code = real_buy_list.loc[~real_buy_list['code'].isin(stock_quote['code'])]
        else:
            miss_code = real_buy_list.copy()
        while not miss_code.empty:  # tushare download quotes not including the stocks to be opened, retry
            try:
                if not isBackTest:
                    tmp_quote = getTushareStockQuote(miss_code['code'].tolist())
                else:
                    tmp_quote = getTushareStockHistoricalQuote(tushareToken, today, miss_code['code'].tolist())
            except HTTPError:
                print('spider blocked by website, try again in 10 mins')
                time.sleep(300)  # try again in 5 mins
                continue
            stock_quote = stock_quote.append(tmp_quote)  # add new downloaded quotes into the existing ones
            stock_quote = stock_quote.drop_duplicates('code', keep='first')

            miss_code = real_buy_list.loc[~real_buy_list['code'].isin(stock_quote['code'])]  # check again if there still missing codes
            if not miss_code.empty:
                print('missing quotes for opening positions:', miss_code['code'].values)
                time.sleep(10)  # try again in 10 secs

        # calculate today's rise stop prices for all stocks
        open_quotes = stock_quote.loc[stock_quote['code'].isin(real_buy_list['code'].values)]  # columns: ['code', 'open']
        rise_fall_stop_flag = getTushareStockRiseFallStop(tushareToken, yesterday, stopLimit, open_quotes)
        rise_stop_codes = rise_fall_stop_flag.loc[rise_fall_stop_flag['rise_stop'], 'code'].unique()  # cannot buy stocks reaching rise limit

        # screen for rise stop
        real_buy_list = real_buy_list.loc[~real_buy_list['code'].isin(rise_stop_codes)]

        # screen for stock already in current position
        real_buy_list = real_buy_list.loc[~real_buy_list['code'].isin(new_position['code'].tolist())]

        # record new positions
        if not real_buy_list.empty:
            # select stocks with highest predicting probability
            real_buy_list = real_buy_list.sort_values(probaLabelBuy, ascending=False)  # sort by probabilities
            real_buy_list = real_buy_list.iloc[:(maxPosNum - new_position.shape[0])]   # choosing maximum number of stocks to buy
            real_buy_list = real_buy_list.merge(open_quotes, on='code', how='left')
            real_buy_list = real_buy_list.rename({'open': 'open_price'}, axis=1) # columns: ['code', 'Y_2D', 'Y_5D', 'Y_10D', 'Y_20D', 'open_price']
            real_buy_list.loc[:, 'current_price'] = real_buy_list['open_price']

            # get available fund to buy stocks
            tmp_available_funds, tmp_last_record_id = getCurrentAvaliableFunds(sql_engine, strategyTradeRecordTableName, his_id_col_name, strategyInitialFund)
            tmp_cur_pos_cost = new_position['cost'].sum() + new_position['commission'].sum()
            tmp_fund_each = (tmp_available_funds - tmp_cur_pos_cost) / float(real_buy_list.shape[0])  # available fund for each stock
            real_buy_list.loc[:, 'avaliable_fund'] = tmp_fund_each

            # get volume & amount for each stock
            real_buy_list.loc[:, 'volume'] = real_buy_list['avaliable_fund'] / (real_buy_list['open_price'] * (1 + buy_commission_rate))  # compute how many stocks could it buy (taking commission into account)
            real_buy_list.loc[:, 'volume'] = (real_buy_list['volume'] / 100).astype('int') * 100  # floor to 100
            real_buy_list.loc[:, 'cost'] = real_buy_list['open_price'] * real_buy_list['volume']
            real_buy_list.loc[:, 'commission'] = real_buy_list['cost'] * buy_commission_rate

            # set other details for buy orders
            real_buy_list.loc[:, 'start_date'] = today
            real_buy_list.loc[:, 'holding_period'] = 0
            real_buy_list.loc[:, 'floating_PnL'] = -real_buy_list['commission']
            real_buy_list.loc[:, 'time_stamp'] = datetime.now()

            new_position.loc[:, 'time_stamp'] = datetime.now()

            # reorder columns
            real_buy_list = real_buy_list[cur_pos_table_cols]

            # add buys to action buffer
            tmp_df = real_buy_list.copy()
            tmp_df = tmp_df.rename({'start_date':'date', 'open_price':'execute_price', 'cost': 'amount'}, axis=1)
            tmp_df.loc[:, 'action'] = 'buy'
            tmp_df = tmp_df[trd_act_table_cols]
            strategy_action = strategy_action.append(tmp_df)

            # add content to email (open position)
            mailContent['content'] += "Buy stocks at open (%s):\n" % today
            tmp_buy_details = real_buy_list.apply(lambda x: '%s @ %.2f' % (x.code, x.open_price), axis=1).tolist()
            tmp_buy_details = '\n'.join(tmp_buy_details)
            mailContent['content'] += tmp_buy_details
            mailContent['content'] += '\n\n'

            # before cover current position table, write old position to backup table
            current_position.loc[:, 'date'] = today
            tmp_conn = sql_engine.connect()
            current_position.to_sql(strategyPositionBackupTableName, tmp_conn, index=False, if_exists='append') # backup old records to the position backup table

            # write new positions to current position table (schema: quant)
            new_position = new_position.append(real_buy_list)
            # tmp_conn = sql_engine.connect()
            new_position.to_sql(strategyPositionTableName, tmp_conn, index=False, if_exists='replace') # rewrite new records to the position table
            tmp_conn.close()

            # # write new positions to current position table (schema: stockshow)
            # tmp_conn = sql_engine_stock_app.connect()
            # sql_statement = "truncate table `%s`" % strategyPositionTableName  # delete data but remain the table structure unchanged
            # tmp_result = tmp_conn.execute(sql_statement)
            # print("truncate all data in %s (schema: stockshow)" % strategyPositionTableName)
            #
            # new_position = new_position[cur_pos_table_cols]  # rearrange data
            # new_position.loc[:, 'id'] = list(range(1, new_position.shape[0] + 1))  # primary key with auto increment starts with 1
            # new_position.to_sql(strategyPositionTableName, tmp_conn, index=False, if_exists='append')
            # print("write new data to %s (schema: stockshow)" % strategyPositionTableName)
            # tmp_conn.close()

    # === add more information to emails
    # if no trade today
    if mailContent['content'] == '':
        mailContent['content'] = 'No trade today. Keep current positions.\n\n\n'

    # add current position to email content
    mailContent['content'] += 'current positions:\n'
    tmp_pos_details = '\n'.join(new_position['code'].tolist())
    mailContent['content'] += tmp_pos_details
    mailContent['content'] += '\n\n'

    # add yesterday's p&l
    tmp_conn = sql_engine.connect()
    tmp_latest_statistics = loadLatestDailyReturn(tmp_conn, strategyPerformanceTableName, indexQuoteTableName)
    tmp_conn.close()
    mailContent['content'] += 'latest trade day(%s): \ntotal return: %.2f%%\ndaily return: %.2f%%\ntotal alpha return (HS300): %.2f%%\ndaily alpha return (HS300): %.2f%%\n\n' % (
        tmp_latest_statistics['date'], tmp_latest_statistics['tot_ret'] * 100, tmp_latest_statistics['daily_ret'] * 100, tmp_latest_statistics['tot_alpha_ret'] * 100, tmp_latest_statistics['daily_alpha_ret'] * 100
    )

    # === send emails
    for tmp_config in adminMailConfig:
        sendEmail(mailContent, tmp_config)  # send email to different subscribers

    for tmp_config in officalMailConfig:
        sendEmail(mailContent, tmp_config)  # send email to official email address

    # ==== record strategy trade actions (schema: stockshow)
    if not strategy_action.empty:
        tmp_conn = sql_engine.connect()
        strategy_action.to_sql(strategyActionTableName, tmp_conn, index=False, if_exists='append')
        tmp_conn.close()


def calculateDailyPnL():
    # setting
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    strategyPositionTableName = 'NAIVE_LGBM_STRATEGY_POS'  # table recording current positions
    tushareToken = 'e37ce8e806bbfc9bfcb9ac35e68998c1710e7f0714e8fb5f257cd13c'
    cur_pos_table_cols = ['code', 'start_date', 'holding_period', 'open_price', 'current_price', 'volume', 'cost',
                          'commission', 'floating_PnL', 'time_stamp']

    # load latest trade day
    trade_calendar = getTradeCalendar(sql_engine)
    today, yesterday = getLatestTradeDay(trade_calendar)
    if today == 'non_trade_day':  # today is not trade day
        print('today is not trade day')
        return

    current_position = loadCurrentPosition(strategyPositionTableName, sql_engine)  # load current position

    # update current position's holding period
    current_position.loc[:, 'holding_period'] = current_position['start_date'].apply(
        lambda x: trade_calendar[(trade_calendar >= x) & (trade_calendar <= today)].size)  # after close, so count today as well

    # update current position's PnL
    stock_quote = getTushareProStockQuote(today, tushareToken)
    if stock_quote.empty:
        print('tushare get stock quote error')
        raise ValueError
    current_position = current_position.merge(stock_quote[['code', 'close']], on='code', how='left')
    current_position.loc[:, 'current_price'] = current_position['close']
    current_position.loc[:, 'floating_PnL'] = (current_position['current_price'] - current_position['open_price']) * current_position['volume'] - current_position['commission']
    current_position.loc[:, 'time_stamp'] = datetime.now()

    # rearrange columns
    current_position = current_position[cur_pos_table_cols]

    # update current position (holding periods)
    tmp_conn = sql_engine.connect()
    if not current_position.empty:
        sql_statement = 'truncate table %s' % strategyPositionTableName
        tmp_conn.execute(sql_statement)  # clear up the data in current position table
        print(sql_statement)
        current_position.to_sql(strategyPositionTableName, tmp_conn, index=False, if_exists='append')
        print('current position table updated!!')

    tmp_conn.close()

def evaluateStrategyPerformance():
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    config_stock_app = ConfigQuant.copy()
    config_stock_app['db'] = StockAppSchema
    sql_engine_stock_app = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**config_stock_app))  # different schema
    strategyPositionTableName = 'NAIVE_LGBM_STRATEGY_POS'  # table recording current positions
    strategyTradeRecordTableName = 'NAIVE_LGBM_STRATEGY_HIS_RECORDS'  # table recording historical trading records
    strategyPerformanceTableName = 'NAIVE_LGBM_STRATEGY_PERFORMANCE'
    strategyInitialFund = 1000000
    tradedayNumYear = 252
    strategyName = 'NAIVE_LGBM'
    his_id_col_name = 'record_id'

    # load latest trade day
    trade_calendar = getTradeCalendar(sql_engine)
    today, yesterday = getLatestTradeDay(trade_calendar)
    if today == 'non_trade_day':  # today is not trade day
        print('today is not trade day')
        return

    strategy_performance_cols = ['date', 'strategy', 'market_value', 'cash', 'daily_return', 'annualized_return',
                                 'annualized_std', 'max_drawndown', 'duration', 'time_stamp']

    available_funds, tmp_last_record_id = getCurrentAvaliableFunds(sql_engine, strategyTradeRecordTableName, his_id_col_name,
                                                                       strategyInitialFund)

    # load strategy's current position
    current_position = loadCurrentPosition(strategyPositionTableName, sql_engine)
    if not current_position.empty:
        stock_market_pnl = current_position['floating_PnL'].sum()  # sum of all stocks current PnL
        current_pos_cost = current_position['cost'].sum()  # the buying cost for the holding stocks
        current_pos_commission = current_position['commission'].sum()
    else:
        stock_market_pnl = 0
        current_pos_cost = 0
        current_pos_commission = 0

    # load strategy's past performance
    strategy_performance = loadCurrentPosition(strategyPerformanceTableName, sql_engine)
    if not strategy_performance.empty:
        current_duration = strategy_performance.shape[0] + 1   # strategy's runtime
        previous_mrk_val = strategy_performance['market_value'].iloc[-1]
        daily_rets = strategy_performance['daily_return'].tolist()
        previous_peak_market_value = strategy_performance['market_value'].max()
        previous_max_drawdown = strategy_performance['max_drawndown'].max()
    else:
        current_duration = 1   # the first day after strategy begun
        previous_mrk_val = strategyInitialFund
        daily_rets = []
        previous_peak_market_value = strategyInitialFund
        previous_max_drawdown = 0

    # calculate newest performance
    market_value = available_funds + stock_market_pnl   # cash + stcok values
    cash = available_funds - current_pos_cost - current_pos_commission
    today_ret = market_value / previous_mrk_val - 1
    daily_rets.append(today_ret)  # add today's return into daily returns series
    annualized_ret = np.power(np.product(np.array(daily_rets) + 1), tradedayNumYear / current_duration) - 1
    annualized_std = np.std(daily_rets, ddof=1) * np.sqrt(tradedayNumYear)
    current_peak_market_value = max([previous_peak_market_value, market_value])
    current_drawndown = 1 - market_value / current_peak_market_value
    current_max_drawndown = max([previous_max_drawdown, current_drawndown])

    new_performance = pd.DataFrame({'date': today, 'strategy':strategyName, 'market_value':market_value, 'cash':cash,
                                    'daily_return':today_ret, 'annualized_return':annualized_ret, 'annualized_std':annualized_std, 'max_drawndown': current_max_drawndown,
                                    'duration':current_duration, 'time_stamp':datetime.now()},index=[0])

    # rearrange columns
    new_performance = new_performance[strategy_performance_cols]

    # write new record to db (quant)
    # tmp_conn =sql_engine.connect()
    # new_performance.to_sql(strategyPerformanceTableName, tmp_conn, index=False, if_exists='append')
    # tmp_conn.close()
    writeDB(sql_engine, ConfigQuant['db'], strategyPerformanceTableName, new_performance, add_primary_key=True)

    # write new record to db (stock app)
    # writeDB(sql_engine_stock_app, config_stock_app['db'], strategyPerformanceTableName, new_performance, add_primary_key=True)

def historicalPredictionPerformance():
    # setting
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    # config_stock_app = ConfigQuant.copy()
    # config_stock_app['db'] = StockAppSchema
    # sql_engine_stock_app = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(
    #     **config_stock_app))  # different schema
    hisRecomRecordTableName = 'NAIVE_LGBM_RECOMMENDATION_HIS_RECORDS'  # table recording historical trading records
    hisRecomEvalTableName = 'NAIVE_LGBM_HIS_RECOMM_EVAL'

    tushareToken = 'e37ce8e806bbfc9bfcb9ac35e68998c1710e7f0714e8fb5f257cd13c'
    evalLength = 20
    probaLabelBuy = 'Y_2D'  # (2D + 5D + 10D + 20D) / 4

    trade_calendar = getTradeCalendar(sql_engine)
    today, yesterday = getLatestTradeDay(trade_calendar)

    if today == 'non_trade_day':
        print('today is not tradeday')
        return

    eval_dates = trade_calendar[trade_calendar <= today][-evalLength:]

    # get historical recommendation records
    sql_statement = "select `date`, `code`, `%s` from `%s` where `date` >= '%s'" % (probaLabelBuy, hisRecomRecordTableName, eval_dates[0])
    tmp_conn = sql_engine.connect()
    his_recommendation = pd.read_sql(sql_statement, tmp_conn)
    tmp_conn.close()

    # === get historical stock quotes from tushare
    stock_quotes = pd.DataFrame([])
    for tmp_date in eval_dates:
        tmp_quotes = getTushareProStockQuote(tmp_date, tushareToken)
        stock_quotes = stock_quotes.append(tmp_quotes)

    stock_quotes.loc[:, 'code'] = stock_quotes['ts_code'].apply(lambda x: x[:-3])  # change stock code format
    stock_quotes.loc[:, 'date'] = stock_quotes['trade_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))  # change date format

    # === get stock returns from each date to today
    current_price = stock_quotes.loc[stock_quotes['date'] == today, ['code', 'close']]
    current_price = current_price.rename({'close': 'current'}, axis=1)

    stock_quotes = stock_quotes.merge(current_price, on='code', how='left')  # if suspensed, then return is nan
    stock_quotes.loc[:, 'rets'] = stock_quotes.apply(lambda x: x.current / x.open - 1, axis=1)  # calculate returns from each day to today
    # stock_quotes = stock_quotes.loc[~stock_quotes['rets'].isnull()]

    stock_quotes = stock_quotes[['date', 'code', 'open', 'current', 'rets']]

    # === get historical index quotes from tushare (HS300, CSI500)
    ts_api = ts.pro_api(tushareToken)
    tmp_start_date = eval_dates[0].replace('-', '')
    tmp_end_date = eval_dates[-1].replace('-', '')
    hs300_quotes = ts_api.index_daily(ts_code='399300.SZ', start_date=tmp_start_date, end_date=tmp_end_date)
    csi500_quotes = ts_api.index_daily(ts_code='399905.SZ', start_date=tmp_start_date, end_date=tmp_end_date)

    hs300_quotes.loc[:, 'date'] = hs300_quotes['trade_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
    csi500_quotes.loc[:, 'date'] = csi500_quotes['trade_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))

    # === get index returns from each date to today
    hs300_quotes.loc[:, 'current'] = hs300_quotes.loc[hs300_quotes['date'] == eval_dates[-1], 'close'].iloc[0]
    hs300_quotes.loc[:, 'rets'] = hs300_quotes.apply(lambda x: x.current / x.open -1, axis=1)
    csi500_quotes.loc[:, 'current'] = csi500_quotes.loc[csi500_quotes['date'] == eval_dates[-1], 'close'].iloc[0]
    csi500_quotes.loc[:, 'rets'] = csi500_quotes.apply(lambda x: x.current / x.open - 1, axis=1)
    index_quotes = hs300_quotes[['date', 'rets']].merge(csi500_quotes[['date', 'rets']], on='date', how='inner', suffixes=['_hs300', '_csi500'])

    # === calculate relative returns (compared with index)
    stock_quotes = stock_quotes.merge(index_quotes, on='date', how='left')
    stock_quotes.loc[:, 'alpha_hs300'] = stock_quotes['rets'] - stock_quotes['rets_hs300']
    stock_quotes.loc[:, 'alpha_csi500'] = stock_quotes['rets'] - stock_quotes['rets_csi500']

    stock_quotes.loc[:, 'time_stamp'] = datetime.now()

    # get absolute returns and relative returns of past recommended stocks
    his_recomm_performance = his_recommendation.merge(stock_quotes[['date', 'code', 'rets', 'alpha_hs300', 'alpha_csi500']], on=['date', 'code'], how='left')

    his_recomm_performance = his_recomm_performance.rename({probaLabelBuy: 'probability_buy'}, axis=1)

    # # rank returns day by day, and divide into n levels based on daily ranking of returns
    # stock_quotes.loc[:, 'rank'] = stock_quotes.groupby('date')['rets'].rank(method='average', ascending=False)
    # tmp_quantiles = list(map(lambda x: x*0.1, range(totalLevelNum)))
    # tmp_quantile_values = stock_quotes['rank'].quantile(tmp_quantiles).tolist()  # get threshold value for each quantile
    # stock_quotes.loc[:, 'level'] = 0
    #
    # for tmp_num in range(len(tmp_quantile_values) - 1):
    #     stock_quotes.loc[(stock_quotes['rank'] >= tmp_quantile_values[tmp_num]) & (stock_quotes['rank'] < tmp_quantile_values[tmp_num + 1]), 'level'] = tmp_num + 1
    # stock_quotes.loc[stock_quotes['rank'] >= tmp_quantile_values[-1], 'level'] = len(tmp_quantile_values)
    #
    # stock_quotes = stock_quotes[['date', 'code', 'open', 'current', 'rets', 'rank', 'level']]
    #
    # # get historical recommendation's level
    # his_recommendation = his_recommendation.merge(stock_quotes, on='code', how='inner')

    # write db
    tmp_conn = sql_engine.connect()
    his_recomm_performance.to_sql(hisRecomEvalTableName, tmp_conn, index=False, if_exists='replace')
    tmp_conn.close()



def sendEmail(mail_content, mail_config):
    if mail_content['content'] != '':
        try:
            # smtpObj = smtplib.SMTP()  # send email
            # smtpObj.connect(mail_config['smtp_server'], mail_config['smtp_port'])
            smtpObj = smtplib.SMTP_SSL()
            smtpObj.connect(mail_config['smtp_server'])
            smtpObj.login(mail_config['user'], mail_config['password'])

            # construct message
            message = MIMEText(mail_content['content'], 'plain', 'utf-8')
            message['From'] = Header(mail_content['from'], 'utf-8')
            message['To'] = Header(mail_content['to'], 'utf-8')
            message['Subject'] = Header(mail_content['subject'], 'utf-8')

            # send E-mail
            smtpObj.sendmail(mail_config['sender'], mail_config['receiver'], message.as_string())
        except smtplib.SMTPException:
            print('send email to %s error' % mail_config['receiver'])

def restoreCurrentPosition(restore_date):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    backupPosTableName = 'NAIVE_LGBM_STRATEGY_POS_BACKUP'
    posTableName = 'NAIVE_LGBM_STRATEGY_POS'
    tablesToClean = ['NAIVE_LGBM_RECOMMENDATION_HIS_RECORDS', 'NAIVE_LGBM_STRATEGY_HIS_RECORDS']
    dateField = ['date', 'end_date']
    cur_pos_table_cols = ['code', 'start_date', 'holding_period', 'open_price', 'current_price', 'volume', 'cost',
                          'commission',
                          'floating_PnL', 'time_stamp']

    # restore current position table
    sql_conn = sql_engine.connect()
    sql_statement = 'select * from `%s`' % backupPosTableName
    backup_position = pd.read_sql(sql_statement, sql_conn)

    backup_position.loc[:, 'time_stamp'] = datetime.now()

    backup_position = backup_position[cur_pos_table_cols]

    backup_position.to_sql(posTableName, sql_conn, index=False, if_exists='replace')

    # delete historical records that exceed restore date
    for tmp_table_name, tmp_date_field in zip(tablesToClean, dateField):
        sql_statement = "delete from `%s` where `%s` > '%s'" % (tmp_table_name, tmp_date_field, restore_date)
        sql_conn.execute(sql_statement)
        print('delete records from table `%s`' % tmp_table_name)

def restore_backup_position():
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    backup_table_name = 'NAIVE_LGBM_STRATEGY_POS_BACKUP'
    current_table_name = 'NAIVE_LGBM_STRATEGY_POS'

    restore_date = '2019-01-24'

    sql_conn = sql_engine.connect()
    sql_statement = "select * from `%s` where `date` = '%s'" % (backup_table_name, restore_date)
    backup_pos = pd.read_sql(sql_statement, sql_conn)

    backup_pos = backup_pos.drop('date', axis=1)
    backup_pos.loc[:, 'time_stamp'] = datetime.now()

    sql_statement = 'truncate table `%s`' % current_table_name
    sql_conn.execute(sql_statement)
    print(sql_statement)

    backup_pos.to_sql(current_table_name, sql_conn, index=False, if_exists='append')



def morningTask():
    strategyBasicPreTrade()
    print('process pre trade successful!')
    strategyBasicActualTrade()
    print('process actual trade successful!')

def afternoonTask():
    calculateDailyPnL()
    print('get daily PnL successful!')
    evaluateStrategyPerformance()
    print('get strategy performance successful!')
    historicalPredictionPerformance()
    print('get prediciton evaluation successful!')


if __name__ == '__main__':
    # strategyBasicPreTrade()
    strategyBasicActualTrade()
    # while True:
    #     try:
    #         strategyBasicActualTrade()
    #         break
    #     except timeout as e:
    #         print(e)
    #         sleep(60)  # retry after 1 min
    # calculateDailyPnL()
    # evaluateStrategyPerformance()
    # historicalPredictionPerformance()

    # restoreCurrentPosition('2018-12-13')

    # morningTask()
    # afternoonTask()
    # restore_backup_position()