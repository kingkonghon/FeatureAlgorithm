import tushare as ts
from datetime import datetime
from sqlalchemy import create_engine
import os
import sys
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

class TushareQuotes:
    def __init__(self):
        ts_token = 'e37ce8e806bbfc9bfcb9ac35e68998c1710e7f0714e8fb5f257cd13c'
        self.ts_api = ts.pro_api(ts_token)
        self.tableCalendarName = 'TRADE_CALENDAR'
        self.tableTushareStockIndexName = 'STOCK_INDEX_QUOTE_TUSHARE'
        self.tableTushareStockQuoteName = 'STOCK_QUOTE_TUSHARE'
        self.tableTushareStockAdjQuoteName = 'STOCK_FOR_ADJ_QUOTE_TUSHARE'
        self.tableTushareStockBasicName = 'STOCK_BASIC_TUSHARE'
        self.sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
        self.sql_conn = None
        self.trade_calendar = None

        self.stock_index_codes = ['399300.SZ', '399905.SZ', '000011.SH']
        self.stock_basic_dict = {'turnover_rate_f':'turnover_rate_freeflow', 'total_mv':'TOT_MRK_CAP', 'circ_mv':'FREE_MRK_CAP'}

    def getDailyStockQuote(self, today):
        # download stock index code from tushare, and update to db
        latest_record_date_unadj = self.getRecordsLatestDate(self.tableTushareStockQuoteName, self.sql_conn)
        latest_record_date_adj = self.getRecordsLatestDate(self.tableTushareStockAdjQuoteName, self.sql_conn)
        if latest_record_date_unadj != latest_record_date_adj:
            print('latest date not match')
            if latest_record_date_unadj > latest_record_date_adj:
                tmp_table_to_prune = self.tableTushareStockQuoteName
                latest_record_date = latest_record_date_adj
            else:
                tmp_table_to_prune = self.tableTushareStockAdjQuoteName
                latest_record_date = latest_record_date_unadj
            self.deleteDBData(tmp_table_to_prune, latest_record_date)
        else:
            latest_record_date = latest_record_date_adj

        missing_dates = self.trade_calendar[(self.trade_calendar > latest_record_date) & (self.trade_calendar <= today)]

        stock_quotes = pd.DataFrame([])
        stock_adj_quotes = pd.DataFrame([])
        for tmp_date in missing_dates:
            tmp_date = tmp_date.replace('-', '')
            tmp_quotes = self.ts_api.daily(trade_date=tmp_date)
            tmp_adj_factor = self.ts_api.adj_factor(ts_code='', trade_date=tmp_date) # adjust factor
            tmp_quotes = tmp_quotes.merge(tmp_adj_factor, on=['ts_code', 'trade_date'], how='inner')
            stock_quotes = stock_quotes.append(tmp_quotes)  # get stock data from tushare date by date

            tmp_adj_quotes = tmp_quotes.copy()
            for tmp_col_name in ['open', 'high', 'low', 'close', 'amount']:
                tmp_adj_quotes.loc[:, tmp_col_name] = tmp_adj_quotes[tmp_col_name] * tmp_adj_quotes['adj_factor']
            tmp_adj_quotes = tmp_adj_quotes[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
            stock_adj_quotes = stock_adj_quotes.append(tmp_adj_quotes)

        stock_quotes = self.convertDateAndCode(stock_quotes)
        stock_adj_quotes = self.convertDateAndCode(stock_adj_quotes)

        self.writeDB(stock_quotes, self.tableTushareStockQuoteName)
        self.writeDB(stock_adj_quotes, self.tableTushareStockAdjQuoteName)

    def getDailyStockIndexQuote(self, today):
        latest_record_date = self.getRecordsLatestDate(self.tableTushareStockIndexName, self.sql_conn)

        new_record_start_date = self.trade_calendar[self.trade_calendar > latest_record_date].iloc[0]

        ts_start_date = new_record_start_date.replace('-', '')  # convert date format, e.g. from 2018-01-01 to 20180101
        ts_end_date = today.replace('-', '')

        index_quotes = pd.DataFrame([])
        for tmp_code in self.stock_index_codes:  # get index data from tushare code by code
            tmp_quotes = self.ts_api.index_daily(ts_code=tmp_code, start_date=ts_start_date, end_date=ts_end_date)
            index_quotes = index_quotes.append(tmp_quotes)

        index_quotes = self.convertDateAndCode(index_quotes)

        index_quotes = index_quotes.rename({'pct_chg': 'pct_change'}, axis=1)

        tmp_cols = index_quotes.columns.tolist()
        tmp_cols.remove('date')
        tmp_cols.remove('code')
        tmp_cols.insert(0, 'code')
        tmp_cols.insert(0, 'date')
        index_quotes = index_quotes[tmp_cols]

        if not index_quotes.empty:
            index_quotes.loc[:, 'time_stamp'] = datetime.now()

            self.writeDB(index_quotes, self.tableTushareStockIndexName)  # write table to index table

    def getDailyStockBasicInfo(self, today):
        sql_statement = "select max(date) from `%s`" % self.tableTushareStockBasicName
        last_record_date = pd.read_sql(sql_statement, self.sql_conn)
        last_record_date = last_record_date.iloc[0, 0]

        # last_record_date = '2018-05-15'
        # today = '2018-05-16'

        missing_dates = self.trade_calendar[(self.trade_calendar > last_record_date) & (self.trade_calendar <= today)]
        missing_dates = list(map(lambda x: x.replace('-', ''), missing_dates))

        stock_basic_info = pd.DataFrame([])
        for ts_date in missing_dates:
            tmp_data = self.ts_api.daily_basic(trade_date=ts_date)
            stock_basic_info = stock_basic_info.append(tmp_data)

        if not stock_basic_info.empty:
            stock_basic_info = self.convertDateAndCode(stock_basic_info)
            stock_basic_info = stock_basic_info.rename(self.stock_basic_dict, axis=1)

            stock_basic_info.loc[:, 'time_stamp'] = datetime.now()

            self.writeDB(stock_basic_info, self.tableTushareStockBasicName)  # write table to index table

    def convertDateAndCode(self, quotes):
        #  convert the format of code and date back to my format
        quotes.loc[:, 'date'] = quotes['trade_date'].apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))
        quotes.loc[:, 'code'] = quotes['ts_code'].apply(lambda x: x[:-3])

        quotes = quotes.drop(['ts_code', 'trade_date'], axis=1)

        tmp_cols = quotes.columns.tolist()
        tmp_cols.remove('date')
        tmp_cols.remove('code')
        tmp_cols.insert(0, 'date')
        tmp_cols.insert(1, 'code')
        quotes = quotes[tmp_cols]  # rearrange the order of the columns

        return quotes

    def getRecordsLatestDate(self, table_name, sql_conn):
        # get the date fo the newest records
        sql_statement = "select max(date) from `%s`" % table_name

        latest_record_date = pd.read_sql(sql_statement, sql_conn)
        latest_record_date = latest_record_date.iloc[0, 0]

        return latest_record_date

    def writeDB(self, data, table_name):
        # write data to db
        date = data['date'].unique()
        start_date = date.min()
        end_date = date.max()

        sql_statement = "select count(1) as num from `%s` where `date` between '%s' and '%s'" % (table_name, start_date, end_date)
        old_record_num = pd.read_sql(sql_statement, self.sql_conn)
        old_record_num = old_record_num.iloc[0, 0]

        if old_record_num > 0:  # old records overlap with new records, delete old data first
            sql_statement = "delete from `%s` where `date` between '%s' and '%s'" % (table_name, start_date, end_date)
            self.sql_conn.execute(sql_statement)

        data.to_sql(table_name, self.sql_conn, index=False, if_exists='append')

    def deleteDBData(self, table_name, start_date):
        sql_statement = "delete from `%s` where date > '%s'" % (table_name, start_date)
        self.sql_conn.execute(sql_statement)
        print('delete data from `%s` where date > %s' % (table_name, start_date))

    def run(self):
        self.sql_conn = self.sql_engine.connect()

        # get today's date
        today = datetime.now()
        today = datetime.strftime(today, '%Y-%m-%d')

        sql_statement = "select date from `%s`" % self.tableCalendarName
        self.trade_calendar = pd.read_sql(sql_statement, self.sql_conn)
        self.trade_calendar = self.trade_calendar['date']

        if today not in self.trade_calendar.tolist():
            print("today is not trade date")
        else:  # today is trade, update data
            self.getDailyStockIndexQuote(today)

            self.getDailyStockQuote(today)



        self.sql_conn.close()

    def getAllDailyStockQuote(self):
        self.sql_conn = self.sql_engine.connect()

        today = datetime.now()
        today = datetime.strftime(today, '%Y-%m-%d')

        sql_statement = "select date from `%s`" % self.tableCalendarName
        trade_calendar = pd.read_sql(sql_statement, self.sql_conn)
        trade_calendar = trade_calendar['date']
        trade_calendar = trade_calendar[trade_calendar <= today]

        trade_calendar = list(map(lambda x: x.replace('-', ''), trade_calendar))

        block_count = 0
        block_size = 20
        block_quotes = pd.DataFrame([])
        block_adj_quotes = pd.DataFrame([])
        for tmp_date in trade_calendar:
            # original quotes
            tmp_quotes = self.ts_api.daily(trade_date=tmp_date)
            tmp_adj_factor = self.ts_api.adj_factor(ts_code='', trade_date=tmp_date)
            tmp_quotes = tmp_quotes.merge(tmp_adj_factor, on=['ts_code', 'trade_date'], how='inner')
            block_quotes = block_quotes.append(tmp_quotes)

            # forward adjusted quotes
            tmp_adj_quotes = tmp_quotes.copy()
            for tmp_col in ['open', 'high', 'low', 'close', 'amount']:
                tmp_adj_quotes.loc[:, tmp_col] = tmp_adj_quotes[tmp_col] * tmp_adj_quotes['adj_factor']
            tmp_adj_quotes = tmp_adj_quotes.drop(['pre_close', 'change', 'pct_chg', 'adj_factor'], axis=1)
            block_adj_quotes = block_adj_quotes.append(tmp_adj_quotes)

            block_count += 1
            if block_count == block_size:  # dump data to db every N round
                block_quotes = block_quotes.reset_index()
                block_quotes = block_quotes.drop('index', axis=1)
                block_quotes = self.convertDateAndCode(block_quotes)

                block_adj_quotes = block_adj_quotes.reset_index()
                block_adj_quotes = block_adj_quotes.drop('index', axis=1)
                block_adj_quotes = self.convertDateAndCode(block_adj_quotes)
                try:
                    block_quotes.to_sql(self.tableTushareStockQuoteName, self.sql_conn, index=False, if_exists='append')
                    print('finish dumped %d original quotes into db (%s - %s)' % (block_quotes.shape[0], block_quotes['date'].min(), block_quotes['date'].max()))
                    block_adj_quotes.to_sql(self.tableTushareStockAdjQuoteName, self.sql_conn, index=False, if_exists='append')
                    print('finish dumped %d adj quotes into db (%s - %s)' % (block_adj_quotes.shape[0], block_adj_quotes['date'].min(), block_adj_quotes['date'].max()))
                except Exception as e:
                    print('lost connection, retry...')
                    self.sql_conn = self.sql_engine.connect() # in case lost connection
                    block_quotes.to_sql(self.tableTushareStockQuoteName, self.sql_conn, index=False, if_exists='append')
                    print('finish dumped %d original quotes into db (%s - %s)' % (block_quotes.shape[0], block_quotes['date'].min(), block_quotes['date'].max()))
                    block_adj_quotes.to_sql(self.tableTushareStockAdjQuoteName, self.sql_conn, index=False,if_exists='append')
                    print('finish dumped %d adj quotes into db (%s - %s)' % (block_adj_quotes.shape[0], block_adj_quotes['date'].min(), block_adj_quotes['date'].max()))

                block_count = 0
                block_quotes = pd.DataFrame([])
                block_adj_quotes = pd.DataFrame([]) # reset buffer and counter

def airflowCallable():
    tq = TushareQuotes()
    tq.run()


if __name__ == '__main__':
    tq = TushareQuotes()
    tq.run()
    # tq.getAllDailyStockQuote()