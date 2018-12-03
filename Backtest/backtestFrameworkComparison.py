import h5py
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

stockQuoteTableName = 'STOCK_FORWARD_ADJ_QUOTE'
stockQuotePriceField = 'open'
stockQuoteHighField = 'high'
calendarTableName = 'TRADE_CALENDAR'

stockQuoteFileName ={
    'open': 'LZ_CN_STKA_QUOTE_TOPEN.h5',
    'high': 'LZ_CN_STKA_QUOTE_THIGH.h5',
    'tradable_flag': 'LZ_CN_STKA_SLCIND_STOP_FLAG.h5',
    'adj_factor': 'LZ_CN_STKA_CMFTR_CUM_FACTOR.h5',
    'listday_count': 'LZ_CN_STKA_SLCIND_TRADEDAYCOUNT.h5',
    'not_st_flag': 'LZ_CN_STKA_SLCIND_ST_FLAG.h5',
    'index_component_flag': 'LZ_CN_STKA_INDEX_HS300WEIGHT.h5'
}

indexQuoteFileName = {
    'open': 'LZ_CN_STKA_INDXQUOTE_OPEN.h5'
}
indexCode = '000300'

def getStockQuoteFromDB(sql_config, start_date, end_date, price_field, high_field):
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    quote_field = ','.join([stockQuotePriceField, stockQuoteHighField])
    sql_statement = "select date, `code`, %s from %s where (date >= '%s') and (date <= '%s')" % (quote_field,
                                                                                                 stockQuoteTableName,
                                                                                                 start_date,
                                                                                                 end_date)
    stock_quotes = pd.read_sql(sql_statement, quant_engine)
    stock_price = stock_quotes.pivot_table(values=price_field, index='date', columns='code')
    stock_high = stock_quotes.pivot_table(values=high_field, index='date', columns='code')
    # reorder index & columns
    trade_dates = stock_price.index.sort_values()  # order by date
    stock_price = stock_price.loc[trade_dates]
    stock_high = stock_high.loc[trade_dates]
    stock_high = stock_high[stock_price.columns]  # the same order in columns

    return trade_dates, stock_price, stock_high

def getStockQuoteFromFile(stock_quote_file_name, start_date, end_date):
    start_date_num = int(''.join([start_date[:4], start_date[5:7], start_date[8:]]))
    end_date_num = int(''.join([end_date[:4], end_date[5:7], end_date[8:]]))

    stock_quote = {}
    for quote_name, file_name in stock_quote_file_name.items():
        tmp_h5_file = h5py.File(file_name, 'r')

        stock_quote[quote_name] = tmp_h5_file['data'][...]
        tmp_trade_dates = tmp_h5_file['date'][...]
        tmp_idx = (tmp_trade_dates >= start_date_num) & (tmp_trade_dates <= end_date_num)
        stock_quote[quote_name] = stock_quote[quote_name][tmp_idx]

        if quote_name == 'open':
            # trade_dates = tmp_h5_file['date'][...]
            trade_dates = tmp_trade_dates[tmp_idx]
            codes = tmp_h5_file['header'][...]

    codes = np.array(list(map(lambda x: x.decode('utf-8').split('.')[1], codes)))
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))

    # tmp_idx = (trade_dates >= start_date) & (trade_dates <= end_date)
    #
    # trade_dates = trade_dates[tmp_idx]
    # for quote_name, quote_data in stock_quote.items():
    #     stock_quote[quote_name] = stock_quote[quote_name][tmp_idx]

    stock_quote['tradable_flag'] = (stock_quote['tradable_flag'] == 0)
    stock_quote['not_st_flag'] = (stock_quote['not_st_flag'] == 0)
    stock_quote['index_component_flag'] = (stock_quote['index_component_flag'] > 0)

    # adj quote
    for quote_name in ['open', 'high']:
        stock_quote[quote_name] = stock_quote[quote_name] * stock_quote['adj_factor']

    return stock_quote, trade_dates, codes

def getIndexQuoteFromFile(index_quote_file_name, target_index_code, start_date, end_date):
    index_quote = {}
    for quote_name, file_name in index_quote_file_name.items():
        tmp_h5_file = h5py.File(file_name, 'r')

        index_quote[quote_name] = tmp_h5_file['data'][...]
        if quote_name == 'open':
            trade_dates = tmp_h5_file['date'][...]
            codes = tmp_h5_file['header'][...]

    codes = np.array(list(map(lambda x: x.decode('utf-8').split('.')[1], codes)))
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))

    tmp_idx_row = (trade_dates >= start_date) & (trade_dates <= end_date)
    tmp_idx_col = (codes == target_index_code)

    trade_dates = trade_dates[tmp_idx_row]
    for quote_name, quote_data in index_quote.items():
        index_quote[quote_name] = index_quote[quote_name][tmp_idx_row]
        index_quote[quote_name] = index_quote[quote_name][:, tmp_idx_col]
        index_quote[quote_name] = index_quote[quote_name].T[0]

    return index_quote


def getPredictionFromFile(file_path):
    prediction_data = pd.read_csv(file_path, dtype={'code': str}, index_col=None, header=0)

    # prediction_data = prediction_data.drop('Unnamed: 0', axis=1)  # drop the first column
    # prediction_data.loc[:, 'code'] = prediction_data['code'].apply(lambda x: 'null' if np.isnan(x) else str(int(x)))
    # prediction_data.loc[:, 'code'] = prediction_data['code'].apply(lambda x: '0'*(6-len(x)) + x if x != 'null' else x)
    return prediction_data

def calRet(price_t, price_t_1):
    # if price_t_1 >= 0:
    #     ret = price_t / price_t_1 - 1
    # else:
    #     ret = 1 -price_t / price_t_1
    ret = price_t / price_t_1 - 1
    return ret

class BacktestingClass:
    def __init__(self, **kwargs):
        self.start_date = kwargs.get("start_date")
        self.end_date = kwargs.get("end_date")
        self.strategy = kwargs.get('strategy')
        # self.prediction_file_path = kwargs.get('prediction_file_path')
        self.max_position_num = kwargs.get('max_position_num')
        # self.prediction_pairs = kwargs.get('prediction_pairs')
        self.prediction = []
        self.stock_price = []
        self.stock_high = []
        self.trade_dates = []
        self.codes = []
        self.tradable_flag = []
        self.index_component_flag = []
        self.not_st_flag = []
        self.listday_count = []
        self.rets = []
        self.cum_rets = []
        self.tradeday_num = 0
        self.today_idx = 1
        self.current_position = {}
        self.stock_holdings = []

    def init(self, method):
        # get prediction
        # self.prediction = getPredictionFromFile(self.prediction_file_path)

        if method == 'DB':
            # get stock quotes
            trade_dates, stock_price, stock_high = getStockQuoteFromDB(ConfigQuant, self.start_date, self.end_date,
                                                                       stockQuotePriceField, stockQuoteHighField)

            self.tradable_flag = (~stock_price.isnull()).values

            stock_price = stock_price.fillna(method='ffill')
            stock_high = stock_high.fillna(method='ffill')

            self.stock_price = stock_price.values
            self.stock_high = stock_high.values
            self.trade_dates = np.array(trade_dates)
            self.codes = np.array(stock_price.columns)
        elif method == 'File':
            stock_quote, trade_dates, codes = getStockQuoteFromFile(stockQuoteFileName, self.start_date, self.end_date)
            index_quote = getIndexQuoteFromFile(indexQuoteFileName, indexCode, self.start_date, self.end_date)

            self.stock_price = stock_quote['open']
            self.stock_high = stock_quote['high']
            self.tradable_flag = stock_quote['tradable_flag']
            self.not_st_flag = stock_quote['not_st_flag']
            self.listday_count = stock_quote['listday_count']
            self.index_component_flag = stock_quote['index_component_flag']
            self.trade_dates = trade_dates
            self.codes = codes

            self.index_price = index_quote['open']

        self.tradeday_num = self.trade_dates.size - 1

        self.rets = np.zeros(self.tradeday_num)
        self.cum_rets = np.ones(self.tradeday_num)

    def loadPrediction(self, file_path):
        self.prediction = getPredictionFromFile(file_path)

    def reinit(self):
        self.rets = np.zeros(self.tradeday_num)
        self.cum_rets = np.ones(self.tradeday_num)
        self.today_idx = 1
        self.current_position = {}
        self.stock_holdings = []


    def calTodayReturn(self):
        position_codes = list(self.current_position.keys())
        col_idx = np.in1d(self.codes, position_codes)
        # ret = self.stock_price[self.today_idx+1, col_idx] / self.stock_price[self.today_idx, col_idx] - 1  # next day's open - today's open
        ret = calRet(self.stock_price[self.today_idx+1, col_idx], self.stock_price[self.today_idx, col_idx])  # next day's open - today's open

        reorder_code = self.codes[col_idx]  # mapping weights to stock returns
        current_stock_weights = np.array(list(map(lambda x: self.current_position[x]['weight'], reorder_code)))  # get weight corresponding to new order of codes
        self.rets[self.today_idx] = np.sum(ret * current_stock_weights)

    def getToday(self):
        return self.trade_dates[self.today_idx]

    def getYesterday(self):
        return self.trade_dates[self.today_idx-1]

    def getPosition(self):
        return self.current_position

    def getTodayTradableStockCode(self):
        return self.codes[self.tradable_flag[self.today_idx]]

    def getTodayIndexComponentStockCode(self):
        return self.codes[self.index_component_flag[self.today_idx]]

    def getTodayNotSTStockCode(self):
        return self.codes[self.not_st_flag[self.today_idx]]

    def getTodayNotNewlyListStockCode(self, min_list_day_num):
        tmp_idx = (self.listday_count[self.today_idx] > min_list_day_num)
        return self.codes[tmp_idx]

    def getCumRets(self):
        cum_rets = np.cumprod(1 + self.rets)
        cum_rets = np.hstack([1, cum_rets])
        return cum_rets

    def maxDrawndown(self):
        series_cum_ret = pd.Series(self.cum_rets)
        expending_max_ret = series_cum_ret.expanding().max()
        drawn_down = expending_max_ret - series_cum_ret

        return drawn_down.max()

    def getAnalysis(self):
        anaylysis_result = {}

        anaylysis_result['avrYield'] = np.mean(self.rets) * 252
        anaylysis_result['std'] = np.std(self.rets) * np.sqrt(252)
        anaylysis_result['shapeRatio'] = anaylysis_result['avrYield'] / anaylysis_result['std']

        self.cum_rets = self.getCumRets()
        anaylysis_result['maxDrawndown'] = self.maxDrawndown()

        # self.cum_rets = np.cumprod(1 + self.rets)
        # self.cum_rets = np.hstack([1, self.cum_rets])
        # dt_trade_dates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), self.trade_dates))

        # index_rets = self.index_price[1:] / self.index_price[:-1] - 1
        # index_rets = np.hstack([0, index_rets])
        # index_cum_rets = np.cumprod(1 + index_rets)

        # plt.plot(dt_trade_dates, self.cum_rets)
        # plt.plot(dt_trade_dates, index_cum_rets)
        # plt.legend(['stategy', 'index'])
        # plt.show()
        #
        # # write stocking holding records
        # tmp_cols = ['date']
        # tmp_cols.extend(['code%d' % n for n in range(self.max_position_num)])
        # tmp_cols.append('return')
        # stock_holding = pd.DataFrame(self.stock_holdings, columns=tmp_cols)
        # stock_holding.to_csv(output_file_name)

        return anaylysis_result

    def run(self, params):
        while self.today_idx < self.tradeday_num:

            # tradable_position_codes = self.getTodayTradableStockCode()  # today's tradable
            # tradable_position_codes = list(tradable_position_codes[np.in1d(tradable_position_codes, list(self.current_position.keys()))])
            tradable_position_codes = list(self.current_position.keys())  # assume all position can be sold

            # close position reaching maximum holding period
            tmp_closed_position = []
            for tmp_code in tradable_position_codes:
                if self.current_position[tmp_code]['holding_length'] >= self.current_position[tmp_code]['max_holding_period']:
                    del self.current_position[tmp_code] # close position, ret already recorded
                    tmp_closed_position.append(tmp_code)  # remove from the buffer
            tradable_position_codes = [tmp_code for tmp_code in tradable_position_codes if tmp_code not in tmp_closed_position]

            # update latest price & close open price already reach tp (but not closed yesterday because just bought)
            position_codes = np.array(list(self.current_position.keys()))
            if len(position_codes) != 0:
                # update stock latest price
                tmp_idx = np.in1d(self.codes, position_codes)  # reorder code to match the order of price
                new_price = self.stock_price[self.today_idx, tmp_idx]
                new_price = dict(zip(self.codes[tmp_idx], new_price))  # mapping code to price
                for tmp_code in position_codes:
                    self.current_position[tmp_code]['last_price'] = new_price[tmp_code]

            # close position that reach tp (just bought yesterday)
            for tmp_code in tradable_position_codes:
                # if self.current_position[tmp_code]['last_price'] / self.current_position[tmp_code]['open_price'] - 1 >= self.current_position[tmp_code]['tp']:
                if calRet(self.current_position[tmp_code]['last_price'], self.current_position[tmp_code]['open_price']) >= self.current_position[tmp_code]['tp']:
                    del self.current_position[tmp_code]  # close position, ret already recorded
                    tradable_position_codes.remove(tmp_code)  # remove from the buffer

            # check for new order
            buy_list, sell_list = self.strategy(self, params)
            # buy
            if len(buy_list) != 0:
                tradable_code = np.array(list(buy_list.keys()))

                # buy rules
                # tradable_code = tradable_code[np.in1d(tradable_code, self.getTodayTradableStockCode())] # drop codes not tradable
                # tradable_code = tradable_code[~np.in1d(tradable_code, position_codes)]  # drop stocks that already bought

                # get buy stock price
                tmp_idx = np.in1d(self.codes, tradable_code) #  reorder codes, in order to match the order of price
                open_price = self.stock_price[self.today_idx, tmp_idx]
                open_price = dict(zip(self.codes[tmp_idx], open_price))  # mapping stock code to open price

                # add new codes to position
                for tmp_code in tradable_code:
                    tmp_buy_details = buy_list[tmp_code]
                    tmp_buy_details['last_price'] = open_price[tmp_code]
                    tmp_buy_details['open_price'] = open_price[tmp_code]
                    tmp_buy_details['holding_length'] = 0
                    self.current_position[tmp_code] = tmp_buy_details

            # sell (P&L already recorded)
            if len(sell_list) > 0:
                for tmp_code in sell_list:
                    del self.current_position[tmp_code]  # remove record from holding list, profit already recorded
                    tradable_position_codes.remove(tmp_code)

            # sell (tp & sl)
            if len(tradable_position_codes) > 0:  # not including stocks bought today
                tmp_idx = np.in1d(self.codes, tradable_position_codes)  #  reorder codes, in order to match the order of price
                high_price = self.stock_high[self.today_idx, tmp_idx]
                high_price = dict(zip(self.codes[tmp_idx], high_price))  # mapping stock code to high price
                for tmp_code in tradable_position_codes:
                    tmp_code_details = self.current_position[tmp_code]
                    highest_ret = calRet(high_price[tmp_code], tmp_code_details['open_price'])
                    # once reach tp, close with tp, use latest price to record profit
                    if highest_ret >= tmp_code_details['tp']:
                        close_price = (1 + tmp_code_details['tp']) * tmp_code_details['open_price']  # tp price
                        self.rets[self.today_idx] += calRet(close_price, tmp_code_details['last_price']) * tmp_code_details['weight'] # previous return already recorded, so use last price instead of open price
                        del self.current_position[tmp_code] # remove record from holding list
                        tradable_position_codes.remove(tmp_code)

            # update before move on to next day
            if len(self.current_position) > 0:
                # update mark-to-market return
                self.calTodayReturn()  # cal return of holding stocks
                # update holding length
                for tmp_code in self.current_position.keys():
                    self.current_position[tmp_code]['holding_length'] += 1

            # record holdings
            tmp_holding = list(self.current_position.keys())
            if len(tmp_holding) < self.max_position_num:
                tmp_empty = ['' for n in range(self.max_position_num - len(tmp_holding))]
                tmp_holding.extend(tmp_empty) # to align the num of position in the record
            tmp_holding.insert(0, self.getToday())
            tmp_holding.append(self.rets[self.today_idx])
            self.stock_holdings.append(tmp_holding)

            self.today_idx += 1
