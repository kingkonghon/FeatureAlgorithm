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
    'close': 'LZ_CN_STKA_QUOTE_TCLOSE.h5',
    'tradable_flag': 'LZ_CN_STKA_SLCIND_STOP_FLAG.h5',
    'adj_factor': 'LZ_CN_STKA_CMFTR_CUM_FACTOR.h5',
    'listday_count': 'LZ_CN_STKA_SLCIND_TRADEDAYCOUNT.h5',
    'not_st_flag': 'LZ_CN_STKA_SLCIND_ST_FLAG.h5'
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

    # adj quote
    for quote_name in ['open', 'high', 'close']:
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


# def getPredictionFromFile(file_path):
#     prediction_data = pd.read_csv(file_path, dtype={'code': str}, index_col=None, header=0)
#
#     # prediction_data = prediction_data.drop('Unnamed: 0', axis=1)  # drop the first column
#     # prediction_data.loc[:, 'code'] = prediction_data['code'].apply(lambda x: 'null' if np.isnan(x) else str(int(x)))
#     # prediction_data.loc[:, 'code'] = prediction_data['code'].apply(lambda x: '0'*(6-len(x)) + x if x != 'null' else x)
#     return prediction_data

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
        self.start_year = kwargs.get('start_year')
        self.start_season = kwargs.get('start_season')
        self.end_year = kwargs.get('end_year')
        self.end_season = kwargs.get('end_season')
        self.threshold_lag = kwargs.get('threshold_lag')
        self.strategy = kwargs.get('strategy')
        self.prediction_folder_path = kwargs.get('prediction_folder_path')
        self.max_holding_period = kwargs.get('max_holding_period')
        self.max_position_num = kwargs.get('max_position_num')
        self.commission = kwargs.get('commission')
        self.prediction = []
        self.Y = []
        self.stock_price = []
        self.stock_high = []
        self.stock_close = []
        self.trade_dates = []
        self.codes = []
        self.tradable_flag = []
        self.not_st_flag = []
        self.listday_count = []
        self.rets = []
        self.cum_rets = []
        self.tradeday_num = 0
        self.today_idx = 1  # day 1 trade is based on day 0 prediction
        self.current_position = {}
        self.stock_holdings = []

    def init(self, method):
        # get prediction
        self.getPredictionFromFile()
        # get historical Y
        # self.getY()

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
            self.stock_close = stock_quote['close']
            self.tradable_flag = stock_quote['tradable_flag']
            self.not_st_flag = stock_quote['not_st_flag']
            self.listday_count = stock_quote['listday_count']
            self.trade_dates = trade_dates
            self.codes = codes

            self.index_price = index_quote['open']

        self.tradeday_num = self.trade_dates.size - 1  # the last day's open is needed to calculate the last return

        self.rets = np.zeros(self.tradeday_num)
        self.cum_rets = np.ones(self.tradeday_num)

    def getPredictionFromFile(self):
        self.prediction = pd.DataFrame([])

        current_year = self.start_year
        current_season = self.start_season
        while True:
            full_path_name = r'%s\stockscore_%ds%d.csv' % (self.prediction_folder_path, current_year, current_season)
            tmp_predict = pd.read_csv(full_path_name, dtype={'code': str}, index_col=None, header=0)
            self.prediction = self.prediction.append(tmp_predict)

            current_season += 1

            if (current_year == self.end_year) and (current_season > self.end_season):
                break

            if current_season == 5:  # move onto next year
                current_season = 1
                current_year += 1

    def getY(self):
        table_name = 'STOCK_TOP_BOTTOM_Y'
        quant_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

        field = '`Y_%dD`' % self.max_holding_period
        sql_statement = "select `date`, `code`, %s from %s where date between '%s' and '%s'" % (field, table_name, self.start_date, self.end_date)
        self.Y = pd.read_sql(sql_statement, quant_engine)
        pass


    def calTodayReturn(self):
        position_codes = list(self.current_position.keys())
        col_idx = np.in1d(self.codes, position_codes)
        # ret = self.stock_price[self.today_idx+1, col_idx] / self.stock_price[self.today_idx, col_idx] - 1  # next day's open - today's open
        ret = calRet(self.stock_price[self.today_idx+1, col_idx], self.stock_price[self.today_idx, col_idx])  # next day's open - today's open

        reorder_code = self.codes[col_idx]  # mapping weights to stock returns
        current_stock_weights = np.array(list(map(lambda x: self.current_position[x]['weight'], reorder_code)))  # get weight corresponding to new order of codes
        self.rets[self.today_idx] += np.sum(ret * current_stock_weights)

    def getToday(self):
        return self.trade_dates[self.today_idx]

    def getYesterday(self):
        return self.trade_dates[self.today_idx-1]

    def getPosition(self):
        return self.current_position

    def getTodayTradableStockCode(self):
        return self.codes[self.tradable_flag[self.today_idx]]

    def getTodayNotSTStockCode(self):
        return self.codes[self.not_st_flag[self.today_idx]]

    def getTodayNotNewlyListStockCode(self, min_list_day_num):
        tmp_idx = (self.listday_count[self.today_idx] > min_list_day_num)
        return self.codes[tmp_idx]

    def getTodayFallStopAtOpenStockCode(self):
        # open price trigger fall stop  (cannot sell)
        tmp_open = self.stock_price[self.today_idx]
        tmp_pre_close = self.stock_close[self.today_idx - 1]
        fall_stop_code = self.codes[(tmp_open / tmp_pre_close - 1) < - 0.095]

        return fall_stop_code

    def getTodayRiseStopAtOpenStockCode(self):
        # drop open price trigger rise stop (cannot buy)
        tmp_open = self.stock_price[self.today_idx]
        tmp_pre_close = self.stock_close[self.today_idx - 1]
        rising_stop_codes = self.codes[(tmp_open / tmp_pre_close - 1) > 0.095]

        return rising_stop_codes

    def getTodayThreshold(self):
        percentage = 0.01
        # min_one_precision = 0.5
        pred_label = 'proba_1_%dD' % (self.max_holding_period)
        # y_label = 'Y_%dD' % self.max_holding_period
        yesterday = self.getYesterday()
        if self.today_idx >= self.threshold_lag:
            lag_day = self.trade_dates[self.today_idx - self.threshold_lag]
            # take lag days' prediction
            past_prediciton = self.prediction.loc[(self.prediction['date']>= lag_day) & (self.prediction['date'] <= yesterday), ['date', 'code', pred_label]]
            past_prediciton = past_prediciton.loc[~past_prediciton[pred_label].isnull()]  # drop NA
            # past_y = self.Y.loc[(self.Y['date']>= lag_day) & (self.Y['date']<= yesterday)]
            # past_prediciton = past_prediciton.merge(past_y, on=['date', 'code'])

            # take top prediction set, also meet the precision requirement, get the threshold
            # final_threshold = np.nan
            final_threshold = past_prediciton[pred_label].quantile(1 - percentage, interpolation='midpoint')
            # while percentage > 0:
            #
        else:
            final_threshold = 0
        return final_threshold

    def getCumRets(self):
        return np.cumprod(1 + self.rets)

    def maxDrawndown(self, cum_rets):
        series_cum_ret = pd.Series(cum_rets+1)
        expending_max_ret = series_cum_ret.expanding().max()
        drawn_down = (expending_max_ret - series_cum_ret) / expending_max_ret

        return drawn_down.max()

    def getAnalysis(self, output_file_name):
        # cumulative returns
        self.cum_rets = np.cumprod(1 + self.rets) - 1
        self.cum_rets = np.hstack([0, self.cum_rets])  # the first date is 1, the second date is 1 * ret_1...

        tmp_analysis = {}

        tmp_analysis['avrYield'] = np.mean(self.rets) * 252
        tmp_analysis['std'] = np.std(self.rets) * np.sqrt(252)
        tmp_analysis['shapeRatio'] = tmp_analysis['avrYield'] / tmp_analysis['std']
        tmp_analysis['maxDrawdown'] = self.maxDrawndown(self.cum_rets)

        anaylysis_result = pd.DataFrame(tmp_analysis, index=['total'])

        dt_trade_dates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), self.trade_dates))

        index_rets = self.index_price[1:] / self.index_price[:-1] - 1
        index_rets = np.hstack([0, index_rets])
        index_cum_rets = np.cumprod(1 + index_rets) - 1

        # total cumulative returns
        plt.figure(1)
        plt.plot(dt_trade_dates, self.cum_rets)
        plt.plot(dt_trade_dates, index_cum_rets)
        plt.legend(['stategy', 'index'])

        # yearly cumulative returns
        plt.figure(2)
        tmp_years = list(map(lambda x: x[:4], self.trade_dates))
        tmp_years = list(set(tmp_years))
        tmp_years.sort(reverse=False)
        year_num = len(tmp_years)
        pro_yearly_cum_rets = {}
        index_yearly_cum_rets = {}
        profolio_rets = np.hstack([0, self.rets])
        dt_trade_dates = np.array(dt_trade_dates) # in order to slice
        for i, tmp_yr in enumerate(tmp_years):
            tmp_idx = list(map(lambda x:x[:4] == tmp_yr, self.trade_dates))
            tmp_pro_rets = profolio_rets[tmp_idx]
            tmp_index_rets = index_rets[tmp_idx]
            tmp_dt = dt_trade_dates[tmp_idx]
            # yearly cumulative returns (from the start of a year)
            pro_yearly_cum_rets[tmp_yr] = np.cumprod(1 + tmp_pro_rets) - 1
            index_yearly_cum_rets[tmp_yr] = np.cumprod(1 + tmp_index_rets) - 1
            # plot p&l curve
            plt.subplot(year_num, 1, i+1)
            plt.plot(tmp_dt, pro_yearly_cum_rets[tmp_yr])
            plt.plot(tmp_dt, index_yearly_cum_rets[tmp_yr])
            plt.plot(tmp_dt, np.zeros(tmp_dt.size), '--')
            plt.legend(['stategy', 'index'])
            # calculate key statistics
            tmp_analysis = {}
            tmp_analysis['avrYield'] = np.mean(tmp_pro_rets) * 252
            tmp_analysis['std'] = np.std(tmp_pro_rets) * np.sqrt(252)
            tmp_analysis['shapeRatio'] = tmp_analysis['avrYield'] / tmp_analysis['std']
            tmp_analysis['maxDrawdown'] = self.maxDrawndown(pro_yearly_cum_rets[tmp_yr])

            anaylysis_result.loc[tmp_yr] = tmp_analysis

        # resembled yearly returns
        plt.figure(3)
        resembled_yearly_pro_rets = np.array([])
        resembled_yearly_index_rets = np.array([])
        for i, tmp_yr in enumerate(tmp_years):
            tmp_pro_rets = pro_yearly_cum_rets[tmp_yr]
            tmp_pro_rets[-1] = None
            resembled_yearly_pro_rets = np.hstack([resembled_yearly_pro_rets,tmp_pro_rets])
            tmp_index_rets = index_yearly_cum_rets[tmp_yr]
            tmp_index_rets[-1] = None
            resembled_yearly_index_rets = np.hstack([resembled_yearly_index_rets, tmp_index_rets])
        plt.plot(dt_trade_dates, resembled_yearly_pro_rets)
        plt.plot(dt_trade_dates, resembled_yearly_index_rets)
        plt.plot(dt_trade_dates, np.zeros(dt_trade_dates.size), '--')
        plt.legend(['stategy', 'index'])

        plt.show()

        # write stocking holding records
        tmp_cols = ['date']
        tmp_cols.extend(['code%d' % n for n in range(self.max_position_num)])
        tmp_cols.append('return')
        stock_holding = pd.DataFrame(self.stock_holdings, columns=tmp_cols)
        stock_holding.to_csv(output_file_name)

        return anaylysis_result

    def run(self):
        while self.today_idx < self.tradeday_num:
            print(self.getToday())

            # only close stocks tradable
            tradable_position_codes = self.getTodayTradableStockCode()  # today's tradable
            tradable_position_codes = tradable_position_codes[np.in1d(tradable_position_codes, list(self.current_position.keys()))]

            # drop stocks open price trigger fall stop (cannot sell)
            fall_stop_codes = self.getTodayFallStopAtOpenStockCode()
            tradable_position_codes = tradable_position_codes[~np.in1d(tradable_position_codes, fall_stop_codes)]

            tradable_position_codes = list(tradable_position_codes)

            # close position reaching maximum holding period
            for tmp_code in tradable_position_codes:
                if self.current_position[tmp_code]['holding_length'] >= self.current_position[tmp_code]['max_holding_period']:
                    del self.current_position[tmp_code] # close position, ret already recorded
                    tradable_position_codes.remove(tmp_code)  # remove from the buffer


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
            buy_list, sell_list = self.strategy(self)
            # buy
            if len(buy_list) != 0:
                tradable_code = np.array(list(buy_list.keys()))

                # buy rules
                tradable_code = tradable_code[np.in1d(tradable_code, self.getTodayTradableStockCode())] # drop codes not tradable
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
                    self.rets[self.today_idx] -= self.commission * tmp_buy_details['weight']

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
