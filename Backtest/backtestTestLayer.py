from backtestingFrameworkLayer import BacktestingClass
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import time
# ================== version feature ==============
#  seperate stock into different layers according to prediciton


config = {
    'start_date' : '2013-01-01',
    'end_date' : '2018-08-31',
    'start_year': 2012,
    'start_season': 1,
    'end_year': 2017,
    'end_season': 4,
    'max_position_num': 5,
    'season_in_year': 6,
    'commission': 0.002,
    'pro_label': 'proba_1_20D'
}

predictionFolderPath = r'D:\model_results\top_bottom\predict_y_rectify'

# # prediction target
# YPairs = {
#     2: [0.01, 0.02, 0.03, 0.04, 0.05],
#     5: [0.02, 0.03, 0.05, 0.07, 0.10],
#     10: [0.03, 0.05, 0.07, 0.10, 0.15],
#     20: [0.04, 0.07, 0.10, 0.15, 0.20],
#     30: [0.05, 0.10, 0.15, 0.20, 0.25]
# }

def strategy(BT_ins, params):
    max_holding_period = BT_ins.max_holding_period
    # tp = BT_ins.prediction_pairs[max_holding_period]
    # tp = 0.02
    # max_position_num = BT_ins.max_position_num
    # probability_threshold = 0.7

    # target_label = 'y_1_%dD_%dPCT' % (max_holding_period, int(tp * 100))
    # target_label = 'proba_1_%d_%d' % (max_holding_period, int(tp * 100))
    # target_label = 'proba_1_%dD' % max_holding_period
    target_label = BT_ins.pro_label

    today = BT_ins.getToday()
    yesterday = BT_ins.getYesterday()
    position_details = BT_ins.getPosition()
    position = list(position_details.keys())
    pos_weights = list(map(lambda x:position_details[x]['weight'], position))
    tot_weights = sum(pos_weights)

    buy_list = {}
    sell_list = []

    if tot_weights == 0:
        # need to open new position
        # get today's stock prediction
        target_predition = BT_ins.prediction.loc[BT_ins.prediction['date'] == yesterday, ['date', 'code', target_label]]
        target_predition = target_predition.sort_values(target_label, ascending=False)

        # drop stocks not tradable or already in holding
        # target_predition = target_predition.loc[target_predition['code'].isin(BT_ins.getTodayTradableStockCode())]
        # target_predition = target_predition.loc[~target_predition['code'].isin(position)]

        # select only stocks within the index
        # target_predition = target_predition.loc[target_predition['code'].isin(bt_ins.getTodayIndexComponentStockCode())]

        # drop those open == high (cannot buy)
        rising_stop_codes = BT_ins.getTodayRiseStopAtOpenStockCode()
        target_predition = target_predition.loc[~target_predition['code'].isin(rising_stop_codes)]

        # drop those not reach the probability threshold
        # target_predition = target_predition.loc[target_predition[target_label] > probability_threshold]

        # drop st stock
        tmp_not_st_codes = BT_ins.getTodayNotSTStockCode()
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_st_codes)]

        # drop newly list stock
        tmp_not_newly_list_codes = BT_ins.getTodayNotNewlyListStockCode(60)
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_newly_list_codes)]

        # select stocks according to portion and probabilities
        tmp_pool_stock_num = target_predition.shape[0]
        tmp_select_start_pos = int(tmp_pool_stock_num * params['start_portion'])
        tmp_select_end_pos = int(tmp_pool_stock_num * params['end_portion'])
        stocks_selected = target_predition.iloc[tmp_select_start_pos:tmp_select_end_pos]
        stocks_selected = stocks_selected['code'].values

        stock_to_buy = stocks_selected[~np.in1d(stocks_selected, position)]  # buy selected stocks not in the position
        # stock_to_buy = np.array(['000002'])
        stock_to_sell = np.array(position)[~np.in1d(position, stocks_selected)]

        sell_weights = np.array(pos_weights)[~np.in1d(position, stocks_selected)]
        sell_weights = np.sum(sell_weights)
        buy_weights = 1 - tot_weights + sell_weights

        tmp_stock_num = stock_to_buy.size

        for stock in stock_to_buy:
            # buy_list[stock] = dict(zip(['weight', 'max_holding_period', 'tp'], [1./max_position_num, max_holding_period, tp]))
            buy_list[stock] = dict(
                zip(['weight', 'max_holding_period', 'tp'], [buy_weights / float(tmp_stock_num), max_holding_period, 99999]))

        sell_list = list(stock_to_sell)

    return buy_list, sell_list


if __name__ == '__main__':

    config['strategy'] = strategy
    config['prediction_folder_path'] = predictionFolderPath
    config['max_holding_period'] = 20
    bt_ins = BacktestingClass(**config)
    bt_ins.init('File')

    # calculate index cumulative returns
    index_rets = bt_ins.index_price[1:] / bt_ins.index_price[:-1] - 1
    index_rets = np.hstack([0, index_rets])
    index_cum_rets = np.cumprod(1 + index_rets)
    index_cum_rets = index_cum_rets - 1

    # calculate strategy cumulative returns
    strategy_cum_rets = pd.read_csv(r'D:\model_results\top_bottom\strategy_backtest_results_%dD.csv' % config['max_holding_period'],index_col=None)
    strategy_cum_rets = strategy_cum_rets['cum_rets'].values

    # calculate all layer cumulative returns
    dt_trade_dates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), bt_ins.trade_dates))

    params = {
        'start_portion': 0,
        'end_portion': 0.1
    }

    all_cum_rets = {}
    ret_labels = []
    for i in range(10):
        # t_start = time.clock()
        params['start_portion'] = i * 0.1
        params['end_portion'] = (i + 1) * 0.1
        bt_ins.run(params)

        tmp_label = '%dth_ptn' % i
        all_cum_rets[tmp_label] = bt_ins.getCumRets()
        ret_labels.append(tmp_label)

        plt.plot(dt_trade_dates, all_cum_rets[tmp_label])

        # reinit instance
        bt_ins.reinit()
    # print("time eclipsed:", time.clock() - t_start)

    # add index to the plot
    plt.plot(dt_trade_dates, index_cum_rets, 'k--')
    ret_labels.append('HS300')

    # add strategy to the plot
    plt.plot(dt_trade_dates, strategy_cum_rets, 'r-.')
    ret_labels.append('strategy')

    plt.legend(ret_labels)
    plt.show()

    # print(bt_ins.getAnalysis('backtest_result.csv'))