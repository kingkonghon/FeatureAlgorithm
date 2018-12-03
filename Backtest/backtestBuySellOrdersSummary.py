from backtestFrameworkBuySellOrders import BacktestingClass
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

config = {
    'start_date' : '2013-01-17',
    'end_date' : '2018-08-31',
    'start_year': 2012,
    'start_season': 1,
    'end_year': 2017,
    'end_season': 4,
    'season_in_year': 6,
    'max_position_num': 5,
    'commission': 0.002
}

# predictionFolderPath = r'D:\model_results\top_bottom\v12'
# predictionFolderPath = r'D:\model_results\top_bottom\prediction'
# predictionFolderPath = r'D:\model_results\top_bottom\v11'
# predictionFolderPath = r'D:\model_results\top_bottom\prediction_most_features'
buySellOrderFilePath = r'D:\model_results\top_bottom\rl_strategy_20_0002.csv'
# predictionFolderPath = r'D:\model_results\top_bottom\prediction_lgbm_rectified'

# buySellOrderFilePathSet = [
#     r'D:\model_results\top_bottom\rl\strategy_2014_2_002.csv',
#     r'D:\model_results\top_bottom\rl\strategy_2014_5_002.csv',
#     r'D:\model_results\top_bottom\rl\strategy_2014_10_002.csv',
#     r'D:\model_results\top_bottom\rl\strategy_2014_20_002.csv',
# ]

compare_strategy = r'D:\model_results\top_bottom\strategy_backtest_results_20D_1318.csv'

# prediction target
# YPairs = {
#     2: [0.01, 0.02, 0.03, 0.04, 0.05],
#     5: [0.02, 0.03, 0.05, 0.07, 0.10],
#     10: [0.03, 0.05, 0.07, 0.10, 0.15],
#     20: [0.04, 0.07, 0.10, 0.15, 0.20],
#     30: [0.05, 0.10, 0.15, 0.20, 0.25]
# }

def strategy(BT_ins):
    yesterday = BT_ins.getYesterday()

    position = list(BT_ins.getPosition().keys())
    max_position_num = BT_ins.max_position_num

    buy_list = {}
    sell_list = []

    # need to open new position
    # get today's stock prediction
    orders_today = BT_ins.prediction.loc[BT_ins.prediction['date'] == yesterday]
    buy_orders = orders_today.loc[orders_today['action'] == 0]
    sell_orders = orders_today.loc[orders_today['action'] == 1]

    # ===== buy orders
    # drop stocks not tradable or already in holding
    # buy_orders = buy_orders.loc[buy_orders['code'].isin(BT_ins.getTodayTradableStockCode())]
    buy_orders = buy_orders.loc[~buy_orders['code'].isin(position)]

    # screen out stocks with open price reaching rise stop
    # rising_stop_codes = BT_ins.getTodayRiseStopAtOpenStockCode()
    # buy_orders = buy_orders.loc[~buy_orders['code'].isin(rising_stop_codes)]

    # # drop st stock
    # tmp_not_st_codes = BT_ins.getTodayNotSTStockCode()
    # buy_orders = buy_orders.loc[buy_orders['code'].isin(tmp_not_st_codes)]

    # # drop newly list stock
    # tmp_not_newly_list_codes = BT_ins.getTodayNotNewlyListStockCode(60)
    # target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_newly_list_codes)]

    # select specific num from qualified stocks
    stock_to_buy = buy_orders['code'].values

    for stock in stock_to_buy:
        # buy_list[stock] = dict(zip(['weight', 'max_holding_period', 'tp'], [1./max_position_num, max_holding_period, tp]))
        buy_list[stock] = dict(
            zip(['weight', 'max_holding_period', 'tp'], [1. / max_position_num, 9999, 9999]))

    # need to close existing position
    sell_orders = sell_orders.loc[sell_orders['code'].isin(position)]
    sell_list = sell_orders['code'].tolist() # sell holdings if become negative

    return buy_list, sell_list


if __name__ == '__main__':
    tmp_path = r'D:\model_results\top_bottom\rl_priority2016'

    buySellOrderFilePathSet = os.listdir(tmp_path)
    curve_labels = list(map(lambda x: x.replace('strategy_rank_data_', ''), buySellOrderFilePathSet))
    curve_labels = list(map(lambda x: x.replace('.csv', ''), curve_labels))
    buySellOrderFilePathSet = list(map(lambda x: r'%s\%s' % (tmp_path, x), buySellOrderFilePathSet))

    config['strategy'] = strategy
    # config['order_file_path'] = buySellOrderFilePath
    config['order_file_path'] = buySellOrderFilePathSet[0]
    config['max_holding_period'] = 20
    config['checkpoint_period'] = [2,5,10]
    bt_ins = BacktestingClass(**config)
    bt_ins.init('File')

    dt_trade_dates = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), bt_ins.trade_dates))

    t_start = time.clock()
    bt_ins.run()

    tmp_res = bt_ins.getAnalysis('')  # record summary
    tmp_res = tmp_res.reset_index()
    tmp_res = tmp_res.rename({'index': 'year'}, axis=1)  # change index name
    tmp_res.loc[:, 'strategy'] = curve_labels[0]
    total_result = tmp_res.copy()

    tmp_cum_rets = bt_ins.getCumRets()
    plt.plot(dt_trade_dates, tmp_cum_rets)

    # buySellOrderFilePathSet.pop(0)
    for tmp_num in range(1, len(buySellOrderFilePathSet)):
        tmp_file_path = buySellOrderFilePathSet[tmp_num]
        tmp_curve_name = curve_labels[tmp_num]

        bt_ins.order_file_path = tmp_file_path
        bt_ins.reinit()
        bt_ins.run()
        tmp_cum_rets = bt_ins.getCumRets()
        plt.plot(dt_trade_dates, tmp_cum_rets)

        tmp_res = bt_ins.getAnalysis('')  # record summary
        tmp_res = tmp_res.reset_index()
        tmp_res = tmp_res.rename({'index': 'year'}, axis=1)  # change index name
        tmp_res.loc[:, 'strategy'] = tmp_curve_name
        total_result = total_result.append(tmp_res)

    benchmark_ret = pd.read_csv(compare_strategy, index_col=None)
    benchmark_ret = benchmark_ret['cum_rets']
    plt.plot(dt_trade_dates, benchmark_ret)

    # file_label = ['2_002', '5_002', '10_002', '20_002', 'old']
    curve_labels.append('old')
    plt.legend(curve_labels)
    plt.show()
    print("time eclipsed:", time.clock() - t_start)


    total_result.to_csv(r'D:\model_results\top_bottom\rl_total_summary_20181123.csv')
    # print(bt_ins.getAnalysis('backtest_result.csv'))