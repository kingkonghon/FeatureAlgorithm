from backtestFrameworkMixProDynamic import BacktestingClass
import numpy as np
import pandas as pd
import time

config = {
    'start_date' : '2013-01-01',
    'end_date' : '2018-08-31',
    'start_year': 2012,
    'start_season': 1,
    'end_year': 2017,
    'end_season': 4,
    'season_in_year': 6,
    'max_position_num': 5,
    'commission': 0.002
}

predictionFolderPath = r'D:\model_results\top_bottom\predict_y_rectify_old_features'
# predictionFolderPath = r'D:\model_results\top_bottom\predict_y_rectify_original_features'

stockPoolFilePath = r'D:\model_results\top_bottom\stock_pool.csv'
predictPrecisionFilePath = r'D:\model_results\top_bottom\predict_y_rectify\precision_recall.csv'

# prediction target
# YPairs = {
#     2: [0.01, 0.02, 0.03, 0.04, 0.05],
#     5: [0.02, 0.03, 0.05, 0.07, 0.10],
#     10: [0.03, 0.05, 0.07, 0.10, 0.15],
#     20: [0.04, 0.07, 0.10, 0.15, 0.20],
#     30: [0.05, 0.10, 0.15, 0.20, 0.25]
# }

def strategy(BT_ins):
    max_holding_period = BT_ins.max_holding_period
    # tp = BT_ins.prediction_pairs[max_holding_period]
    tp = 0.02
    max_position_num = BT_ins.max_position_num
    # probability_threshold = 0.7

    tot_prd_daynums = [2,5,10,20]
    tot_labels = list(map(lambda x: 'proba_1_%dD' % x, tot_prd_daynums))
    weighted_label = 'weighted_proba'
    # target_label = 'y_1_%dD_%dPCT' % (max_holding_period, int(tp * 100))
    # target_label = 'proba_1_%d_%d' % (max_holding_period, int(tp * 100))
    # target_label = 'proba_1_%dD' % (max_holding_period)
    # target_label = 'proba_1'

    # today = BT_ins.getToday()
    yesterday = BT_ins.getYesterday()
    position = list(BT_ins.getPosition().keys())
    hs300_constituents = BT_ins.getTodayHS300Contituents()

    buy_list = {}
    sell_list = []

    # need to open new position
    if len(position) < max_position_num:
        # get today's stock prediction
        target_predition = BT_ins.prediction.loc[BT_ins.prediction['date'] == yesterday]

        # calculate weighted probability
        target_predition.loc[:, weighted_label] = 0
        for tmp_label in tot_labels:
            target_predition.loc[:, weighted_label] = \
                target_predition[weighted_label] + target_predition[tmp_label] * 0.25 # equally weighted  #bt_ins.current_prd_precision.loc[tmp_label, bt_ins.prd_precision_threshold]  # sum (probability * weight)

        # sorted by weighted proba
        target_predition = target_predition.sort_values(weighted_label, ascending=False)

        # drop stocks not tradable or already in holding
        target_predition = target_predition.loc[target_predition['code'].isin(BT_ins.getTodayTradableStockCode())]
        target_predition = target_predition.loc[~target_predition['code'].isin(position)]

        # drop open price trigger rise stop (cannot buy)
        # tmp_open = BT_ins.stock_price[BT_ins.today_idx]
        # tmp_pre_close = BT_ins.stock_close[BT_ins.today_idx-1]
        # rising_stop_codes = BT_ins.codes[(tmp_open / tmp_pre_close - 1) > 0.095]
        rising_stop_codes = BT_ins.getTodayRiseStopAtOpenStockCode()
        target_predition = target_predition.loc[~target_predition['code'].isin(rising_stop_codes)]

        # drop codes belonging to GEB
        # target_predition = target_predition.loc[target_predition['code'].apply(lambda x: x[0] != '3')]

        # choose only from HS300 constituents
        target_predition = target_predition.loc[target_predition['code'].isin(hs300_constituents)]
        # choose only from given stock pool
        # target_predition = target_predition.loc[target_predition['code'].isin(BT_ins.stock_pool)]

        # drop those not reach the probability threshold
        # target_predition = target_predition.loc[target_predition[target_label] > probability_threshold]

        # drop st stock
        tmp_not_st_codes = BT_ins.getTodayNotSTStockCode()
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_st_codes)]

        # drop newly list stock
        tmp_not_newly_list_codes = BT_ins.getTodayNotNewlyListStockCode(60)
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_newly_list_codes)]

        # select specific num from qualified stocks
        stocks_to_buy = target_predition.iloc[:(max_position_num - len(position))]
        stock_to_buy = stocks_to_buy['code'].values

        for stock in stock_to_buy:
            # buy_list[stock] = dict(zip(['weight', 'max_holding_period', 'tp'], [1./max_position_num, max_holding_period, tp]))
            buy_list[stock] = dict(
                zip(['weight', 'max_holding_period', 'tp'], [1. / max_position_num, max_holding_period, 9999]))

    # need to close existing position
    if len(position) > 0:
        for pos_code, pos_details in BT_ins.getPosition().items():
            tmp_remaining_days = pos_details['max_holding_period'] - pos_details['holding_length']
            if tmp_remaining_days in BT_ins.checkpoint_period:  # remaining day num match one of the checkpoint
                target_label = 'proba_1_%dD' % (tmp_remaining_days)
                pos_predict = BT_ins.prediction.loc[(BT_ins.prediction['date'] == yesterday) & (BT_ins.prediction['code'] == pos_code), target_label]
                if not pos_predict.empty:
                    pos_predict = pos_predict.iloc[0]
                    if pos_predict < 0.5:
                        sell_list.append(pos_code) # sell holdings if become negative

    return buy_list, sell_list


if __name__ == '__main__':
    config['strategy'] = strategy
    config['prediction_folder_path'] = predictionFolderPath
    config['prd_pre_file_path'] = predictPrecisionFilePath
    config['prd_precision_threshold'] = '0.8'
    config['max_holding_period'] = 20
    config['checkpoint_period'] = [2,5,10]
    config['stock_pool_file_path'] = stockPoolFilePath
    bt_ins = BacktestingClass(**config)
    bt_ins.init('File')

    t_start = time.clock()
    bt_ins.run()
    print("time eclipsed:", time.clock() - t_start)

    # record cumulative returns
    tmp_cum_rets = bt_ins.getCumRets()
    tmp_cum_rets = pd.DataFrame({'date': bt_ins.trade_dates, 'cum_rets': tmp_cum_rets})
    tmp_cum_rets.to_csv(r'D:\model_results\top_bottom\strategy_backtest_results_%dD.csv' % config['max_holding_period'],
                        index=False)

    print(bt_ins.getAnalysis('backtest_result.csv'))