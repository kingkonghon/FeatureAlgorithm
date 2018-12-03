from backtestFrameworkTimingConDynamic import BacktestingClass
import numpy as np
import pandas as pd
import time

config = {
    'start_date' : '2017-01-01',
    'end_date' : '2018-08-31',
    'start_year': 2016,
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
predictionFolderPath = r'D:\model_results\top_bottom\prediction_nocheating_lgbm'
timingFilePath = 'D:/FeatureAlgorithm/Timing/lgbm_timing_prediction.csv'
# predictionFolderPath = r'D:\model_results\top_bottom\prediction_lgbm_rectified'

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
    max_position_num = BT_ins.max_position_num

    target_label = 'proba_1_%dD' % (max_holding_period)
    # target_label = 'proba_1'

    yesterday = BT_ins.getYesterday()
    position = list(BT_ins.getPosition().keys())
    hs300_constituents = BT_ins.getTodayHS300Contituents()

    timing_signal = BT_ins.timing.loc[yesterday, 'predict']

    buy_list = {}
    sell_list = []

    # need to open new position
    if (len(position) < max_position_num) and timing_signal == 1:
        # get today's stock prediction
        target_predition = BT_ins.prediction.loc[BT_ins.prediction['date'] == yesterday, ['date', 'code', target_label]]
        target_predition = target_predition.sort_values(target_label, ascending=False)

        # drop stocks not tradable or already in holding
        target_predition = target_predition.loc[target_predition['code'].isin(BT_ins.getTodayTradableStockCode())]
        target_predition = target_predition.loc[~target_predition['code'].isin(position)]

        # drop open price trigger rise stop (cannot buy)
        # tmp_open = BT_ins.stock_price[BT_ins.today_idx]
        # tmp_pre_close = BT_ins.stock_close[BT_ins.today_idx-1]
        # rising_stop_codes = BT_ins.codes[(tmp_open / tmp_pre_close - 1) > 0.095]
        rising_stop_codes = BT_ins.getTodayRiseStopAtOpenStockCode()
        target_predition = target_predition.loc[~target_predition['code'].isin(rising_stop_codes)]

        # drop codes belongs to GEB
        # target_predition = target_predition.loc[target_predition['code'].apply(lambda x: x[0] != '3')]

        # only select stocks from HS300 constituents
        target_predition = target_predition.loc[target_predition['code'].isin(hs300_constituents)]

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
    config['timing_file_path'] = timingFilePath
    config['max_holding_period'] = 20
    config['checkpoint_period'] = [2,5,10]
    bt_ins = BacktestingClass(**config)
    bt_ins.init('File')

    t_start = time.clock()
    bt_ins.run()
    print("time eclipsed:", time.clock() - t_start)

    print(bt_ins.getAnalysis('backtest_result.csv'))