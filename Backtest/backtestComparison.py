from backtestFrameworkComparison import BacktestingClass
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime
import pandas as pd
import copy
import time
# ================== version feature ==============
#  seperate stock into different layers according to prediciton

Config = {
    '2013':{
        'start_date' : '2014-01-01',
        'end_date' : '2014-06-30',
        'max_position_num': 5
    },

    '2014':{
        'start_date' : '2015-01-01',
        'end_date' : '2015-06-30',
        'max_position_num': 5
    },

    '2016':{
        'start_date' : '2017-01-01',
        'end_date' : '2017-06-30',
        'max_position_num': 5
    }
}

predictionFileFullPath = {
    'v6':{
        '2013': r'D:\model_results\v6\stockscore_2013.csv',
        '2014': r'D:\model_results\v6\stockscore_2014.csv',
        '2016': r'D:\model_results\v6\stockscore_2016.csv'
    },
    'v8':{
        '2013': r'D:\model_results\v8\stockscore_2013.csv',
        '2014': r'D:\model_results\v8\stockscore_2014.csv',
        '2016': r'D:\model_results\v8\stockscore_2016.csv'
    }
}


# prediction target
YPairs = {
    2: [0.01, 0.02, 0.03, 0.04, 0.05],
    5: [0.02, 0.03, 0.05, 0.07, 0.10],
    10: [0.03, 0.05, 0.07, 0.10, 0.15],
    20: [0.04, 0.07, 0.10, 0.15, 0.20],
    30: [0.05, 0.10, 0.15, 0.20, 0.25]
}
Versions = ['v6', 'v8']
Years = ['2013', '2014', '2016']


def maxDrawndown(rets):
    cum_rets = np.cumprod(1 + rets)
    series_cum_ret = pd.Series(cum_rets)
    expending_max_ret = series_cum_ret.expanding().max()
    drawn_down = expending_max_ret - series_cum_ret

    return drawn_down.max()

def strategy(BT_ins, params):
    max_holding_period = params['max_holding_period']
    # tp = BT_ins.prediction_pairs[max_holding_period]
    tp = params['tp']
    max_position_num = BT_ins.max_position_num
    # probability_threshold = 0.7

    # target_label = 'y_1_%dD_%dPCT' % (max_holding_period, int(tp * 100))
    target_label = 'proba_1_%d_%d' % (max_holding_period, int(tp * 100))

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
        target_predition = target_predition.sort_values(target_label, ascending=False)  # index 0 is the most probable to win

        #  ====================  buy stock pool screening ==============
        # drop stocks not tradable or already in holding
        target_predition = target_predition.loc[target_predition['code'].isin(BT_ins.getTodayTradableStockCode())]
        target_predition = target_predition.loc[~target_predition['code'].isin(position)]

        # select only stocks within the index
        # target_predition = target_predition.loc[target_predition['code'].isin(bt_ins.getTodayIndexComponentStockCode())]

        # drop those open == high (cannot buy)
        tmp_open = BT_ins.stock_price[BT_ins.today_idx]
        tmp_high = BT_ins.stock_high[BT_ins.today_idx]
        rising_stop_codes = BT_ins.codes[np.abs(tmp_high - tmp_open) < 0.001]
        target_predition = target_predition.loc[~target_predition['code'].isin(rising_stop_codes)]

        # drop those not reach the probability threshold
        # target_predition = target_predition.loc[target_predition[target_label] > probability_threshold]

        # drop st stock
        tmp_not_st_codes = BT_ins.getTodayNotSTStockCode()
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_st_codes)]

        # drop newly list stock
        tmp_not_newly_list_codes = BT_ins.getTodayNotNewlyListStockCode(60)
        target_predition = target_predition.loc[target_predition['code'].isin(tmp_not_newly_list_codes)]


        # ================ select buy list =======================
        # select stocks according to portion and probabilities
        # tmp_pool_stock_num = target_predition.shape[0]
        # tmp_select_start_pos = int(tmp_pool_stock_num * params['start_portion'])
        # tmp_select_end_pos = int(tmp_pool_stock_num * params['end_portion'])
        stocks_selected = target_predition.iloc[: max_position_num]

        stock_to_buy = stocks_selected['code'].values
        # stock_to_buy = np.array(['000002'])
        stock_to_sell = np.array(position)[~np.in1d(position, stocks_selected)]

        sell_weights = np.array(pos_weights)[~np.in1d(position, stocks_selected)]
        sell_weights = np.sum(sell_weights)
        buy_weights = 1 - tot_weights + sell_weights

        tmp_stock_num = stock_to_buy.size

        for stock in stock_to_buy:
            # buy_list[stock] = dict(zip(['weight', 'max_holding_period', 'tp'], [1./max_position_num, max_holding_period, tp]))
            buy_list[stock] = dict(
                zip(['weight', 'max_holding_period', 'tp'], [buy_weights / float(tmp_stock_num), max_holding_period, tp]))

        sell_list = list(stock_to_sell)

    return buy_list, sell_list

def multi_proc_run(bt_ins, params):
    print(params)
    bt_ins.run(params)
    result = bt_ins.getAnalysis()

    tmp_row_label = '%dD_%dPCT_%s' % (params['max_holding_period'], int(params['tp'] * 100), params['year'])
    tmp_col_labels = list(result.keys())
    for tmp_col_label in tmp_col_labels:
        tmp_new_col_label = '%s_%s' % (tmp_col_label, params['version'])
        result[tmp_new_col_label] = result.pop(tmp_col_label)  # rename label

    final_result = {'row':tmp_row_label, 'data':result}

    return final_result

if __name__ == '__main__':
    tot_results = pd.DataFrame([])
    #  === loop for different years ===
    for tmp_year in Years:
        tmp_config = Config[tmp_year]
        tmp_config['strategy'] = strategy
        # config['prediction_file_path'] = predictionFileFullPath['2016']
        # config['prediction_pairs'] = YPairs
        bt_ins = BacktestingClass(**tmp_config)  # initialize instance
        bt_ins.init('File')

        # ========== index =============
        index_rets = bt_ins.index_price[1:] / bt_ins.index_price[:-1] - 1
        index_result = {}
        index_result['avrYield_index'] = np.mean(index_rets) * 252
        index_result['std_index'] = np.std(index_rets) * np.sqrt(252)
        index_result['shapeRatio_index'] = index_result['avrYield_index'] / index_result['std_index']
        index_result['maxDrawndown_index'] = maxDrawndown(index_rets)

        # create record buffer
        tmp_row_labels = []
        for tmp_day, tmp_tps in YPairs.items():
            for tmp_tp in tmp_tps:
                tmp_row_labels.append('%dD_%dPCT_%s' % (tmp_day, int(tmp_tp*100), tmp_year))
        tmp_col_labels = []
        for tmp_indicator in ['avrYield', 'std', 'shapeRatio', 'maxDrawndown']:
            tmp_vers = Versions + ['index']
            for tmp_version in tmp_vers:
                tmp_col_labels.append('%s_%s' % (tmp_indicator, tmp_version))
        year_results = pd.DataFrame([], index=tmp_row_labels, columns=tmp_col_labels)

        # record index performance results
        for tmp_label, tmp_value in index_result.items():
            year_results.loc[:, tmp_label] = tmp_value

        pool = Pool(processes=3)
        #  === loop for different models ===
        for tmp_version in Versions:
            # load prediction
            tmp_pred_path = predictionFileFullPath[tmp_version][tmp_year]
            bt_ins.loadPrediction(tmp_pred_path)

            tmp_mulprc_results = []
            for tmp_day, tmp_tps in YPairs.items():
                for tmp_tp in tmp_tps:
                    params = {'max_holding_period': tmp_day, 'tp': tmp_tp, 'year':tmp_year, 'version':tmp_version}
                    tmp_bt_ins = copy.deepcopy(bt_ins)
                    tmp_prc = pool.apply_async(multi_proc_run, (tmp_bt_ins, params))
                    tmp_mulprc_results.append(tmp_prc)

            for tmp_prc in tmp_mulprc_results:
                tmp_result = tmp_prc.get()
                tmp_labels = list(tmp_result['data'].keys())
                tmp_values = [tmp_result['data'][x] for x in tmp_labels]
                year_results.loc[tmp_result['row'], tmp_labels] = tmp_values

        # end of the version loop (still in the year loop)
        tot_results = tot_results.append(year_results)
        pool.terminate()

    # end of the year loop
    tot_results.to_csv('mod_ver_comp.csv')