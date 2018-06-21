from scipy import stats
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import h5py
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

predictionFileFullPath = {
    '2012': r'E:\model_results\2012\stock_proba_2012.csv',
    '2016': r'E:\model_results\stockscore_2016.csv'
}

comparingFactorFullPath = {
    'PB': 'LZ_CN_STKA_VAL_PB.h5',
    'PE': 'LZ_CN_STKA_VAL_PE.h5'
}

comparingFactorConfig ={
    'PE': {
        'table_name': 'STOCK_FUNDAMENTAL_TTM',
        'fields': ['date', 'code', 'PE_TTM']
    },
    'RET_5D': {
        'table_name': 'DERI_STOCK_TECH_INDICATORS',
        'fields': ['date', 'code', 'RET_5D']
    }
}

quoteFileName ={
    'close': 'LZ_CN_STKA_QUOTE_TCLOSE.h5',
    'high': 'LZ_CN_STKA_QUOTE_THIGH.h5',
    'tradable': 'LZ_CN_STKA_SLCIND_STOP_FLAG.h5',
    'adj_factor': 'LZ_CN_STKA_CMFTR_CUM_FACTOR.h5'
}

# prediction target
YPairs = {
    2: [0.01, 0.02, 0.03, 0.04, 0.05],
    5: [0.02, 0.03, 0.05, 0.07, 0.10],
    10: [0.03, 0.05, 0.07, 0.10, 0.15],
    20: [0.04, 0.07, 0.10, 0.15, 0.20],
    30: [0.05, 0.10, 0.15, 0.20, 0.25]
}

def calSerieIC(factor, rets):
    tmp_idx = (~np.isnan(factor)) & (~np.isnan((rets)))
    sp_corr = stats.spearmanr(factor[tmp_idx], rets[tmp_idx])[0]

    return  sp_corr

def getFactorsICs(factors, stock_rets):
    IC = np.zeros(factors.shape[0])
    for i in range(factors.shape[0]):
        tmp_factor = factors[i]
        tmp_ret = stock_rets[i]

        IC[i] = calSerieIC(tmp_factor, tmp_ret)

    return IC

def getDataFromDB(db_config, table_config):
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

def getDataFromFile(file_names, start_date, end_date):
    stock_data = {}
    for quote_name,quote_file_name in file_names.items():
        tmp_h5_file = h5py.File(quote_file_name, 'r')

        stock_data[quote_name] = tmp_h5_file['data'][...]
        if quote_name == 'close':
            trade_dates = tmp_h5_file['date'][...]
            codes = tmp_h5_file['header'][...]

    codes = np.array(list(map(lambda x: x.decode('utf-8').split('.')[1], codes)))
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))

    tmp_idx = (trade_dates >= start_date) & (trade_dates <= end_date)

    trade_dates = trade_dates[tmp_idx]
    for quote_name, quote_data in stock_data.items():
        stock_data[quote_name] = stock_data[quote_name][tmp_idx]

    stock_tradable_flag = (stock_data['tradable'] == 0)

    # adj quote
    stock_quotes = {}
    for quote_name in ['close', 'high']:
        stock_quotes[quote_name] = stock_data[quote_name] * stock_data['adj_factor']

    return stock_quotes, stock_tradable_flag, trade_dates, codes

def getMomentumFactor(file_names, days, comparing_factors, start_date, end_date):
    tmp_h5_file = h5py.File(file_names['close'])
    close_price = tmp_h5_file['data'][...]

    trade_dates = tmp_h5_file['date'][...]
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))
    tmp_row_idx = (trade_dates >= start_date) & (trade_dates <= end_date)

    tmp_nan_row = np.full(close_price.shape[1], np.nan)
    for tmp_day in days:
        tmp_rets = close_price[1:] / close_price[:-1] - 1
        tmp_rets = np.vstack([tmp_nan_row, tmp_rets])
        tmp_rets = tmp_rets[tmp_row_idx]

        tmp_label = 'RET_%dD' % tmp_day
        comparing_factors[tmp_label] = tmp_rets

    return comparing_factors

def getComparingFactorFromFile(factor_file_name, start_date, end_date):
    comparing_factor = {}
    for factor_name, file_name in factor_file_name.items():
        tmp_h5_file = h5py.File(file_name, 'r')
        comparing_factor[factor_name] = tmp_h5_file['data'][...]

    trade_dates = tmp_h5_file['date'][...]
    trade_dates = np.array(list(map(lambda x: str(x), trade_dates)))
    trade_dates = np.array(list(map(lambda x: '-'.join([x[:4], x[4:6], x[6:]]), trade_dates)))
    tmp_row_idx = (trade_dates >= start_date) & (trade_dates <= end_date)

    # trim rows
    for factor_name, factor_values in comparing_factor.items():
        comparing_factor[factor_name] = comparing_factor[factor_name][tmp_row_idx]

    # inverse
    for factor_name in ['PE', 'PB']:
        comparing_factor[factor_name] = 1. / comparing_factor[factor_name]

    return comparing_factor

def getPredicition(file_path, trade_dates, codes):
    prediction_data = pd.read_csv(file_path, dtype={'code': str}, index_col=None, header=0)

    # get all prediction types
    prediction_types = prediction_data.columns.tolist()
    for tmp_col in ['code', 'date']:
        prediction_types.remove(tmp_col)

    # pivot each prediciton
    tot_predicition_data = {}
    for tmp_predict_type in prediction_types:
        tmp_predict = prediction_data.pivot_table(values=tmp_predict_type, index = 'date', columns='code')
        tmp_predict = tmp_predict.reindex(trade_dates) # rearrange index
        tmp_predict = tmp_predict.reindex(codes, axis=1) # rearrange columns

        tot_predicition_data[tmp_predict_type] = tmp_predict.values

    return  tot_predicition_data

def calActualHighestHigh(stock_quotes, trade_dates, codes, predict_day):
    tmp_high_df = pd.DataFrame(stock_quotes['high'], index=trade_dates, columns=codes)
    highest_high = tmp_high_df.rolling(predict_day).max().shift(-predict_day)  # rolling is 1 row less than shift, so it is the high of the next n days
    highest_high = highest_high.values

    highest_ret = highest_high / stock_quotes['close'] - 1  # highest return for the next 2 days

    return highest_ret


if __name__ == '__main__':
    start_date = '2017-01-01'
    end_date = '2017-05-31'

    # get data from files
    stock_quotes, stock_tradable_flag, trade_dates, codes = getDataFromFile(quoteFileName, start_date, end_date)
    comparing_factors = getComparingFactorFromFile(comparingFactorFullPath, start_date, end_date)
    # calculate past returns as comparing factor
    tmp_days = [1, 5, 20]
    comparing_factors = getMomentumFactor(quoteFileName, tmp_days, comparing_factors, start_date, end_date)

    # get predicition
    file_path = predictionFileFullPath['2016']
    stock_predicition = getPredicition(file_path, trade_dates, codes)  # make sure having the same col & index num as stock quotes

    # calculate IC
    tot_predicition_IC = {}
    prediction_mean_IC = {}
    for predict_day, profit_percent in YPairs.items():
        tmp_highest_rets = calActualHighestHigh(stock_quotes, trade_dates, codes, predict_day)
        # predicition IC
        for tmp_percent in profit_percent:
            tmp_label = 'proba_1_%d_%d' % (predict_day, int(tmp_percent*100))
            tot_predicition_IC[tmp_label] = getFactorsICs(stock_predicition[tmp_label], tmp_highest_rets)

            prediction_mean_IC[tmp_label] = np.mean(tot_predicition_IC[tmp_label][:-predict_day])
        # comparing factors ICs
        for factor_name, factor_values in comparing_factors.items():
            tmp_label = '%s_PDT_%dD' % (factor_name, predict_day)

            tot_predicition_IC[tmp_label] = getFactorsICs(factor_values, tmp_highest_rets)

            prediction_mean_IC[tmp_label] = np.mean(tot_predicition_IC[tmp_label][:-predict_day])

    print(prediction_mean_IC)
    tmp_df = pd.Series(prediction_mean_IC)
    tmp_df.to_csv('predict_IC.csv')