from sklearn.metrics import recall_score, precision_score
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RandomizedSearchCV
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import chi2

import sklearn_crfsuite
# from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import gc

from calStockIndexPeakTrough import calFull as loadY
from loadTimingDataFromDB import loadData as loadX

def Xpoint2Set(data, set_length):
    all_x_points = data.to_dict(orient='records')
    X = [all_x_points[tmp_idx:tmp_idx+set_length] for tmp_idx in range(len(all_x_points) - set_length + 1)]
    return X

def Chi2(bin1, bin2, col, target):
    N = bin1.shape[0] + bin2.shape[0]
    bad1 = bin1[target].sum()
    bad2 = bin2[target].sum()
    good1 = bin1.shape[0] - bad1
    good2 = bin2.shape[0] - bad2

    if (good1 + good2 == 0) or (bad1 + bad2 == 0):  # both bins are one-sided
        return 0

    O = np.matrix([[good1, good2], [bad1, bad2]])
    E = np.matrix(np.zeros([2,2]))
    chi2_val = 0
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            E[i,j] = O[i].sum() * O[:,j].sum() / N
            chi2_val += ((O[i,j] - E[i,j]) ** 2) / E[i,j]

    return chi2_val

def ChiMerge_MaxInterval_Original_Numeric(data, col, target, max_interval=5, pvalue=0.05):
    # form original bins
    original_bin_num = 100
    tmp_percentage = np.array(range(1, original_bin_num + 1)) * (1. / original_bin_num)
    tmp_quantile = data[col].quantile(tmp_percentage, interpolation='midpoint')
    real_colLevels = tmp_quantile.unique()

    real_colLevels = np.append([-np.inf], real_colLevels)
    real_colLevels = np.append(real_colLevels, [np.inf])  # len: original_bin_num + 2

    bin_lower_boundary_idx = list(range(len(real_colLevels) - 1))
    bin_upper_boundary_idx = list(range(1, len(real_colLevels)))

    # combine all the empty bins with neigbours
    flag_finished = False
    while not flag_finished:
        flag_finished = True
        for pos, (tmp_low_idx, tmp_high_idx) in enumerate(zip(bin_lower_boundary_idx, bin_upper_boundary_idx)):
            tmp_bin = data.loc[(data[col] > real_colLevels[tmp_low_idx]) & (data[col] <= real_colLevels[tmp_high_idx]), [col,target]]  # data in this bin

            if tmp_bin.empty: # empty bin, combine with previous or next bin
                if pos == 0:
                    remain_pos = pos   # combine to the next bins
                elif pos == len(bin_lower_boundary_idx) - 1:
                    remain_pos = pos - 1   # combine to the previous bins
                else:
                    tmp_previous_low_idx = bin_lower_boundary_idx[pos - 1]
                    tmp_previous_high_idx = bin_upper_boundary_idx[pos - 1]
                    tmp_previous_bin = data.loc[(data[col] > real_colLevels[tmp_previous_low_idx]) &
                                            (data[col] <= real_colLevels[tmp_previous_high_idx]), [col,target]]  # data in the previous bin

                    tmp_next_low_idx = bin_lower_boundary_idx[pos + 1]
                    tmp_next_high_idx = bin_upper_boundary_idx[pos + 1]
                    tmp_next_bin = data.loc[(data[col] > real_colLevels[tmp_next_low_idx]) &
                                                (data[col] <= real_colLevels[tmp_next_high_idx]), [col,target]]  # data in the next bin

                    # compare which bin's bad ratio is closer to 0
                    next_br = tmp_previous_bin[target].sum() / tmp_previous_bin.shape[0]
                    previous_br = tmp_next_bin[target].sum() / tmp_next_bin.shape[0]
                    if next_br > previous_br:
                        remain_pos = pos - 1  # combine with the previous bin
                    else:
                        remain_pos = pos  # combine with the next bin

                bin_lower_boundary_idx.pop(remain_pos + 1)
                bin_upper_boundary_idx[remain_pos] = bin_upper_boundary_idx.pop(remain_pos + 1)  # combine bins
                flag_finished = False
                break

    chi2_thre_val = chi2.isf(pvalue, df=1)  # == degree of freedom: (2 bins - 1) * (2 categories(good or bad) - 1) = 1

    flag_finished = False
    while not flag_finished:
        flag_finished = True

        tmp_zip_idx = zip(bin_lower_boundary_idx[:-1], bin_upper_boundary_idx[:-1])  # less the last index, to aviod out of range error
        for pos, (tmp_low_idx, tmp_high_idx) in enumerate(tmp_zip_idx):
            tmp_bin = data.loc[(data[col] > real_colLevels[tmp_low_idx]) & (data[col] <= real_colLevels[tmp_high_idx]), [col, target]]  # data in this bin

            tmp_next_low_idx = bin_lower_boundary_idx[pos + 1]
            tmp_next_high_idx = bin_upper_boundary_idx[pos + 1]
            tmp_next_bin = data.loc[(data[col] > real_colLevels[tmp_next_low_idx]) &
                                    (data[col] <= real_colLevels[tmp_next_high_idx]), [col, target]]  # data in the next bin

            tmp_chi_val = Chi2(tmp_bin, tmp_next_bin, col, target)
            if tmp_chi_val <= chi2_thre_val: #  unable to reject null hypothesis -->>H0: the separation of two bins is irrelevant to target
                bin_lower_boundary_idx.pop(pos+1)
                bin_upper_boundary_idx[pos] = bin_upper_boundary_idx.pop(pos+1)  # combine two neighbour bins together
                flag_finished = False  # not finished
                break

    cutOffPoints = real_colLevels[bin_upper_boundary_idx]
    return cutOffPoints

def AssignBinNumeric(x, cutOffPoints):
    numBin = len(cutOffPoints)
    if x <= cutOffPoints[0]:
        return 0
    elif x >= cutOffPoints[-1]:
        print('error, larger than infinity!')
        raise ValueError
    else:
        for i in range(0, numBin - 1):
            if cutOffPoints[i] < x <= cutOffPoints[i + 1]:
                return i + 1
        return np.nan

def binsOneHotEncoding(data, bin_cols):
    tot_bin_cols = []
    for tmp_col in bin_cols:
        data.loc[:, tmp_col] = data[tmp_col].astype('float')
        tot_bin_num = data.loc[~data[tmp_col].isnull(), tmp_col].unique()
        for tmp_bin in tot_bin_num:
            bin_name = tmp_col + '.' + str(tmp_bin)
            tot_bin_cols.append(bin_name)
            data.loc[:, bin_name] = (data[tmp_col] == tmp_bin).astype('float')

    data = data.drop(bin_cols, axis=1)

    return data, tot_bin_cols


def getDiscreteFeatures(all_data, cols_to_processed):
    # original_features_name = all_data.columns.tolist()
    # original_features_name.remove('date')
    # original_features_name.remove('Y')

    all_data.loc[:, 'bad'] = all_data['Y'] == -1.0

    tot_cutoff_points = {}
    bin_features_name = []
    for tmp_col in cols_to_processed:
        print(tmp_col)
        tmp_cutoff_points = ChiMerge_MaxInterval_Original_Numeric(all_data, tmp_col, 'bad')
        print('bin num:', len(tmp_cutoff_points))

        if len(tmp_cutoff_points) == 1:  # drop feature if only 1 bin left
            continue
        else:
            tmp_bin_col = tmp_col + '_Bin'
            bin_features_name.append(tmp_bin_col)
            tot_cutoff_points[tmp_col] = tmp_cutoff_points
            all_data.loc[:, tmp_bin_col] = all_data[tmp_col].apply(lambda x: AssignBinNumeric(x, tmp_cutoff_points))

    # final_cols = ['date', 'Y']
    # final_cols.extend(bin_features_name)
    # all_bins_data = all_data[final_cols]

    all_data = all_data.drop(cols_to_processed, axis=1)  # drop original columns
    all_data = all_data.drop('bad', axis=1)
    all_data, tot_bin_cols = binsOneHotEncoding(all_data, bin_features_name)

    return all_data, tot_bin_cols, tot_cutoff_points

def Ypoint2Set(data, set_length):
    data = data.astype('str')
    all_y_points = data.tolist()
    Y = [all_y_points[tmp_idx:tmp_idx+set_length] for tmp_idx in range(len(all_y_points) - set_length + 1)]
    return Y



def model_training(Y_train, output_path, training_start_date, training_end_date, chain_len):
    X_train = loadX(training_start_date, training_end_date)
    tmp_columns = X_train.columns.tolist()
    tmp_columns.remove('date')

    all_data = X_train.merge(Y_train, on='date', how='inner')

    cols_to_process = ['OUTSTANDING_CASH_TO_FREE_CAP']
    all_data, tot_bin_cols, tot_cutoff_points = getDiscreteFeatures(all_data, cols_to_process)
    tmp_columns = [x for x in tmp_columns if x not in cols_to_process]
    tmp_columns = tmp_columns + tot_bin_cols

    X_train = all_data[tmp_columns]
    Y_train = all_data['Y']
    del all_data
    gc.collect()

    X_train = Xpoint2Set(X_train, chain_len)
    y_train = Ypoint2Set(Y_train, chain_len)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    with open(output_path + 'crf_model.pkl', 'wb') as tmp_fo:  # dump model
        pickle.dump(crf, tmp_fo)

def model_testing(Y_test, output_path, testing_start_date, testing_end_date, chain_len):
    X_test = loadX(testing_start_date, testing_end_date)
    tmp_columns = X_test.columns.tolist()
    tmp_columns.remove('date')

    all_data = X_test.merge(Y_test, on='date', how='inner')
    X_test = all_data[tmp_columns]
    Y_test = all_data['Y']
    test_dates = all_data['date']
    del all_data
    gc.collect()

    X_test = Xpoint2Set(X_test, chain_len)
    Y_test_pair = Ypoint2Set(Y_test, chain_len)

    with open(output_path + 'crf_model.pkl', 'rb') as tmp_fi:  # dump model
        crf = pickle.load(tmp_fi)

    y_pred = crf.predict(X_test)

    # test pair
    labels = ['-1.0', '1.0']
    print(metrics.flat_classification_report(Y_test_pair, y_pred, labels=labels, digits=3))

    # test single
    y_pred_single = y_pred[0].copy()
    y_pred_single.pop(-1)
    y_pred_single.extend([tmp_y[1] for tmp_y in y_pred])
    # y_pred_single.insert(0, y_pred[0][0])
    y_real_singel = Y_test.astype('str').tolist()
    prsc = precision_score(y_real_singel, y_pred_single, labels=labels, average='micro')
    print('%s to %s weighted precision: %f' % (testing_start_date, testing_end_date, prsc))

    prediction = pd.DataFrame(test_dates)
    prediction.loc[:, 'predict'] = y_pred_single

    return prediction

def train_and_test(tot_test_Y, training_start_date, training_end_date, testing_start_date, testing_end_date,
                   chg_pct, chg_threshold, chain_len, folder_path):
    # load training Y
    # chg_pct = 0.2
    # chg_threshold = 0.15
    train_Y = loadY(chg_pct, chg_threshold, training_start_date, training_end_date)
    # tot_Y = tot_Y.rename(columns={'PeakTrough': 'Y'})
    train_Y.loc[:, 'Y'] = train_Y['PeakTrough'].shift(-1)  # predict tomorrow !!!!
    train_Y = train_Y.loc[~train_Y['Y'].isnull()]  # drop nan
    # train_Y = tot_Y.loc[(tot_Y['date'] >= training_start_date) & (tot_Y['date'] <= training_end_date)]
    # test_Y = tot_Y.loc[(tot_Y['date'] >= testing_start_date) & (tot_Y['date'] <= testing_end_date)]

    model_training(train_Y, folder_path, training_start_date, training_end_date, chain_len)

    # get testing Y
    test_Y = tot_test_Y.loc[(tot_test_Y['date'] >= testing_start_date) & (tot_test_Y['date'] <= testing_end_date)]
    prediction = model_testing(test_Y, folder_path, testing_start_date, testing_end_date, chain_len)

    return prediction


if __name__ == '__main__':
    tot_start_date = '2007-01-01'
    tot_end_date = '2018-08-31'

    folder_path = 'D:/FeatureAlgorithm/Timing/'

    # load testing Y
    chg_pct = 0.2
    chg_threshold = 0.15
    tot_test_Y = loadY(chg_pct, chg_threshold, tot_start_date, tot_end_date)
    tot_test_Y.loc[:, 'Y'] = tot_test_Y['PeakTrough'].shift(-1)  #  predict tomorrow !!!!
    tot_test_Y = tot_test_Y.loc[~tot_test_Y['Y'].isnull()]

    # loop over seasons
    training_duration = 3
    Years = list(range(2013 - training_duration, 2018 + 1 - training_duration))
    Seasons = list(range(1, 5))

    season_start_dates = {
        1: '-01-01',
        2: '-04-01',
        3: '-07-01',
        4: '-10-01'
    }

    season_end_dates = {
        1: '-03-31',
        2: '-06-30',
        3: '-09-30',
        4: '-12-31'
    }

    chain_len = 3
    total_prediction = pd.DataFrame([])
    for tmp_y in Years:
        for tmp_s in Seasons:
            if (tmp_y == 2018 - training_duration) and (tmp_s == 3):
                break

            print('%d S%d' % (tmp_y + training_duration, tmp_s))

            training_start_date = '%d%s' % (tmp_y, season_start_dates[tmp_s])
            training_end_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            training_end_date = datetime.strftime(datetime.strptime(training_end_date, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d') # less one day of the train period

            testing_start_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            testing_end_date = '%d%s' % (tmp_y + training_duration, season_end_dates[tmp_s])

            # train and test
            tmp_prediction = train_and_test(tot_test_Y, training_start_date, training_end_date, testing_start_date, testing_end_date,
                           chg_pct, chg_threshold, chain_len, folder_path)

            # store seasonal prediction
            total_prediction = total_prediction.append(tmp_prediction)

    total_prediction.to_csv(folder_path + 'timing_prediction.csv', index=False)