from sklearn.metrics import recall_score, precision_score, roc_curve, auc
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hyperopt import fmin, hp, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.externals import joblib
import os

import sklearn_crfsuite
# from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import lightgbm as lgb
import gc
import scipy.stats
from sklearn.model_selection import train_test_split

from calStockIndexPeakTrough import calFull as loadY
from loadContinuousTimingDataFromDB import loadData as loadX

# def objective(X_train, chg_pct, chg_threshold, chain_len, training_start_date, training_end_date):
def objective(params):
    global X_train, sub_train_data, sub_val_data, sub_val_x, sub_val_y
    # X_train = params['X_train']
    chg_pct = params['chg_pct']
    chg_threshold = params['chg_threshold']
    training_start_date = params['training_start_date']
    training_end_date = params['training_end_date']

    Y_train = loadY(chg_pct, chg_threshold, training_start_date, training_end_date)
    if Y_train is None:  # failed to load Y
        return {'loss':9999, 'status':STATUS_FAIL, 'learning_rate': np.nan, 'max_depth': np.nan,
                'bagging_fraction': np.nan, 'feature_fraction':np.nan}

    Y_train.loc[:, 'Y'] = Y_train['PeakTrough'].shift(-1)  # predict tomorrow !!!!
    Y_train = Y_train.loc[~Y_train['Y'].isnull()]  # drop nan
    Y_train.loc[:, 'Y'] = Y_train['Y'].replace({-1:0})

    tmp_columns = X_train.columns.tolist()
    tmp_columns.remove('date')

    all_data = X_train.merge(Y_train, on='date', how='inner')
    sub_whole_x = all_data[tmp_columns]
    sub_whole_y = all_data['Y']
    del all_data
    gc.collect()

    # sub optimization for params of light-gbm
    sub_train_x, sub_val_x, sub_train_y, sub_val_y = train_test_split(sub_whole_x, sub_whole_y, test_size=0.1, random_state=68)
    sub_train_data = lgb.Dataset(sub_train_x, label=sub_train_y)
    sub_val_data = lgb.Dataset(sub_val_x, label=sub_val_y, reference=sub_train_data)

    sub_params_space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
        'max_depth': hp.randint('max_depth', 10),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 0.9),
        'feature_fraction': hp.uniform('feature_fraction', 0.1, 0.9),
    }

    tmp_sub_trails = Trials()
    best_sub_params = fmin(subObjective, space=sub_params_space, algo=tpe.suggest, max_evals=100, trials=tmp_sub_trails)

    # get boost round
    tmp_idx = np.argmin(np.array(tmp_sub_trails.losses()))
    best_boost_round_num = tmp_sub_trails.results[tmp_idx]['iteration']
    print('best sub cv score:', tmp_sub_trails.results[tmp_idx]['loss'])
    print('best num of round:', best_boost_round_num)

    params = {
        'task': 'train',
        'num_threads': 45,
        'objective': 'binary',
        'boosting': 'dart',
        'verbosity': -1,
        'tree_learner': 'data',
        'seed': 66,
        'min_data_in_leaf': 200,
        'metric': 'auc',
        'learning_rate': best_sub_params['learning_rate'],
        'feature_fraction': best_sub_params['feature_fraction'],
        'max_depth': best_sub_params['max_depth'] + 6,
        'bagging_fraction': best_sub_params['bagging_fraction'],
        'num_leaves': np.math.floor(2 ** (best_sub_params['max_depth'] + 6) * 0.7),
    }

    whole_data = lgb.Dataset(sub_whole_x, label=sub_whole_y)
    clf = lgb.train(params, whole_data, num_boost_round=best_boost_round_num, verbose_eval=1000)

    # calculate predict P&L
    y_pred = clf.predict(sub_whole_x, num_iteration=best_boost_round_num)
    Y_train.loc[:, 'predict'] = y_pred
    Y_train.loc[:, 'predict'] = Y_train['predict'].shift(1)  # predict tomorrow
    tmp_buy = Y_train.loc[Y_train['predict'] > 0.5]
    tmp_buy.loc[:, 'cum_ret'] = tmp_buy['ret'].cumprod()
    final_pnl = tmp_buy['cum_ret'].iloc[-1]

    # best_cv_score = rs_cv.best_score_

    obj_result = {'loss':-final_pnl, 'status':STATUS_OK, 'best_boost_round_num': best_boost_round_num}
    obj_result.update(best_sub_params)  # subObjective parmas

    return obj_result
    # return (best_sub_params, best_cv_score)

def subObjective(args):
    global sub_train_data, sub_val_data, sub_val_x, sub_val_y

    params = {
        'task': 'train',
        'num_threads': 45,
        'objective': 'binary',
        'boosting': 'dart',
        'verbosity': -1,
        'tree_learner': 'data',
        'seed': 66,
        'min_data_in_leaf': 200,
        'metric': 'auc',
        'max_depth': args['max_depth'] + 6,
        'learning_rate': args['learning_rate'],
        'feature_fraction': args['feature_fraction'],
        'bagging_fraction': args['bagging_fraction'],
        'num_leaves': np.math.floor(2 ** (args['max_depth'] + 6) * 0.7)
    }

    clf = lgb.train(params, sub_train_data, num_boost_round=1000000,
                    valid_sets=[sub_train_data, sub_val_data], valid_names=['train', 'val'],
                    early_stopping_rounds=15, verbose_eval=1000)

    y_score = clf.predict(sub_val_x, num_iteration=clf.best_iteration)
    fpr, tpr, threshods = roc_curve(sub_val_y, y_score, pos_label=1)
    aucscore = auc(fpr, tpr)
    return {'loss':-aucscore, 'status':STATUS_OK, 'iteration': clf.best_iteration}


def model_training(output_path, training_start_date, training_end_date, year, season):
    global  X_train
    X_train = loadX(training_start_date, training_end_date)

    max_nan_rate = 0.7
    nan_rate = X_train.isnull().sum(axis=0) / X_train.shape[0]
    cols_to_drop = nan_rate[nan_rate > max_nan_rate].index.tolist()
    if len(cols_to_drop) > 0:
        print('drop nan columns:', cols_to_drop)
        X_train = X_train.drop(cols_to_drop, axis=1)

    # ==== hyperopt, outer optimization for determining Y
    params = {
        'chg_pct': hp.uniform('chg_pct', 0.05, 0.3),
        'chg_threshold': hp.uniform('chg_threshold', 0.05, 0.3),
        'training_start_date': training_start_date,
        'training_end_date': training_end_date
    }

    tmp_trial = Trials()
    best_params = fmin(objective, space=params, algo=tpe.suggest, max_evals=50, trials=tmp_trial)

    # get sub-params
    tmp_idx = np.argmin(np.array(tmp_trial.losses()))
    best_params['learning_rate'] = tmp_trial.results[tmp_idx]['learning_rate']
    best_params['feature_fraction'] = tmp_trial.results[tmp_idx]['feature_fraction']
    best_params['max_depth'] = tmp_trial.results[tmp_idx]['max_depth']
    best_params['bagging_fraction'] = tmp_trial.results[tmp_idx]['bagging_fraction']
    best_params['boost_round_num'] = tmp_trial.results[tmp_idx]['best_boost_round_num']

    print('best cv score:', tmp_trial.results[tmp_idx]['loss'])
    print('best params:', best_params)

    # ==== train with the best params (final)
    Y_train = loadY(best_params['chg_pct'], best_params['chg_threshold'], training_start_date, training_end_date)
    Y_train.loc[:, 'Y'] = Y_train['PeakTrough'].shift(-1)  # predict tomorrow !!!!
    Y_train = Y_train.loc[~Y_train['Y'].isnull()]  # drop nan
    Y_train.loc[:, 'Y'] = Y_train['Y'].replace({-1:0})

    tmp_columns = X_train.columns.tolist()
    tmp_columns.remove('date')

    with open('%sx_name_list_%d_%s.pkl' % (folder_path, year, season), 'wb') as tmp_fo:  # record columns used in training
        pickle.dump(tmp_columns, tmp_fo)

    all_data = X_train.merge(Y_train, on='date', how='inner')
    X_train = all_data[tmp_columns]
    Y_train = all_data['Y']

    params = {
        'task': 'train',
        'num_threads': 45,
        'objective': 'binary',
        'boosting': 'dart',
        'verbosity': -1,
        'tree_learner': 'data',
        'seed': 66,
        'min_data_in_leaf': 200,
        'metric': 'auc',
        'learning_rate': best_params['learning_rate'],
        'feature_fraction': best_params['feature_fraction'],
        'max_depth': best_params['max_depth'] + 6,
        'bagging_fraction': best_params['bagging_fraction'],
        'num_leaves': np.math.floor(2 ** (best_params['max_depth'] + 6) * 0.7),
    }

    final_whole_data = lgb.Dataset(X_train, label=Y_train)
    clf = lgb.train(params, final_whole_data, num_boost_round=best_params['boost_round_num'], verbose_eval=1000)

    joblib.dump(clf, '%smodel_%s_%s.m' % (output_path, year, season))
    importance = pd.DataFrame({'feature': clf.feature_name(), 'importance': clf.feature_importance('gain')})  # feature importance
    importance.to_csv('%sfeature_importance_%s_%s.csv' % (output_path, year, season), index=False)

    return best_params

def model_testing(Y_test, output_path, testing_start_date, testing_end_date, boost_round_num, year, season):
    X_test = loadX(testing_start_date, testing_end_date)

    with open('%sx_name_list_%d_%d.pkl' % (folder_path, year, season), 'rb') as tmp_fi:  # load X column names
        x_col_names = pickle.load(tmp_fi)

    all_data = X_test.merge(Y_test, on='date', how='inner')
    X_test = all_data[x_col_names]
    Y_test = all_data['Y']
    test_dates = all_data['date']
    del all_data
    gc.collect()

    clf = joblib.load('%smodel_%s_%s.m' % (output_path, year, season))  # load model

    X_test = X_test.astype('float')  # in case all nan columns have the type 'Object'
    y_pred = clf.predict(X_test, num_iteration=boost_round_num)
    # test pair
    y_pred = np.int64(y_pred > 0.5)
    # print(metrics.flat_classification_report(Y_test, y_pred, labels=[0, 1], digits=3))

    prsc = precision_score(Y_test, y_pred, labels=[0, 1], average='micro')
    print('%s to %s weighted precision: %f' % (testing_start_date, testing_end_date, prsc))

    prediction = pd.DataFrame(test_dates)
    prediction.loc[:, 'predict'] = y_pred

    return prediction, prsc

def train_and_test(training_start_date, training_end_date, testing_start_date, testing_end_date,
                   folder_path, year, season):

    best_params = model_training(folder_path, training_start_date, training_end_date, year, season)

    # get testing Y
    tot_test_Y = loadY(best_params['chg_pct'], best_params['chg_threshold'], tot_start_date, tot_end_date)
    tot_test_Y.loc[:, 'Y'] = tot_test_Y['PeakTrough'].shift(-1)  # predict tomorrow !!!!
    tot_test_Y = tot_test_Y.loc[~tot_test_Y['Y'].isnull()]  # drop nan
    tot_test_Y.loc[:, 'Y'] = tot_test_Y['Y'].replace({-1, 0})

    test_Y = tot_test_Y.loc[(tot_test_Y['date'] >= testing_start_date) & (tot_test_Y['date'] <= testing_end_date)] # trim Y for test dates
    prediction, prsc = model_testing(test_Y, folder_path, testing_start_date, testing_end_date, best_params['boost_round_num'], year, season)

    return prediction, prsc


if __name__ == '__main__':
    tot_start_date = '2007-01-01'
    tot_end_date = '2018-09-30'

    # folder_path = 'D:/FeatureAlgorithm/Timing/'
    folder_path = os.getcwd() + '/'

    # load testing Y
    # chg_pct = 0.2
    # chg_threshold = 0.15
    # tot_test_Y = loadY(chg_pct, chg_threshold, tot_start_date, tot_end_date)
    # tot_test_Y.loc[:, 'Y'] = tot_test_Y['PeakTrough'].shift(-1)  #  predict tomorrow !!!!
    # tot_test_Y = tot_test_Y.loc[~tot_test_Y['Y'].isnull()]

    # loop over seasons
    training_duration = 7
    Years = list(range(2017 - training_duration, 2018 + 1 - training_duration))
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
            if (tmp_y == 2018 - training_duration) and (tmp_s == 4):
                break

            print('%d S%d' % (tmp_y + training_duration, tmp_s))

            training_start_date = '%d%s' % (tmp_y, season_start_dates[tmp_s])
            training_end_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            training_end_date = datetime.strftime(datetime.strptime(training_end_date, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d') # less one day of the train period

            testing_start_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            testing_end_date = '%d%s' % (tmp_y + training_duration, season_end_dates[tmp_s])

            # train and test
            tmp_prediction, tmp_prsc = train_and_test(training_start_date, training_end_date, testing_start_date, testing_end_date,
                            folder_path, tmp_y + training_duration, tmp_s)

            # store seasonal prediction
            total_prediction = total_prediction.append(tmp_prediction)

    total_prediction.to_csv(folder_path + 'lgbm_timing_prediction.csv', index=False)