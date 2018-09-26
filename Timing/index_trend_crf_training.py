from sklearn.metrics import recall_score, precision_score
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RandomizedSearchCV
import pickle
import pandas as pd
from datetime import datetime, timedelta

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

def Ypoint2Set(data, set_length):
    data = data.astype('str')
    all_y_points = data.tolist()
    Y = [all_y_points[tmp_idx:tmp_idx+set_length] for tmp_idx in range(len(all_y_points) - set_length + 1)]
    return Y

def dataFillNA(data):
    data = data.fillna(method='ffill')  # forward fill
    for tmp_col in data.columns:
        if (data[tmp_col].isnull().sum() > 0) and (data[tmp_col].dtypes == 'float'):
            tmp_mean = data[tmp_col].mean()
            data.loc[:, tmp_col] = data[tmp_col].fillna(tmp_mean)  # still nan, fill with mean

    return data

def model_training(Y_train, output_path, training_start_date, training_end_date, chain_len):
    X_train = loadX(training_start_date, training_end_date)
    X_train = dataFillNA(X_train)   # fill na
    tmp_columns = X_train.columns.tolist()
    tmp_columns.remove('date')

    all_data = X_train.merge(Y_train, on='date', how='inner')
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
    X_test = dataFillNA(X_test)  # fill na
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
    print('f1 score: %f, precision: %f' % (metrics.flat_f1_score(Y_test_pair, y_pred, labels=labels, average='weighted'),
                                           metrics.flat_precision_score(Y_test_pair, y_pred, labels=labels, average='micro')))

    prediction = pd.DataFrame(test_dates)
    prediction.loc[:, 'predict'] = y_pred_single

    return prediction, prsc

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
    prediction, prsc = model_testing(test_Y, folder_path, testing_start_date, testing_end_date, chain_len)

    return prediction, prsc


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
            # training_start_date = '2010-01-01'
            training_end_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            training_end_date = datetime.strftime(datetime.strptime(training_end_date, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d') # less one day of the train period

            testing_start_date = '%d%s' % (tmp_y + training_duration, season_start_dates[tmp_s])
            testing_end_date = '%d%s' % (tmp_y + training_duration, season_end_dates[tmp_s])

            # train and test
            tmp_prediction, tmp_prsc = train_and_test(tot_test_Y, training_start_date, training_end_date, testing_start_date, testing_end_date,
                           chg_pct, chg_threshold, chain_len, folder_path)

            # store seasonal prediction
            total_prediction = total_prediction.append(tmp_prediction)

    total_prediction.to_csv(folder_path + 'timing_prediction.csv', index=False)