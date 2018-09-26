#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:21:04 2018

@author: lipchiz
"""

import sys
# sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
sys.path.append("/home/nyuser/jianghan/FeatureAlgorithm/Tools")
# sys.path.append(r"D:\FeatureAlgorithm\Tools")
#sys.path.append("/home/lipchiz/文档/pythonscripts/quant/datafetch")
# from loadSmallDataFromDB import loadData
from loadSmallDataFromDBV2 import loadData
# from loadDataFromDBVScreenStocks import loadData
import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.externals import joblib
import pandas.io.sql as sql
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
import datetime
import pymysql
import gc
from datetime import datetime, timedelta
import logging
import os
import pickle



def yload(table_name, starttime, endtime):
    connection = {'host': '10.46.228.175', 'port': 3306, 'user': 'alg',
                  'passwd': 'Alg#824', 'db': 'quant', 'charset': 'utf8'}
    sql_order = "select * from %s where date>='%s' and date<='%s'" % (table_name, starttime, endtime)
    con = pymysql.connect(**connection)
    y = sql.read_sql(sql_order, con)
    return y

def objective(args):
    params = {
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 45,
        'learning_rate': args['learning_rate'],
        'colsample_bytree': args['colsample_bytree'],
        'max_depth': args['max_depth'] + 6,
        'subsample': args['subsample']
    }
    xgb1 = xgb.train(params, args['train_data'], evals_result={'eval_metric': 'auc'},
                     num_boost_round=100000, evals=[(args['train_data'], 'train'), (args['val_data'], 'val')],
                     verbose_eval=False, early_stopping_rounds=15)
    y_score = xgb1.predict(args['val_data'])
    #    y_predict = np.int64(y_score>0.5)
    fpr, tpr, threshods = roc_curve(args['val_data'].get_label(), y_score, pos_label=1)
    aucscore = auc(fpr, tpr)
    print('searching auc-score:',aucscore)
    return {'loss':-aucscore, 'status':STATUS_OK, 'iteration': xgb1.best_iteration}

def model_training(day, data, x_list, params_space, logger, out_folder_path):
#   select the right y
#     tmp_yname = 'Y_%sD_%sPCT'%(day,rate)

    tmp_yname = 'Y_%sD'%(day)
    train_list = x_list.copy()
    train_list.append(tmp_yname)
    tmp_data = data[train_list]
    tmp_data = tmp_data.dropna(subset=[tmp_yname])

    y = tmp_data[tmp_yname]
    x = tmp_data.drop(tmp_yname,axis=1)
    # train_x, val_x, train_y, val_y = train_test_split(x,y,test_size=0.1,random_state=68)
    tmp_train_size = int(x.shape[0] * 0.7)
    train_x = x.iloc[:tmp_train_size]
    train_y = y.iloc[:tmp_train_size]
    val_x = x.iloc[tmp_train_size:]
    val_y = y.iloc[tmp_train_size:]
    del tmp_data
    gc.collect()

    # check data
    length_1 = len(y[y==1])
    length_0 = len(y[y==0])

    logger.info('The size of data used for modeling: ' + str(len(train_x)))
    logger.info('The number of y=1 is: ' + str(length_1))
    logger.info('The number of y=0 is: ' + str(length_0))
    logger.info('feature num: %d' % len(x_list))
    logger.info('train data types:')
    logger.info(x.dtypes.unique())
    logger.info(y.unique())


    params_space['train_data'] = xgb.DMatrix(train_x,label=train_y,feature_names=train_x.columns)
    params_space['val_data'] = xgb.DMatrix(val_x,label=val_y,feature_names=val_x.columns)

    tmp_trial = Trials()
    best_sln = fmin(objective, space=params_space, algo=tpe.suggest, max_evals=10, trials=tmp_trial)

    # get best boost rounds
    tmp_idx = np.argmin(np.array(tmp_trial.losses()))
    best_boost_num = tmp_trial.results[tmp_idx]['iteration']

    # train_y = tmp_data[tmp_yname]
    # train_x = tmp_data.drop(tmp_yname,axis=1)

    # del params_space['train_data']
    # del params_space['val_data']

    params = {
        'silent':1 ,                        #设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 45,
        'learning_rate': best_sln['learning_rate'],
        'colsample_bytree': best_sln['colsample_bytree'],
        'max_depth': best_sln['max_depth'] + 6,
        'subsample': best_sln['subsample']
    }


    # re-divide train/validation sets
    # train_x, val_x, train_y, val_y = train_test_split(x,y,test_size=0.1,random_state=68)

    # use all data to train
    xgbdata = xgb.DMatrix(x,label=y,feature_names=train_x.columns)

    # use the num of boost get from early stopping
    clf = xgb.train(params, xgbdata, num_boost_round=best_boost_num)
    fea_imp = clf.get_score(importance_type='gain')
    importance = pd.DataFrame(fea_imp,index=['importance'])
    importance = pd.DataFrame({'feature':importance.columns.tolist(),'importance':importance.values[0].tolist()})
    # importance.to_excel('feature_importance_%sD_%sPCT.xlsx'%(str(day),str(rate)))
    importance.to_excel('%s/feature_importance_%dD.xlsx'%(out_folder_path, day))
#   save the model which has been trained
#     joblib.dump(clf,'train_model_%sD.m'%(str(day),str(rate)))
    clf.save_model('%s/train_model_%dD.m'% (out_folder_path, day))
    report = 'day = %dD'% day
    logger.info(report)
    logger.info('-----------------------------------------------------')

def run(starttime_train, endtime_train, out_folder_path):
    # ------------------  setting --------------------------------

    Y_table_name = 'STOCK_TOP_BOTTOM_Y2'
    Y_days = [2,5,10,20]

    params_space = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 50,
        'learning_rate': hp.uniform("learning_rate", 0.05, 0.15),
        'max_depth': hp.randint('max_depth', 10),
        'subsample': hp.uniform("subsample", 0.5, 0.9),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    }

    # create logger
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='%s/train_live.log' % out_folder_path,
                        filemode='w')

    logger = logging.getLogger('train_live')


    '''---------------------------- training -----------------------------------'''
    # prepare training data
    logger.info('training has been started.')

    tmp_dt_start = datetime.strptime(starttime_train, '%Y-%m-%d')
    tmp_dt_end = datetime.strptime(endtime_train, '%Y-%m-%d')
    tmp_dt_mid = tmp_dt_start + (tmp_dt_end - tmp_dt_start) / 2
    end1_train = datetime.strftime(tmp_dt_mid, '%Y-%m-%d')
    start2_train = datetime.strftime(tmp_dt_mid + timedelta(days=1), '%Y-%m-%d')

    train_x1 = loadData(starttime_train, end1_train)
    train_x2 = loadData(start2_train, endtime_train)
    train_x = train_x1.append(train_x2)
    del train_x1, train_x2
    gc.collect()

    # train_x = loadData(starttime_train,endtime_train)
    train_y = yload(Y_table_name, starttime_train, endtime_train)
    train_y.drop('time_stamp', axis=1, inplace=True)
    xnamelist = train_x.columns.tolist()  # feature names (without code & date)
    xnamelist.remove('code')
    xnamelist.remove('date')
    train_data = pd.merge(train_x, train_y, on=['date', 'code'], how='left')

    del train_x, train_y
    gc.collect()

    # save training feature name
    out_path = '%s/xnamelist.pcl' % out_folder_path
    with open(out_path, 'wb') as out_file:
        pickle.dump(xnamelist, out_file)

    # preprocessing training data
    try:
        train_data.drop_duplicates(['code', 'date'], inplace=True)
        train_data = train_data.sort_values('date', ascending=True)
    except:
        logger.error('train_data error')

    train_data.drop(['date', 'code'], axis=1, inplace=True)  # drop code & date

    # resultscoredf_h = pd.DataFrame()

    #training the model
    # for day in [2,5,10,20,30]:
    for day in Y_days:
        model_training(day, train_data, xnamelist,params_space, logger, out_folder_path)
    #delete all the variables
    del day, params_space, train_data
    gc.collect()

    # write feature list
    out_path = '%s/xnamelist.pcl' % out_folder_path
    with open(out_path, 'wb') as out_file:
        pickle.dump(xnamelist, out_file)

    logger.info('training has finished')


if __name__ == '__main__':
    # result_folder_path = '/home/konghon/Documents/model_results/2012'
    starttime_train = '2017-07-01'
    endtime_train = '2018-06-30'
    result_folder_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results_live'
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    run(starttime_train, endtime_train,result_folder_path)