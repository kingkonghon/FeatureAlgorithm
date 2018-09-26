#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:21:04 2018

@author: lipchiz
"""

import sys
import pickle

# sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
# sys.path.append("/home/konghon/Documents/FeatureAlgorithm/Tools")
# sys.path.append("/home/nyuser/jianghan/FeatureAlgorithm/Tools")
sys.path.append(r"D:\FeatureAlgorithm\Tools")
# sys.path.append("/home/lipchiz/文档/pythonscripts/quant/datafetch")
# from loadSmallDataFromDB import loadData
from loadSmallDataFromDBLive import loadData
import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.externals import joblib
import pandas.io.sql as sql
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_curve, classification_report, auc, \
    confusion_matrix
import datetime
import pymysql
import gc
from datetime import datetime, timedelta
import logging
import os



def run(out_folder_path):
    Y_days = [2, 5, 10, 20]

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='%s/predict_live.log' % out_folder_path,
                        filemode='w')

    logger = logging.getLogger('predict_live')

    feature_list_path = '%s/xnamelist.pcl' % out_folder_path
    with open(feature_list_path, 'rb') as in_file:
        xnamelist = pickle.load(in_file)

    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('backtest data generating.')
    # print(nowtime)

    test_data = loadData()
    try:
        test_data.drop_duplicates(['code', 'date'], inplace=True)
        if 'index' in test_data.columns.tolist():
            test_data = test_data.drop(['index'], axis=1)
        test_data.reindex()
    except:
        logger.error('train_data error')

    stock_index = test_data[['date', 'code']]
    test_data.drop(['date', 'code'], axis=1, inplace=True)
    test_data = test_data[xnamelist] # rearrange columns

    logger.info('features:')
    logger.info(' , '.join(test_data.columns.tolist()))

    test_data = xgb.DMatrix(test_data, feature_names=xnamelist)

    for day in Y_days:
        xgb1 = xgb.Booster()
        xgb1.load_model('%s/train_model_%dD.m' % (out_folder_path, day))
        y_score = xgb1.predict(test_data)
        y_score = pd.DataFrame(y_score, columns=['proba_1_%dD' % day])
        stock_index = pd.concat([stock_index, y_score], axis=1)
        logger.info('day = %sD' % day)
    stock_index.to_csv("%s/stockscore_live.csv" % (out_folder_path), index=False, sep=',')

    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('backtest data has generated. ')
    # print(nowtime)


if __name__ == '__main__':

    train_folder_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results_live'
    if os.path.exists(train_folder_path):
        run(train_folder_path)