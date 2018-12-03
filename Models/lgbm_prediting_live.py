#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:30:25 2018

@author: lipchiz
"""

import sys
import os

# sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
# sys.path.append("/home/nyuser/zlrmodeltest/datafetch2")
sys.path.append("/data2/jianghan/FeatureAlgorithm/Models")
# sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch")
# sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch2")
from loadStationaryDataFromDB import loadData, ConfigQuant
from loadStationaryDataFromDBLive import loadData as loadDataLive
from sqlalchemy import create_engine
# from fetchDataFromDB import loadData
import pandas as pd
import numpy as np
import lightgbm as lgb
import pandas.io.sql as sql
from sklearn.externals import joblib
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_curve, classification_report, auc, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe
from datetime import datetime, timedelta
import pymysql
import gc
import pickle


def yload(starttime, endtime):
    connection = {'host': '10.46.228.175', 'port': 3306, 'user': 'alg',
                  'passwd': 'Alg#824', 'db': 'quant', 'charset': 'utf8'}
    sql_order = "select * from STOCK_TOP_BOTTOM_Y where date>='%s' and date<='%s'" % (starttime, endtime)
    con = pymysql.connect(**connection)
    y = sql.read_sql(sql_order, con)
    return y

def writeDB(table_name, prediction_data, sql_engine):
    sql_conn = sql_engine.connect()

    tmp_date = prediction_data.unique()
    if tmp_date.size > 1:
        print('Prediction data contain more than 1 day')
        raise ValueError
    else:
        tmp_date = tmp_date[0]
    sql_statement = "select count(1) as record_num in `%s` where date='%s'" % (table_name, tmp_date)
    record_num = pd.read_sql(sql_statement, sql_conn)
    record_num = record_num.loc[0, 0]

    if record_num > 0:
        print('data in db already exist')
    elif record_num == 0:
        prediction_data.to_sql(table_name, sql_conn, index=False, if_exists='append')

    sql_conn.close()

def predictComplete(predict_path, backup_path):
    y_days = [2, 5, 10, 20]
    result_table_name = 'LGBM_LIVE_PREDICTION_FINAL'
    '__________________________________prediction___________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('predicting has been started')
    print(nowtime)

    # find the latest trade day
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    sql_statement = "select max(`date`) from %s where `date` < '%s'" % ('TRADE_CALENDAR', today)
    sql_conn = sql_engine.connect()
    yesterday = pd.read_sql(sql_statement, sql_conn)
    sql_conn.close()
    yesterday = yesterday.iloc[0, 0]

    # read features
    test_x = loadData(yesterday, yesterday)

    with open('%s/xnamelist.pcl' % predict_path, 'rb') as tmp_fo:  # load feature name
        xnamelist = pickle.load(tmp_fo)

    resultscoredf_h = pd.DataFrame([])
    all_y_scores = test_x[['date', 'code']].copy()
    test_x = test_x[xnamelist] # select  features used in model training
    for tmp_day in y_days:
        y_name = 'Y_%dD' % tmp_day
        clf = joblib.load('%s/model_%s.m' % (predict_path, y_name))
        y_score = clf.predict(test_x)

        # 生成回测数据
        all_y_scores.loc[:, y_name] = y_score

    # write prediction in local folder
    all_y_scores.to_csv('%s/stockscore_%s.csv' % (backup_path, yesterday))
    # write prediction to database
    writeDB(result_table_name, all_y_scores, sql_engine)

    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has finished')
    print(nowtime)


def predictNewest(predict_path, backup_path):
    y_days = [2, 5, 10, 20]
    result_table_name = 'LGBM_LIVE_PREDICTION_TEMP'
    '__________________________________prediction___________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('predicting has been started')
    print(nowtime)

    # find the latest trade day
    today = datetime.strftime(datetime.now(), '%Y-%m-%d')
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    sql_statement = "select max(`date`) from %s where `date` < '%s'" % ('TRADE_CALENDAR', today)
    sql_conn = sql_engine.connect()
    yesterday = pd.read_sql(sql_statement, sql_conn)
    sql_conn.close()
    yesterday = yesterday.iloc[0, 0]

    # load newest features
    test_x = loadDataLive()

    with open('%s/xnamelist.pcl' % predict_path, 'rb') as tmp_fo:  # load feature name
        xnamelist = pickle.load(tmp_fo)

    resultscoredf_h = pd.DataFrame([])
    all_y_scores = test_x[['date', 'code']].copy()
    test_x = test_x[xnamelist] # select  features used in model training
    for tmp_day in y_days:
        y_name = 'Y_%dD' % tmp_day
        clf = joblib.load('%s/model_%s.m' % (predict_path, y_name))
        y_score = clf.predict(test_x)

        # 生成回测数据
        all_y_scores.loc[:, y_name] = y_score

    # write prediction in local folder
    all_y_scores.to_csv('%s/stockscore_tmp_%s.csv' % (backup_path, yesterday))
    # write prediction to database
    writeDB(result_table_name, all_y_scores, sql_engine)

    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has finished')
    print(nowtime)


if __name__ == '__main__':
    backup_path = '/data2/jianghan/FeatureAlgorithm/lgbm_live/prediction'
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    predict_path = '/data2/jianghan/FeatureAlgorithm/lgbm_live'

    predictNewest(predict_path, backup_path)
