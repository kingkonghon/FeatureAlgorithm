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
# sys.path.append("/data2/jianghan/FeatureAlgorithm/Models")
sys.path.append(r"F:\FeatureAlgorithm\Live")
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


def getLatestTradeDay(sql_engine, calendar_table_name):
    today = datetime.now()
    today = datetime.strftime(today, '%Y-%m-%d')

    today = '2019-01-09'

    sql_statement = "select `date` from %s" % calendar_table_name
    sql_conn = sql_engine.connect()
    trade_calendar = pd.read_sql(sql_statement, sql_conn)
    sql_conn.close()

    trade_calendar = trade_calendar['date'].values
    yesterday = trade_calendar[trade_calendar < today][-1]  # latest trade day before today

    print('today: %s, yesterday:%s' % (today, yesterday))

    return today, yesterday, trade_calendar

def writeDB(table_name, prediction_data, sql_engine):
    sql_conn = sql_engine.connect()

    tmp_date = prediction_data['date'].unique()
    if tmp_date.size > 1:
        print('Prediction data contain more than 1 day')
        raise ValueError
    elif tmp_date.size == 0:
        print('Empty prediction data')
        raise ValueError
    else:
        tmp_date = tmp_date[0]

    sql_statement = "SELECT count(1) FROM information_schema.tables WHERE table_schema = 'quant' AND table_name = '%s'" % table_name
    tmp_table_exists = pd.read_sql(sql_statement, sql_conn)
    tmp_table_exists = tmp_table_exists.iloc[0, 0]

    prediction_data.loc[:, 'time_stamp'] = datetime.now()  # add timestamp

    if tmp_table_exists == 0:  # table not exists
        prediction_data.to_sql(table_name, sql_conn, index=False, if_exists='replace') # create new table
    elif tmp_table_exists == 1:  # table exists
        sql_statement = "select count(1) as record_num from `%s` where date='%s'" % (table_name, tmp_date)
        record_num = pd.read_sql(sql_statement, sql_conn)
        record_num = record_num.iloc[0, 0]

        if record_num > 0:
            print('data of %s already in %s, replace' % (tmp_date, table_name))
            sql_statement = "delete from `%s` where date='%s'" % (table_name, tmp_date)
            sql_conn.execute(sql_statement)
            print('records has been deleted')
            prediction_data.to_sql(table_name, sql_conn, index=False, if_exists='append')  # add records to existing table
            print('new records has been add to table')
        elif record_num == 0:
            prediction_data.to_sql(table_name, sql_conn, index=False, if_exists='append')  # add records to existing table

    sql_conn.close()

def predictComplete(predict_path, backup_path):
    y_days = [2, 5, 10, 20]
    result_table_name = 'LGBM_LIVE_PREDICTION_FINAL'
    calendar_table_name = 'TRADE_CALENDAR'
    '__________________________________prediction___________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('predicting has been started')
    print(nowtime)

    # find the latest trade day
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    today, yesterday, trade_calendar = getLatestTradeDay(sql_engine, calendar_table_name)

    if today not in trade_calendar:
        print('%s is not trade date' % today)
        return

    # read features
    test_x = loadData(yesterday, yesterday)

    with open('%s/xnamelist.pcl' % predict_path, 'rb') as tmp_fo:  # load feature name
        xnamelist = pickle.load(tmp_fo)

    resultscoredf_h = pd.DataFrame([])
    all_y_scores = test_x[['date', 'code']].copy()
    test_x = test_x[xnamelist] # select  features used in model training

    # check data type of loaded data
    tmp_dtypes = test_x.dtypes
    tmp_dtypes = tmp_dtypes[tmp_dtypes == 'O']
    if tmp_dtypes.size > 0:
        print('data corrupted')
        raise ValueError

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
    print(nowtime)


def predictNewest(predict_path, backup_path):
    y_days = [2, 5, 10, 20]
    result_table_name = 'LGBM_LIVE_PREDICTION_TEMP'
    calendar_table_name = 'TRADE_CALENDAR'
    '__________________________________prediction___________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('predicting has been started')
    print(nowtime)

    # find the latest trade day
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    today, yesterday, trade_calendar = getLatestTradeDay(sql_engine, calendar_table_name)

    if today not in trade_calendar:
        print('%s is not trade date' % today)
        return

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
    # backup_path = '/data2/jianghan/FeatureAlgorithm/Live/prediction'
    # backup_path = r'D:\FeatureAlgorithm\Live\lgbm_live\prediction'
    backup_path = r'F:\FeatureAlgorithm\Live\lgbm_live\prediction'
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    # predict_path = '/data2/jianghan/FeatureAlgorithm/Live/lgbm_live'
    # predict_path = r'D:\FeatureAlgorithm\Live\lgbm_live'
    predict_path = r'F:\FeatureAlgorithm\Live\lgbm_live'

    predictNewest(predict_path, backup_path)
    print('temp prediction completed')
    try:
        predictComplete(predict_path, backup_path)
        print('complete prediction completed')
    except Exception as e:
        print(e)
        print('complete prediction stopped')