#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:30:25 2018

@author: lipchiz
"""

import sys
#sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
# sys.path.append("/home/nyuser/zlrmodeltest/datafetch2")
sys.path.append("/data2/jianghan/FeatureAlgorithm/Tools")
#sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch")
#sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch2")
from loadSmallDataFromDB import loadData
#from fetchDataFromDB import loadData
import pandas as pd
import numpy as np
import lightgbm as lgb
import pandas.io.sql as sql
from sklearn.externals import joblib
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe
from datetime import datetime,timedelta
import pymysql
import gc

def yload(starttime,endtime):
    connection = {'host':'10.46.228.175', 'port':3306, 'user':'alg', 
       'passwd':'Alg#824', 'db':'quant', 'charset':'utf8'}
    sql_order = "select * from STOCK_TOP_BOTTOM_Y where date>='%s' and date<='%s'"%(starttime,endtime)
    con=pymysql.connect(**connection)
    y = sql.read_sql(sql_order,con)
    return y
def run(year,season, output_path, predict_path):
    y_days = [2, 5, 10, 20]
    # ------------------  setting --------------------------------
    season_start_date ={
    1: '-01-01',
    2: '-04-01',
    3: '-07-01',
    4: '-10-01'
    }
    season_end_date = {
        1: '-03-31',
        2: '-06-30',
        3: '-09-30',
        4: '-12-31'
    }
    
    starttime_train = str(year) + season_start_date[season]
    endtime_train = str(year+1) + season_start_date[season]
    endtime_train = datetime.strftime(datetime.strptime(endtime_train, '%Y-%m-%d') - timedelta(days=30), '%Y-%m-%d') # minus 30 days to avoid usage of future data
    
    starttime_test = str(year+1) + season_start_date[season]
    endtime_test = str(year+1) + season_end_date[season]
    
    #def the training function and testing function
    #trainig function
    #按需要更改模型名称
    global train_x,train_y,val_x,val_y,train_data,val_data

    # ============== objective function =============
    def objective(args):
        params = {
            'task':'train',
            'num_threads':45,
            'objective':'binary',
            'boosting':'dart',
            'verbosity':-1,
            'tree_learner':'data',
            'seed':66,
            'min_data_in_leaf':200,
            'metric':'auc',
            'max_depth': args['max_depth'] + 6,
            'learning_rate': args['learning_rate'],
            'feature_fraction': args['feature_fraction'],
            'bagging_fraction': args['bagging_fraction'],
            'num_leaves':np.math.floor(2**(args['max_depth'] + 6)*0.7)
        }
        clf = lgb.train(params,train_data,num_boost_round=1000000,
                        valid_sets=[train_data,val_data],valid_names=['train','val'],
                        early_stopping_rounds=20,verbose_eval=1000)
        
        y_score = clf.predict(val_x)
        fpr,tpr,threshods = roc_curve(val_y,y_score,pos_label = 1)
        aucscore = auc(fpr,tpr)
        return -aucscore
    # ==========================================
    # ============= optimization parameter space ===============
    params_space = {
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.15),
        'max_depth': hp.randint('max_depth', 10),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 0.9),
        'feature_fraction': hp.uniform('feature_fraction', 0.7, 0.9),
    }
    # ==========================================

    x = loadData(starttime_train, endtime_train)
    y = yload(starttime_train, endtime_train)
    for tmp_day in y_days:
        y_name = 'Y_%dD' % tmp_day
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(y_name)
        print('training has started.')
        print(nowtime)

        tmp_y = y[['code', 'date', y_name]]
        tmp_data = pd.merge(x, tmp_y, on=['date', 'code'], how='inner')
        tmp_data.dropna(subset=[y_name], inplace=True)
        tmp_y = tmp_data[y_name]
        tmp_x = tmp_data.drop(['code', 'date', y_name], axis=1)
        train_x, val_x, train_y, val_y = train_test_split(tmp_x, tmp_y, test_size=0.1, random_state=68)
        del tmp_data, tmp_x, tmp_y
        gc.collect()
        train_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(val_x, label=val_y, reference=train_data)
    
        #可以调整max_evals来进行更多的尝试
        best_sln = fmin(objective, space=params_space, algo=tpe.suggest, max_evals=15)

        params = {
            'task':'train',
            'num_threads':45,
            'objective':'binary',
            'boosting':'dart',
            'verbosity':-1,
            'tree_learner':'data',
            'seed':66,
            'min_data_in_leaf':200,
            'metric':'auc',
            'learning_rate': best_sln['learning_rate'],
            'feature_fraction': best_sln['feature_fraction'],
            'max_depth': best_sln['max_depth'] + 6,
            'bagging_fraction': best_sln['bagging_fraction'],
            'num_leaves':np.math.floor(2**(best_sln['max_depth'] + 6)*0.7),
        }
    
        clf = lgb.train(params,train_data,num_boost_round=10000000,valid_sets=[train_data,val_data],
                        valid_names=['train','val'],early_stopping_rounds=20,verbose_eval=1000)

        joblib.dump(clf,'%s/model_%s_%s_%s.m'%(output_path, year+1,season,y_name))
        importance = pd.DataFrame({'feature':clf.feature_name(),'importance':clf.feature_importance('gain')})
        importance.to_excel('%s/feature_importance_%s_%s_%s.xlsx'%(output_path, year+1,season,y_name))
    
        nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('training has finished')
        print(nowtime)
        del train_x, train_y, val_x, val_y
        gc.collect()

    del x, y
    gc.collect()
    '_____________________________________________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has been started')
    print(nowtime)
    #生成下一季度测试结果
    test_x = loadData(starttime_test,endtime_test)
    test_y = yload(starttime_test,endtime_test)

    all_y_scores = test_x[['date', 'code']].copy()
    for tmp_day in y_days:
        y_name = 'Y_%dD' % tmp_day
        tmp_y = test_y[['code','date',y_name]]
        tmp_data = pd.merge(test_x,tmp_y,on=['date','code'],how='inner')
        values={y_name:int(0)}
        tmp_data.fillna(value=values,inplace=True)
        tmp_data.reindex()
        stock_index = tmp_data[['date','code']]
        tmp_y = tmp_data[y_name]
        tmp_x = tmp_data.drop(['code','date',y_name],axis=1)
        clf = joblib.load('%s/model_%s_%s_%s.m'%(output_path, year+1,season,y_name))
        y_score = clf.predict(tmp_x)
        y_boundle = pd.DataFrame({'proba':y_score,'real':tmp_y})
        y_boundle.sort_values(by='proba',ascending=False,inplace=True)
        y_boundle.reindex()
        tmp_list = np.repeat(np.nan,len(y_boundle))
        tmp_list[:int(np.floor(len(y_boundle)/100))]=1
        tmp_list[int(np.floor(len(y_boundle)/100)):]=0

        y_boundle['predict'] = tmp_list
        accuracyscore = accuracy_score(y_boundle['real'], y_boundle['predict'])
        fpr,tpr,threshods = roc_curve(y_boundle['real'],y_score,pos_label = 1)
        ks = np.max(np.abs(tpr-fpr))
        aucscore = auc(fpr,tpr)
        precision = precision_score(y_boundle['real'],y_boundle['predict'],average='binary')
        recall = recall_score(y_boundle['real'],y_boundle['predict'],average='weighted')
        print('___________________________________________________________________')
        print('%s_%s_%s'%(year,season,y_name))
        print('precision:',precision)
        print('recall:',recall)
        print('auc:',aucscore)
        print('accuracyscore:',accuracyscore)
        print('K-S:',ks)
        print(classification_report(y_boundle['real'],y_boundle['predict']))
        print(confusion_matrix(y_boundle['real'],y_boundle['predict']))
        print('___________________________________________________________________')

        #生成回测数据
        y_score = pd.DataFrame(y_score,columns=['proba_1_%dD' % tmp_day])
        stock_index = pd.concat([stock_index,y_score],axis=1)
        all_y_scores = all_y_scores.merge(stock_index, on=['date', 'code'], how='left')

    all_y_scores.to_csv('%s/stockscore_%ds%d.csv'%(predict_path, year,season))

    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has finished')
    print(nowtime)

if __name__ == '__main__':
    year = 2012
    season = 1
    # for y in ['Y_2D','Y_5D','Y_10D','Y_20D']:
    #     for season in range(1,5):
    output_path = '/data2/jianghan/FeatureAlgorithm/lgbm_model_results/2012_s1'
    predict_path = '/data2/jianghan/FeatureAlgorithm/lgbm_model_results/2012_s1'
    run(year,season, output_path, predict_path)
