#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:30:25 2018

@author: lipchiz
"""

import sys
#sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
sys.path.append("/home/nyuser/zlrmodeltest/datafetch2")
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
def run(year,season,yname = 'Y_20D'):
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
    endtime_train = datetime.strftime(datetime.strptime(endtime_train, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d')

    starttime_test = str(year+1) + season_start_date[season]
    endtime_test = str(year+1) + season_end_date[season]
    
    #def the training function and testing function
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('training has started.')
    print(nowtime)
    #trainig function
    #按需要更改模型名称
    global train_x,train_y,val_x,val_y,train_data,val_data
    
    x = loadData(starttime_train,endtime_train)
    y = yload(starttime_train,endtime_train)
    y = y[['code','date',yname]]
    tmp_data = pd.merge(x,y,on=['date','code'],how='inner')
    tmp_data.dropna(subset=[yname],inplace=True)
    y = tmp_data[yname]
    x = tmp_data.drop(['code','date',yname],axis=1)
    train_x,val_x,train_y,val_y = train_test_split(x,y,test_size=0.1,random_state=68)
    del tmp_data,x,y
    gc.collect()
    train_data = lgb.Dataset(train_x,label=train_y)
    val_data = lgb.Dataset(val_x,label=val_y,reference=train_data)
    
    
    def objective(args):
        params = {
            'task':'train',
            'num_threads':15,
            'objective':'binary',
            'boosting':'dart',
            'verbose':-1,
            'tree_learner':'data',
            'seed':66,
            'min_data_in_leaf':300,
            'metric':'auc',
            'max_depth': args['max_depth'] + 6,
            'learning_rate': args['learning_rate'],
            'feature_fraction': args['feature_fraction'],
            'bagging_fraction': args['bagging_fraction'],
            'num_leaves':np.math.floor(2**(args['max_depth'] + 6)/2)
        }
        clf = lgb.train(params,train_data,num_boost_round=1000000,
                        valid_sets=[train_data,val_data],valid_names=['train','val'],
                        early_stopping_rounds=20,verbose_eval=100)
        
        y_score = clf.predict(val_x)
        fpr,tpr,threshods = roc_curve(val_y,y_score,pos_label = 1)
        aucscore = auc(fpr,tpr)
        return -aucscore
    
    params_space = {
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.15),
        'max_depth': hp.randint('max_depth',10),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 0.9),
        'feature_fraction': hp.uniform('feature_fraction', 0.7, 0.9),
    }
    
    #可以调整max_evals来进行更多的尝试
    best_sln = fmin(objective, space=params_space, algo=tpe.suggest, max_evals=20)
    
    params = {
        'task':'train',
        'num_threads':15,
        'objective':'binary',
        'boosting':'dart',
        'verbose':0,
        'tree_learner':'data',
        'seed':66,
        'min_data_in_leaf':300,
        'metric':'auc',
        'learning_rate': best_sln['learning_rate'],
        'feature_fraction': best_sln['feature_fraction'],
        'max_depth': best_sln['max_depth'] + 6,
        'bagging_fraction': best_sln['bagging_fraction'],
        'num_leaves':np.math.floor(2**(best_sln['max_depth'] + 6)/2),
    }
    
    clf = lgb.train(params,train_data,num_boost_round=1000000,valid_sets=[train_data,val_data],
                    valid_names=['train','val'],early_stopping_rounds=20,verbose_eval=100)
    
    joblib.dump(clf,'model_%s_%s.m'%(year+1,season))
    importance = pd.DataFrame({'feature':clf.feature_name(),'importance':clf.feature_importance('gain')})
    importance.to_excel('feature_importance_%s_%s.xlsx'%(year+1,season))
    
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('training has finished')
    print(nowtime)
    
    '_____________________________________________________________________________'
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has been started')
    print(nowtime)
    #生成下一季度测试结果
    test_x = loadData(starttime_test,endtime_test)
    test_y = yload(starttime_test,endtime_test)
    test_y = test_y[['code','date',yname]]
    tmp_data = pd.merge(test_x,test_y,on=['date','code'],how='inner')
    values={yname:int(0)}
    tmp_data.fillna(value=values,inplace=True)
    tmp_data.reindex()
    stock_index = tmp_data[['date','code']]
    test_y = tmp_data[yname]
    test_x = tmp_data.drop(['code','date',yname],axis=1)
    clf = joblib.load('model_%s_%s.m'%(year+1,season))
    y_score = clf.predict(test_x)
    y_predict = np.int64(y_score>0.9)
    accuracyscore = accuracy_score(test_y, y_predict)
    fpr,tpr,threshods = roc_curve(test_y,y_score,pos_label = 1)
    ks = np.max(np.abs(tpr-fpr))
    aucscore = auc(fpr,tpr)
    precision = precision_score(test_y,y_predict,average='binary')
    recall = recall_score(test_y,y_predict,average='weighted')
    print('precision:',precision)
    print('recall:',recall)
    print('auc:',aucscore)
    print('accuracyscore:',accuracyscore)
    print('K-S:',ks)
    print(classification_report(test_y,y_predict))
    print(confusion_matrix(test_y,y_predict))
    
    #生成回测数据
    y_score = pd.DataFrame(y_score,columns=['proba_1']) 
    stock_index = pd.concat([stock_index,y_score],axis=1)
    stock_index.to_csv('backtest_data_%s_%s.csv'%(year+1,season))
    
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('testing has finished')
    print(nowtime)

if __name__ == '__main__':
    year = 2012
    for season in range(1,5):
        run(year,season)