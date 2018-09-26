#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:21:04 2018

@author: lipchiz
"""

import sys
#sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
#sys.path.append("/home/nyuser/zlrmodeltest/datafetch2")
#sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch")
sys.path.append("/home/lipchiz/Documents/pythonscripts/quant/datafetch2")
from loadSmallDataFromDB import loadData
#from fetchDataFromDB import loadData
import pandas as pd
import numpy as np
import xgboost as xgb
import pandas.io.sql as sql
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe
import datetime
import pymysql
import gc

year = 2012
#starttime_train = '%s-01-01'%year
#endtime_train = '%s-12-31'%year
#starttime_test = '%s-01-01'%(year+1)
#endtime_test = '%s-03-31'%(year+1)
#yname = "Y_20D"
#excel_h = 'resultscore_%s.xlsx'%(year)

starttime_train = '%s-08-24'%year
endtime_train = '%s-08-24'%year
starttime_test = '%s-08-24'%year
endtime_test = '%s-08-24'%year
yname = "Y_20D"
excel_h = 'resultscore_%s.xlsx'%(year)

def yload(starttime,endtime):
    connection = {'host':'10.46.228.175', 'port':3306, 'user':'alg', 
       'passwd':'Alg#824', 'db':'quant', 'charset':'utf8'}
    sql_order = "select * from STOCK_TOP_BOTTOM_Y where date>='%s' and date<='%s'"%(starttime,endtime)
    con=pymysql.connect(**connection)
    y = sql.read_sql(sql_order,con)
    return y


def objective(args):
#    global train_data,val_data
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 50,
        'silent':1,
        'learning_rate': args['learning_rate'],
        'colsample_bytree': args['colsample_bytree'],
        'max_depth': args['max_depth'] + 6,
        'subsample': args['subsample']
    }
    xgb1 = xgb.train(params,args['train_data'],evals_result = {'eval_metric':'auc'},
                num_boost_round=100000,evals=[(args['train_data'],'train'),(args['val_data'],'val')],
                verbose_eval=5,early_stopping_rounds=20)
    y_score = xgb1.predict(args['val_data'])
#    y_predict = np.int64(y_score>0.5)
    fpr,tpr,threshods = roc_curve(args['val_data'].get_label(),y_score,pos_label = 1)
    aucscore = auc(fpr,tpr)
    print(aucscore)
    return -aucscore

'________________________________split line___________________________________'
#def the training function and testing function
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('training has been started.')
print(nowtime)
#trainig function
#按需要更改模型名称
x = loadData(starttime_train,endtime_train)
y = yload(starttime_train,endtime_train)
y = y[['code','date',yname]]
tmp_data = pd.merge(x,y,on=['date','code'],how='inner')
tmp_data.dropna(subset=[yname],inplace=True)
tmp_data_1 = tmp_data[tmp_data[yname]==1]
tmp_data_0 = tmp_data[tmp_data[yname]==0]
if len(tmp_data_1)>len(tmp_data_0):
    tmp_data_1.sample(n=len(tmp_data_0),replace=False,random_state=68)
else:
    tmp_data_0.sample(n=len(tmp_data_1),replace=False,random_state=68)
tmp_data = tmp_data_0.append(tmp_data_1)
y = tmp_data[yname]
x = tmp_data.drop(['code','date',yname],axis=1)
train_x,val_x,train_y,val_y = train_test_split(x,y,test_size=0.1,random_state=68)
del tmp_data,x,y
gc.collect()

train_data = xgb.DMatrix(train_x,label=train_y,feature_names=train_x.columns)
val_data = xgb.DMatrix(val_x,label=val_y,feature_names=val_x.columns)

#parameters tuning
# Searching space
params_space = {
    'learning_rate': hp.uniform("learning_rate", 0.05, 0.15),
    'max_depth': hp.randint('max_depth',10),
    'subsample': hp.uniform("subsample", 0.5, 0.9),
    'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    'train_data':train_data,
    'val_data':val_data,
    
}

#可以调整max_evals来进行更多的尝试
best_sln = fmin(objective, space=params_space, algo=tpe.suggest, max_evals=20)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 50,
    'silent':1,
    'learning_rate': best_sln['learning_rate'],
    'colsample_bytree': best_sln['colsample_bytree'],
    'max_depth': best_sln['max_depth'] + 6,
    'subsample': best_sln['subsample']
}

xgb1 = xgb.train(params,train_data,evals_result = {'eval_metric':'auc'},
            num_boost_round=100000,evals=[(train_data,'train'),(val_data,'val')],
            verbose_eval=5,early_stopping_rounds=20)
#保存feature importance
fea_imp = xgb1.get_score(importance_type='gain')
importance = pd.DataFrame(fea_imp,index=['importance'])
importance = pd.DataFrame({'feature':importance.columns.tolist(),'importance':importance.values[0].tolist()})
importance.to_excel('feature_importance_%s.xlsx'%yname)
#按需要更改模型名称
xgb1.save_model('%s_20D.m'%year)

nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('training has finished')
print(nowtime)


'________________________________split line___________________________________'
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing has been started')
print(nowtime)
#生成下一季度测试结果
test_x = loadData(starttime_test,endtime_test)
test_y = yload(starttime_test,endtime_test)
xgb1 = xgb.Booster()
#按需要更改模型和标签名称
xgb1.load_model('%s_20D.m'%year)
test_y = test_y[['code','date',yname]]
test_y.drop(['time_stamp'],axis=1)
tmp_data = pd.merge(test_x,test_y,on=['date','code'],how='inner')
values={yname:int(0)}
tmp_data.fillna(value=values,inplace=True,axis=1)
tmp_data.reindex()
stock_index = tmp_data[['date','code']]
test_y = tmp_data[yname]
test_x = tmp_data.drop(['code','date',yname],axis=1)
test_data = xgb.DMatrix(test_x,label=test_y,feature_names=test_x.columns)


y_score = xgb1.predict(test_data)
y_predict = np.int64(y_score>0.5)
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
stock_index.to_csv('backtest_data_%s.csv'%year)

nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing has finished')
print(nowtime)
