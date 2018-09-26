#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:21:04 2018

@author: lipchiz
"""

import sys
sys.path.append("/home/nyuser/zlrmodeltest/datafetch")
#sys.path.append("/home/lipchiz/文档/pythonscripts/quant/datafetch")
from fetchDataFromDB import loadData
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.externals import joblib
import pandas.io.sql as sql
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_curve,classification_report,auc,confusion_matrix
import datetime
import pymysql
import gc
import time

year = 2017
starttime_train1 = '%s-01-01'%year
endtime_train1 = '%s-06-30'%year
starttime_train2 = '%s-07-31'%year
endtime_train2 = '%s-12-31'%year
starttime_q1 = '%s-01-01'%(year+1)
endtime_q1 = '%s-03-31'%(year+1)
starttime_q2 = '%s-04-01'%(year+1)
endtime_q2 = '%s-06-30'%(year+1)
excel_h = 'resultscore_%s.xlsx'%(year)

#starttime_train = '%s-08-24'%year
#endtime_train = '%s-08-24'%year
#starttime_q1 = '%s-08-24'%year
#endtime_q1 = '%s-08-24'%year
#starttime_q2 = '%s-08-24'%year
#endtime_q2 = '%s-08-24'%year
#excel_h = 'resultscore_%s.xlsx'%(year)

#the paremeters of model
parameters = {
            'silent':1 ,                        #设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
            'nthread':39,                       # cpu 线程数 默认最大
            'learning_rate':0.1,                # 如同学习率
            #min_child_weight=0.5,              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                                                #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                                #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'max_depth':6,                      # 构建树的深度，越大越容易过拟合
            'gamma':0,                          # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
            'subsample':0.9,                    # 随机采样训练样本 训练实例的子采样比
            'max_delta_step':0,                 #最大增量步长，我们允许每个树的权重估计。
            'colsample_bytree':0.9,             # 生成树时进行的列采样 
            'reg_lambda':1,                     # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            #reg_alpha=0,                       # L1 正则项参数
            #scale_pos_weight=1.3,              #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
            #objective= 'multi:softmax',        #多分类的问题 指定学习任务和相应的学习目标
            #num_class=10,                      # 类别数，多分类与 multisoftmax 并用
           ' n_estimators':500,                 #树的个数
            'seed':100                          #随机种子
            #eval_metric= 'auc'
        }

#the return rate of stocks
return_rate = {'rate_2':[1,2,3,4,5],
               'rate_5':[2,3,5,7,10],
               'rate_10':[3,5,7,10,15],
               'rate_20':[4,7,10,15,20],
               'rate_30':[5,10,15,20,25]
        }

ynamelist = []
for day in [2,5,10,20,30]:
    for rate in return_rate['rate_%s'%(str(day))]:
        ynamelist.append('Y_%sD_%sPCT'%(day,rate))

def precision_1(preds,dtrain):
    label = dtrain.get_label()
    pred = [int(i>=0.5) for i in preds]
    precision = precision_score(label,pred,pos_label=1,average='binary')
    return '1-precision',precision

def yload(starttime,endtime):
    connection = {'host':'10.46.228.175', 'port':3306, 'user':'alg', 
       'passwd':'Alg#824', 'db':'quant', 'charset':'utf8'}
    sql_order = "select * from STOCK_HIGH_Y where date>='%s' and date<='%s'"%(starttime,endtime)
    con=pymysql.connect(**connection)
    y = sql.read_sql(sql_order,con)
    return y
    
#def the training function and testing function
#trainig function
#day,rate,data,x_list = 2,1,train_data,xnamelist
def model_training(day,rate,data,x_list,parameters):
#   select the right y
    tmp_yname = 'Y_%sD_%sPCT'%(day,rate)
    train_list = x_list.copy()
    train_list.append(tmp_yname)
    tmp_data = data[train_list]
    tmp_data = tmp_data.dropna(subset=[tmp_yname])
    train_y = tmp_data[tmp_yname]
    train_x = tmp_data.drop(tmp_yname,axis=1)
    length_1 = len(train_y[train_y==1])
    length_0 = len(train_y[train_y==0])
    print('The size of data used for modeling: ' + str(len(train_x)))
    print('The number of y=1 is: ' + str(length_1))
    print('The number of y=0 is: ' + str(length_0))
    xgbdata = xgb.DMatrix(train_x,label=train_y,feature_names=train_x.columns)
    clf = xgb.train(parameters,xgbdata,num_boost_round=300)
    fea_imp = clf.get_score(importance_type='gain')
    importance = pd.DataFrame(fea_imp,index=['importance'])
    importance = pd.DataFrame({'feature':importance.columns.tolist(),'importance':importance.values[0].tolist()})
    importance.to_excel('feature_importance_%sD_%sPCT.xlsx'%(str(day),str(rate)))
#   save the model which has been trained 
    joblib.dump(clf,'train_model_%sD_%sPCT.m'%(str(day),str(rate)))
    report = 'day = %sD,rate=%sPCT'%(str(day),str(rate))
    print(report)
    print('-----------------------------------------------------')
     

#testing function
#    day,rate,data,x_list = 2,1,test_data,xnamelist
def model_testing(day,rate,data,x_list,stage):
    #   select the right y
    tmp_yname = 'Y_%sD_%sPCT'%(day,rate)
    test_list = x_list.copy()
    test_list.append(tmp_yname)
    tmp_data = data[test_list]
    tmp_data = tmp_data.dropna(subset=[tmp_yname])
    test_y = tmp_data[tmp_yname]
    test_x = tmp_data.drop(tmp_yname,axis=1)
    test_x = data[x_list]
#    test_y = data[tmp_yname]
    test_y = pd.DataFrame(test_y)
    xgbdata = xgb.DMatrix(test_x,label=test_y,feature_names=xnamelist)
    print('The size of data used for testing: ' + str(len(test_x)))
    print('The number of y is: ' + str(len(test_y[test_y==1])))
    
#   evaluating the model
    clf = joblib.load('train_model_%sD_%sPCT.m'%(str(day),str(rate)))
    y_score = clf.predict(xgbdata)
    y_predict = np.int64(y_score>0.5)
    accuracyscore = accuracy_score(test_y, y_predict)
#    f1 = f1_score(test_y,y_predict,pos_label=0)
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
    print(confusion_matrix(test_y,y_predict,labels=[0,1]))
    print(confusion_matrix(test_y,y_predict,labels=[1,0]))
    report = 'day = %sD,rate=%sPCT'%(str(day),str(rate))
    print(report)
    print('-------------------------------------------------------')
#   save the evaluating result
    resultscore = [precision,recall,aucscore,accuracyscore,ks,str(classification_report(test_y,y_predict)),str(confusion_matrix(test_y,y_predict)),'%sD_%sPCT'%(str(day),str(rate)),stage]
    columnname = ['precision','recall','auc','accuracyscore','K-S','classification_report','confusion_matrix','modeltype','quantile']
    result =pd.DataFrame(np.array(resultscore).reshape(1,9),columns = columnname)
    return result


'''----------------------------split line-----------------------------------'''
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('training has been started.')
print(nowtime)
#laod the train data part1
train_x1 = loadData(starttime_train1,endtime_train1)
train_y1 = yload(starttime_train1,endtime_train1)
train_x2 = loadData(starttime_train2,endtime_train2)
train_y2 = yload(starttime_train2,endtime_train2)
train_x = train_x1.append(train_x2)
train_y = train_y1.append(train_y2)
#train_x = loadData(starttime_train,endtime_train)
#train_y = yload(starttime_train,endtime_train)
train_y.drop('time_stamp',axis=1,inplace=True)
xnamelist = train_x.columns.tolist()
xnamelist.remove('code')
xnamelist.remove('date')
train_data = pd.merge(train_x,train_y,on = ['date','code'],how='left')
#del train_x,train_y,train_x1,train_x2,train_y1,train_y2
gc.collect()

#preprocessing of the original data
try:
    train_data.drop_duplicates(['code','date'],inplace = True)
    if 'index' in train_data.columns.tolist():
        train_data = train_data.drop(['index'],axis = 1)
except:
    print('train_data error')
    
train_data.drop(['date','code'],axis=1,inplace=True)

resultscoredf_h = pd.DataFrame()

#training the model
for day in [2,5,10,20,30]:
    for rate in return_rate['rate_%s'%(str(day))]:
        model_training(day,rate,train_data,xnamelist,parameters)
#delete all the variables
del day,parameters,rate,train_data
gc.collect()

nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('training has finished')
print(nowtime)
'''----------------------------split line-----------------------------------'''
#Q1
#Q1
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing_q1 has been started')
print(nowtime)

#load the test data
test_x = loadData(starttime_q1,endtime_q1)
test_y = yload(starttime_q1,endtime_q1)
test_y.drop('time_stamp',axis=1,inplace=True)
test_data = pd.merge(test_x,test_y,on = ['date','code'],how='left')

xnamelist = test_x.columns.tolist()
xnamelist.remove('date')
xnamelist.remove('code')

del test_x,test_y
gc.collect()

#preprocessing of the original data
try:
    test_data.drop_duplicates(['code','date'],inplace = True)
    if 'index' in test_data.columns.tolist():
        test_data = test_data.drop(['index'],axis = 1)
except:
    print('test_data error')

test_data.drop(['date','code'],axis=1,inplace=True)
#dataframe to save the result
resultscoredf_h = pd.DataFrame()

for day in [2,5,10,20,30]:
    for rate in return_rate['rate_%s'%(str(day))]:
        result = model_testing(day,rate,test_data,xnamelist,'Q1')
        resultscoredf_h = resultscoredf_h.append(result)

nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing_q1 has finished')
print(nowtime)

'______________________________split line_____________________________________'
#Q2
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing_q2 has been started')
print(nowtime)

#load the test data
test_x = loadData(starttime_q2,endtime_q2)
test_y = yload(starttime_q2,endtime_q2)
test_y.drop('time_stamp',axis=1,inplace=True)
test_data = pd.merge(test_x,test_y,on = ['date','code'],how='left')

del test_x,test_y
gc.collect()
#preprocessing of the original data
try:
    test_data.drop_duplicates(['code','date'],inplace = True)
    if 'index' in test_data.columns.tolist():
        test_data = test_data.drop(['index'],axis = 1)
except:
    print('train_data error')

test_data.drop(['date','code'],axis=1,inplace=True)

#release the memory
gc.collect()
time.sleep(20)

#dataframe to save the result
for day in [2,5,10,20,30]:
    for rate in return_rate['rate_%s'%(str(day))]:
        result = model_testing(day,rate,test_data,xnamelist,'Q2')
        resultscoredf_h = resultscoredf_h.append(result)

resultscoredf_h.to_excel(excel_h)
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('testing_q2 has finished')
print(nowtime)

'_________________________________split line__________________________________'
nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('backtest data generating.')
print(nowtime)

test_data = loadData(starttime_q1,endtime_q2)
try:
    test_data.drop_duplicates(['code','date'],inplace = True)
    if 'index' in test_data.columns.tolist():
        test_data = test_data.drop(['index'],axis = 1)
    test_data.reindex()
except:
    print('train_data error')

stock_index = test_data[['date','code']]
test_data.drop(['date','code'],axis=1,inplace=True)
test_data = xgb.DMatrix(test_data,feature_names=xnamelist)

for day in [2,5,10,20,30]:
    for rate in return_rate['rate_%s'%(str(day))]:
        clf = joblib.load('train_model_%sD_%sPCT.m'%(str(day),str(rate)))
        y_score = clf.predict(test_data)
        y_score = pd.DataFrame(y_score,columns=['proba_1_%s_%s'%(day,rate)])
        stock_index = pd.concat([stock_index,y_score],axis=1)
        print('day = %sD,rate=%sPCT'%(str(day),str(rate)))
stock_index.to_csv("stockscore_%s.csv"%year,index=False,sep=',')

nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('backtest data has generated. ')
print(nowtime)
