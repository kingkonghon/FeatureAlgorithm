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


def precision_1(preds, dtrain):
    label = dtrain.get_label()
    pred = [int(i >= 0.5) for i in preds]
    precision = precision_score(label, pred, pos_label=1, average='binary')
    return '1-precision', precision


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

def model_testing_old(day,data,x_list,season, logger, out_folder_path):
    #   select the right y
    tmp_yname = 'Y_%sD'%(day)
    test_list = x_list.copy()
    test_list.append(tmp_yname)
    tmp_data = data[test_list]
    tmp_data = tmp_data.dropna(subset=[tmp_yname])

    test_y = tmp_data[tmp_yname]
    test_x = tmp_data.drop(tmp_yname,axis=1)
    xgbdata = xgb.DMatrix(test_x,label=test_y,feature_names=x_list)
    logger.info('The size of data used for testing: ' + str(len(test_x)))
    logger.info('The number of y is: ' + str(len(test_y[test_y==1])))

    logger.info('feature num: %d' % len(x_list))
    logger.info('test data types:')
    logger.info(test_x.dtypes.unique())
    logger.info(test_y.unique())

#   evaluating the model
    xgb1 = xgb.Booster()
    xgb1.load_model('%s/train_model_%dD.m' % (out_folder_path, day))
    # clf = joblib.load('%s/train_model_%dD.m'%(out_folder_path, day))
    y_score = xgb1.predict(xgbdata)
    y_predict = np.int64(y_score>0.5)
    accuracyscore = accuracy_score(test_y, y_predict)
#    f1 = f1_score(test_y,y_predict,pos_label=0)
    fpr,tpr,threshods = roc_curve(test_y,y_score,pos_label = 1)
    ks = np.max(np.abs(tpr-fpr))
    aucscore = auc(fpr,tpr)
    precision = precision_score(test_y,y_predict,average='binary')
    recall = recall_score(test_y,y_predict,average='weighted')
    logger.info('precision: %f' % precision)
    logger.info('recall: %f' % recall)
    logger.info('auc: %f' % aucscore)
    logger.info('accuracyscore: %f' % accuracyscore)
    logger.info('K-S: %f' % ks)
    logger.info(classification_report(test_y,y_predict))
    logger.info(confusion_matrix(test_y,y_predict,labels=[0,1]))
    logger.info(confusion_matrix(test_y,y_predict,labels=[1,0]))
    report = 'day = %dD'%(day)
    logger.info(report)
    logger.info('-------------------------------------------------------')
#   save the evaluating result
    resultscore = [precision,recall,aucscore,accuracyscore,ks,str(classification_report(test_y,y_predict)),str(confusion_matrix(test_y,y_predict)),'%dD'%(day),'s%d'%season]
    columnname = ['precision','recall','auc','accuracyscore','K-S','classification_report','confusion_matrix','modeltype','season']
    result =pd.DataFrame(np.array(resultscore).reshape(1,9),columns = columnname)
    return result

def model_testing_new(day,data,x_list,season, logger, out_folder_path):
    #   select the right y
    tmp_yname = 'Y_%sD'%(day)
    test_list = x_list.copy()
    test_list.append(tmp_yname)
    tmp_data = data[test_list]
    # tmp_data = tmp_data.dropna(subset=[tmp_yname])

    # fill y'nan with 0
    tmp_data.loc[:, tmp_yname] = tmp_data[tmp_yname].fillna(0)
    tmp_data.loc[:, tmp_yname] = tmp_data[tmp_yname].astype('int')

    test_y = tmp_data[tmp_yname]
    test_x = tmp_data.drop(tmp_yname,axis=1)
    xgbdata = xgb.DMatrix(test_x,label=test_y,feature_names=x_list)
    logger.info('The size of data used for testing: ' + str(len(test_x)))
    logger.info('The number of y is: ' + str(len(test_y[test_y==1])))

#   evaluating the model
    xgb1 = xgb.Booster()
    xgb1.load_model('%s/train_model_%dD.m' % (out_folder_path, day))
    # clf = joblib.load('%s/train_model_%dD.m'%(out_folder_path, day))
    y_score = xgb1.predict(xgbdata)
    y_predict = np.int64(y_score>0.5)   ############## threshold to be predicted as 1
    accuracyscore = accuracy_score(test_y, y_predict)
#    f1 = f1_score(test_y,y_predict,pos_label=0)
    fpr,tpr,threshods = roc_curve(test_y,y_score,pos_label = 1)
    ks = np.max(np.abs(tpr-fpr))
    aucscore = auc(fpr,tpr)
    precision = precision_score(test_y,y_predict,average='binary')
    recall = recall_score(test_y,y_predict,average='weighted')
    logger.info('precision: %f' % precision)
    logger.info('recall: %f' % recall)
    logger.info('auc: %f' % aucscore)
    logger.info('accuracyscore: %f' % accuracyscore)
    logger.info('K-S: %f' % ks)
    logger.info(classification_report(test_y,y_predict))
    logger.info(confusion_matrix(test_y,y_predict,labels=[0,1]))
    logger.info(confusion_matrix(test_y,y_predict,labels=[1,0]))

    # check score under different thresholds
    threshold_list = list(range(50,100, 5))
    threshold_list = [round(x * 0.01,2) for x in threshold_list]
    scores_list = {}
    for tmp_thrhd in threshold_list:
        tmp_y_predict = np.int64(y_score > tmp_thrhd)
        tmp_precision = precision_score(test_y, tmp_y_predict, average='binary')
        tmp_recall = recall_score(test_y, tmp_y_predict, average='weighted')
        scores_list[tmp_thrhd] = [tmp_precision, tmp_recall]
    scores_list = pd.DataFrame(scores_list, index=['precision', 'recall'])
    logger.info("scores under different thresholds:")
    logger.info(scores_list)
    report = 'day = %dD'%(day)
    logger.info(report)
    logger.info('-------------------------------------------------------')
#   save the evaluating result
    resultscore = [precision,recall,aucscore,accuracyscore,ks,str(classification_report(test_y,y_predict)),str(confusion_matrix(test_y,y_predict)),'%dD'%(day),'s%d'%season]
    columnname = ['precision','recall','auc','accuracyscore','K-S','classification_report','confusion_matrix','modeltype','season']
    result =pd.DataFrame(np.array(resultscore).reshape(1,9),columns = columnname)
    return result



def run(year, season, out_folder_path, out_predict_path):
    # ------------------  setting --------------------------------
    # season_start_date ={
    #     1: '-01-01',
    #     2: '-04-01',
    #     3: '-07-01',
    #     4: '-10-01'
    # }
    # season_end_date = {
    #     1: '-03-31',
    #     2: '-06-30',
    #     3: '-09-30',
    #     4: '-12-31'
    # }
    season_start_date = {
        1: '-01-01',
        2: '-03-01',
        3: '-05-01',
        4: '-07-01',
        5: '-09-01',
        6: '-11-01'
    }
    season_end_date = {
        1: '-02-31',
        2: '-04-31',
        3: '-06-31',
        4: '-08-31',
        5: '-10-31',
        6: '-12-31'
    }
    starttime_train = str(year) + season_start_date[season]
    endtime_train = str(year+1) + season_start_date[season]
    endtime_train = datetime.strftime(datetime.strptime(endtime_train, '%Y-%m-%d') - timedelta(days=30), '%Y-%m-%d') # drop 30 days to avoid using future data in training

    starttime_test = str(year+1) + season_start_date[season]
    endtime_test = str(year+1) + season_end_date[season]

    # starttime_train = '2012-01-01'
    # endtime_train = '2012-01-14'
    #
    # starttime_test = '2013-01-01'
    # endtime_test = '2013-01-04'

    # starttime_train1 = '%s-01-01'%year
    # endtime_train1 = '%s-06-30'%year
    # # endtime_train1 = '%s-01-04' % year
    # starttime_train2 = '%s-07-01'%year
    # endtime_train2 = '%s-12-31'%year
    # # endtime_train2 = '%s-07-04' % year
    # starttime_q1 = '%s-01-01'%(year+1)
    # endtime_q1 = '%s-03-31'%(year+1)
    # # endtime_q1 = '%s-01-04' % (year + 1)
    # # starttime_q2 = '%s-04-01'%(year+1)
    # # endtime_q2 = '%s-06-30'%(year+1)
    # # excel_h = 'resultscore_%s.xlsx'%(year)

    Y_table_name = 'STOCK_TOP_BOTTOM_Y'
    Y_days = [2,5,10,20]

    #starttime_train = '%s-06-21'%year
    #endtime_train = '%s-06-21'%year
    #starttime_q1 = '%s-06-21'%year
    #endtime_q1 = '%s-06-21'%year
    #starttime_q2 = '%s-06-21'%year
    #endtime_q2 = '%s-06-21'%year
    #excel_h = 'resultscore_%s.xlsx'%(year)

    #the scope of paremeters of model
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

    # parameters = {
    #             'silent':1 ,                        #设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    #             'nthread':30,                       # cpu 线程数 默认最大
    #             'learning_rate':0.1,                # 如同学习率
    #             #min_child_weight=0.5,              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #                                                 #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #                                                 #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    #             'max_depth':6,                      # 构建树的深度，越大越容易过拟合
    #             'gamma':0,                          # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    #             'subsample':0.9,                    # 随机采样训练样本 训练实例的子采样比
    #             'max_delta_step':0,                 #最大增量步长，我们允许每个树的权重估计。
    #             'colsample_bytree':0.9,             # 生成树时进行的列采样
    #             'reg_lambda':1,                     # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #             #reg_alpha=0,                       # L1 正则项参数
    #             #scale_pos_weight=1.3,              #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    #             #objective= 'multi:softmax',        #多分类的问题 指定学习任务和相应的学习目标
    #             #num_class=10,                      # 类别数，多分类与 multisoftmax 并用
    #            ' n_estimators':500,                 #树的个数
    #             'seed':100,                          #随机种子
    #             'eval_metric': 'auc'
    #         }

    #the return rate of stocks
    # return_rate = {'rate_2':[1,2,3,4,5],
    #                'rate_5':[2,3,5,7,10],
    #                'rate_10':[3,5,7,10,15],
    #                'rate_20':[4,7,10,15,20],
    #                'rate_30':[5,10,15,20,25]
    #         }

    # ynamelist = []
    # for day in [2,5,10,20,30]:
    #     for rate in return_rate['rate_%s'%(str(day))]:
    #         ynamelist.append('Y_%sD_%sPCT'%(day,rate))

    # create logger
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='%s/%d_s%d.log' % (out_folder_path, year, season),
                        filemode='w')

    logger = logging.getLogger('%d_s%d'%(year, season))


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
    train_y = yload(Y_table_name, starttime_train,endtime_train)
    train_y.drop('time_stamp',axis=1,inplace=True)
    xnamelist = train_x.columns.tolist()  # feature names (without code & date)
    xnamelist.remove('code')
    xnamelist.remove('date')
    train_data = pd.merge(train_x,train_y,on = ['date','code'],how='left')

    del train_x, train_y
    gc.collect()

    # save training feature name
    out_path = '%s/xnamelist.pcl' % out_folder_path
    with open(out_path, 'wb') as out_file:
        pickle.dump(xnamelist, out_file)

    # preprocessing training data
    try:
        train_data.drop_duplicates(['code','date'],inplace = True)
        train_data = train_data.sort_values('date', ascending=True)
    except:
        logger.error('train_data error')

    train_data.drop(['date','code'],axis=1,inplace=True)  # drop code & date

    #training the model
    # for day in [2,5,10,20,30]:
    for day in Y_days:
        model_training(day, train_data, xnamelist,params_space, logger, out_folder_path)
    #delete all the variables
    del day,params_space,train_data
    gc.collect()

    logger.info('training has finished')
    '''---------------------------- testing S1 -----------------------------------'''
    #S1
    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('testing_q1 has been started')
    # logger.info(nowtime)

    # load training feature name
    feature_list_path = '%s/xnamelist.pcl' % out_folder_path
    with open(feature_list_path, 'rb') as in_file:
        xnamelist = pickle.load(in_file)

    #load the test data
    # test_x = loadData(starttime_q1,endtime_q1)
    # test_y = yload(Y_table_name, starttime_q1,endtime_q1)
    test_x = loadData(starttime_test, endtime_test)
    test_y = yload(Y_table_name, starttime_test, endtime_test)
    test_y.drop('time_stamp',axis=1,inplace=True)
    test_data = pd.merge(test_x,test_y,on = ['date','code'],how='left')

    del test_x,test_y
    gc.collect()

    #preprocessing testing data
    try:
        test_data.drop_duplicates(['code','date'],inplace = True)
        if 'index' in test_data.columns.tolist():
            test_data = test_data.drop(['index'],axis = 1)
    except:
        logger.error('test_data error')

    # stock_index_q1 = test_data[['date','code']]
    test_data.drop(['date','code'],axis=1,inplace=True)  # drop code & date

    #dataframe to save the result
    resultscoredf_h = pd.DataFrame()

    for day in Y_days:
        result = model_testing_new(day,test_data,xnamelist,season, logger, out_folder_path)
        # y_score = pd.DataFrame(y_score)
        # y_score.columns = ["y_1_%sD_%sPCT"%(day,rate)]
        # stock_index_q1 = pd.concat([stock_index_q1,y_score],axis=1)
        resultscoredf_h = resultscoredf_h.append(result)

    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('testing s%d has finished' % season)
    # print(nowtime)

    '_________________________________ Record Prediction __________________________________'
    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('backtest data generating.')
    # print(nowtime)

    test_data = loadData(starttime_test,endtime_test)
    try:
        test_data.drop_duplicates(['code','date'],inplace = True)
        if 'index' in test_data.columns.tolist():
            test_data = test_data.drop(['index'],axis = 1)
        test_data.reindex()
    except:
        logger.error('train_data error')

    stock_index = test_data[['date','code']]
    test_data.drop(['date','code'],axis=1,inplace=True)
    test_data = xgb.DMatrix(test_data,feature_names=xnamelist)

    for day in Y_days:
        xgb1 = xgb.Booster()
        xgb1.load_model('%s/train_model_%dD.m'%(out_folder_path, day))
        y_score = xgb1.predict(test_data)
        y_score = pd.DataFrame(y_score,columns=['proba_1_%dD'%day])
        stock_index = pd.concat([stock_index,y_score],axis=1)
        logger.info('day = %sD'%day)
    stock_index.to_csv("%s/stockscore_%ds%d.csv" % (out_predict_path, year, season),index=False,sep=',')

    # nowtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info('backtest data has generated. ')
    # print(nowtime)


if __name__ == '__main__':
    # result_folder_path = '/home/konghon/Documents/model_results/2012'
    year = 2012
    season = 1
    result_folder_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results/%d_s%d' % (year, season)
    # result_folder_path = r'D:\model_results\top_bottom\test_v2'
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    result_predict_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results/prediction'
    # result_predict_path = r'D:\model_results\top_bottom\test_v2'
    run(year, season,result_folder_path, result_predict_path)