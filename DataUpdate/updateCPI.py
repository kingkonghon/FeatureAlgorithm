#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:29:14 2018

@author: lipchiz
"""

import numpy as np
import pandas as pd
import pandas.io.sql as sql
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.exc import  ProgrammingError
from datetime import datetime, timedelta
import calendar
import os
import sys


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant

sourceTableName = 'dongfangcaifuCPI'
targetTableName = 'DONGFANGCAIFU_CPI'
drop_cols = ['id', 'time_stamp', 'info_from_url']

def datelist(beginDate, endDate):
    # beginDate, endDate---->‘20160601’
    date_list = [datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_list


def expandMonth2Day(data):
       year = []
       mon = []
       for i in data['Months']:
              tmp_year = int(i[:4])
              tmp_mon = int(i[5:7]) + 1
              if tmp_mon == 13:
                     tmp_year += 1
                     tmp_mon = 1
              year.append(tmp_year)
              mon.append(tmp_mon)
       data['year'], data['mon'] = year, mon
       data_list = (np.asarray(data)).tolist()
       cols = data.columns.tolist()
       # column.append('date')
       y_idx = cols.index('year')  #*****************************
       m_idx = cols.index('mon')

       newdata = pd.DataFrame()
       for i in data_list:
              tmp_date = []
              trash, month_length = calendar.monthrange(i[y_idx], i[m_idx]) #*****************************
              begindate = str(i[y_idx]) + '-' + str(i[m_idx]) + '-' + '15'
              begindate = datetime.strptime(begindate, '%Y-%m-%d')
              enddate = begindate + timedelta(days=month_length - 1)
              tmp_date = datelist(begindate, enddate)
              tmp_data = np.repeat(i, month_length)
              tmp_data = np.reshape(tmp_data, (month_length, len(cols)), order='F') #*****************************
              tmp_data = pd.DataFrame(tmp_data)
              tmp_data.columns = cols
              tmp_data['date'] = tmp_date
              newdata = newdata.append(tmp_data)

       newdata.drop(columns=['year', 'mon'], inplace=True)

       #****************************************
       # data['date'] = data['Months'].apply(lambda x: datetime.strptime('-'.join([x[:4], x[5:7], '15']), '%Y-%m-%d'))
       # data.index = data['date']
       # data = data.drop('date', axis=1)
       # data = data.resample('D').ffill()
       #
       # data = data.reset_index()
       # data.loc[:, 'date'] = data['date'].apply(lambda x:datetime.strftime(x, '%Y-%m-%d'))
       # yconnect = create_engine('mysql+pymysql://alg:Alg#824@10.46.228.175:3306/quant?charset=utf8')
       # newdata.to_sql('dffcpi', yconnect, if_exists='append', index=True)

       # reindex columns
       cols = newdata.columns.tolist()
       cols.insert(0, cols.pop(cols.index('date')))
       newdata = newdata.reindex(columns=cols)

       newdata = newdata.sort_values('date')

       return  newdata

def updateFull():
       con_quant = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
       con_spider = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

       state_spider = "select * from %s " % sourceTableName  #*******************
       state_quant = "select * from %s " % targetTableName   #*******************

       cpi_spider = sql.read_sql(state_spider,con_spider)  #********************
       cpi_spider.drop_duplicates(subset=['Months'],inplace=True)
       cpi_spider = cpi_spider.drop(drop_cols, axis=1) #*********************

       # try:
       #        # con = pymysql.connect(**ConfigQuant)
       #        cpi_quant = sql.read_sql(state_quant,con_quant)  #********************
       #        cpi_quant.drop_duplicates(subset=['Months'],inplace=True)
       #
       #        cpi = cpi_quant.copy().append(cpi_spider)
       #        cpi = cpi.drop_duplicates(subset=['Months'], keep=False)
       # except ProgrammingError:
       #        cpi = cpi_spider #************************



       # if cpi_s.empty: #**********************
       #        return

       # expand data
       cpi = expandMonth2Day(cpi_spider)

       # ****************************************
       # change data types
       cpi = cpi.drop('Months', axis=1)
       data_cols = cpi.columns.tolist()
       data_cols.remove('date')
       for col in data_cols:
              cpi.loc[:, col] = cpi[col].apply(lambda x: float(x.replace('%','')) / 100. if x.find('%') >=0 else float(x))
       # ****************************************

       # dump data
       cpi.to_sql(targetTableName, con_quant,if_exists='replace',index=False)   #*******************


def airflowCallable():
       updateFull()

if __name__ == '__main__':
    updateFull()