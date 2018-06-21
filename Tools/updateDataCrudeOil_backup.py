# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pymysql
import pandas as pd
from datetime import datetime, timedelta
import quandl
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

# SOURCE
sourceQuickTableName = 'ChinaOilNetHangqing'
sourceQuickField = 'preClose'
sourceQuickTimestampField = 'time_stamp'
sourceNameField = 'stockName'
sourceOilName = r'美国原油期货'

targetQuickField = ['Date', 'Settle']

# CALENDAR
calendarTableName = 'TRADE_CALENDAR'
calendarField = 'date'

# TARGET
targetTableName = 'WTI_CONTINUOUS_QUOTE'
chgDataTypeCol = ['EFP Volume', 'EFS Volume']
targetTimeStamp = 'Date'

targetNewTimeStamp = 'time_stamp'

historicalDataPath = r'F:\FeatureAlgorithm\Tools\Cushing_OK_WTI_Spot_Price_FOB.csv'
quandlAuthenCode = 'n1Wbo94VP2FN9yRVN2iy'
quandlProductCode = 'CHRIS/ICE_T1'


def updateFull(start_date='2007-01-01'):
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    quandl.ApiConfig.api_key = quandlAuthenCode

    # get data from file
    his_data = quandl.get(quandlProductCode, start_date=start_date)

    # change data type
    his_data = his_data.reset_index()
    his_data.loc[:, targetTimeStamp] = his_data[targetTimeStamp].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    his_data = chgDFDataType(his_data, chgDataTypeCol, 'float')

    # add time stamp
    his_data[targetNewTimeStamp] = datetime.now()

    his_data.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')

# update from quandl in the afternoon
def updateIncrm():
    # delete quick quote
    con_quant = pymysql.connect(**ConfigQuant)
    cursor = con_quant.cursor()
    sql_statement = "delete from %s where %s is null" % (targetTableName, targetNewTimeStamp)
    cursor.execute(sql_statement)
    con_quant.commit()
    con_quant.close()

    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    quandl.ApiConfig.api_key = quandlAuthenCode

    # get lastest tradedate
    sql_statement = "select max(`%s`) from %s where %s is not null" % (targetTimeStamp, targetTableName, targetNewTimeStamp)
    latest_date = pd.read_sql(sql_statement, quant_engine).iloc[0,0]

    # get incremental data
    incrm_data = quandl.get(quandlProductCode, start_date=latest_date)
    if incrm_data.empty: # no new data
        return

    # change data type
    incrm_data = incrm_data.reset_index()
    incrm_data.loc[:, targetTimeStamp] = incrm_data[targetTimeStamp].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

    incrm_data = incrm_data.loc[incrm_data[targetTimeStamp] > latest_date]
    if incrm_data.empty:
        return

    # add time stamp
    incrm_data.loc[:, targetNewTimeStamp] = datetime.now()

    # write data to db
    incrm_data.to_sql(targetTableName, quant_engine, index=False, if_exists='append')
    pass

# update from spider in the morning
def updateQuick():
    # create target connection
    con_quick = pymysql.connect(**ConfigQuant)

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # get target latest date
    sql_statement = "select max(`%s`) from %s" % (targetTimeStamp, targetTableName)
    target_max_date = pd.read_sql(sql_statement, con_quick)
    if not target_max_date.empty:
        target_max_date = target_max_date.iloc[0,0]

    # get data from spider
    target_max_date_format = r'%s年%s月%s日' %(target_max_date[:4], target_max_date[5:7], target_max_date[8:10])
    sql_statement = "select `%s`, `%s` from %s where (%s > '%s') and (%s = '%s')" % (sourceQuickTimestampField, sourceQuickField,
                                                                   sourceQuickTableName, sourceQuickTimestampField,
                                                                   target_max_date_format, sourceNameField, sourceOilName)
    quick_data = pd.read_sql(sql_statement, spider_engine)

    if quick_data.empty:
        return

    # get trade calendar
    sql_statement = "select %s from %s" % (calendarField, calendarTableName)
    calendar = pd.read_sql(sql_statement, con_quick).values.T[0]

    # change data type & format
    quick_data.columns = targetQuickField
    tmp_dt = quick_data['Date'].apply(lambda x: datetime.strptime(x[:11], r'%Y年%m月%d日') - timedelta(days=1))
    quick_data.loc[:, 'Date'] = tmp_dt.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    quick_data = quick_data.loc[quick_data['Date'].isin(calendar)]
    quick_data = quick_data.loc[quick_data['Date'] > target_max_date]
    if quick_data.empty:
        return

    quick_data.loc[:, 'Settle'] = quick_data['Settle'].astype('float')

    # sql statemet
    quick_data.loc[:, 'sql'] = quick_data.apply(lambda x: "insert into `%s` (Date, Settle) values ('%s', %f)" % (targetTableName, x.Date, x.Settle), axis=1)

    # insert quick quote into DB
    cursor = con_quick.cursor()
    for i in range(quick_data.shape[0]-1, -1, -1):
        cursor.execute(quick_data.iloc[i]['sql'])

    con_quick.commit()
    con_quick.close()

def airflowCallableMorning():
    updateQuick()


def airflowCallableAfternoon():
    updateIncrm()


if __name__ == '__main__':
    # updateFull()
    updateIncrm()
    # updateQuick()
