# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
from Utils.DB_config import ConfigSpider2, ConfigQuant
# from Utils.ProcessFunc import renameDF, chgDFDataType
import pandas as pd
from datetime import datetime, timedelta
import quandl

# TARGET
targetTableName = 'NASDAQ_COMPOSITE_QUOTE'
targetTimeStamp = 'Trade Date'

targetNewTimeStamp = 'time_stamp'

historicalDataPath = r'F:\FeatureAlgorithm\Tools\Cushing_OK_WTI_Spot_Price_FOB.csv'
quandlAuthenCode = 'n1Wbo94VP2FN9yRVN2iy'
quandlProductCode = 'NASDAQOMX/COMP'


def updateFull(quant_engine, spider_engine, chunk_size, start_date='2007-01-01'):
    # get data from file
    his_data = quandl.get(quandlProductCode, start_date=start_date)

    # change data type
    his_data = his_data.reset_index()
    his_data.loc[:, targetTimeStamp] = his_data[targetTimeStamp].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # his_data = chgDFDataType(his_data, chgDataTypeCol, 'float')

    # add time stamp
    his_data[targetNewTimeStamp] = datetime.now()

    his_data.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')


def updateIncrm(quant_engine, spider_engine):
    # get lastest tradedate
    sql_statement = "select max(`%s`) from %s" % (targetTimeStamp, targetTableName)
    latest_date = pd.read_sql(sql_statement, quant_engine).iloc[0,0]

    # get incremental data
    incrm_data = quandl.get(quandlProductCode, start_date=latest_date)
    if incrm_data.empty: # no new data
        return

    # change data type
    incrm_data = incrm_data.reset_index()
    incrm_data.loc[:, targetTimeStamp] = incrm_data[targetTimeStamp].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')
    incrm_data = incrm_data.loc[incrm_data[targetTimeStamp] > latest_date]

    if incrm_data.empty:
        return

    # add time stamp
    incrm_data[targetNewTimeStamp] = datetime.now()

    # write data to db
    incrm_data.to_sql(targetTableName, quant_engine, index=True, if_exists='append')
    pass


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    quandl.ApiConfig.api_key = quandlAuthenCode

    chunk_size = 10

    # updateFull(quant_engine, spider_engine, chunk_size)
    updateIncrm(quant_engine, spider_engine)
