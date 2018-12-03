
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider2, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

calendarTableName = 'TRADE_CALENDAR'

# SOURCE SPIDER
sourceTableName = 'shangzhengjijinzhishu'
sourceFields = ['report_time', 'open_price', 'top_price', 'low_price', 'close_price', 'VOL',
                'Transaction_Amount']
sourceCodeField = 'stock'
sourceStockCode = ' 000011'  # space before code !!
sourceTimeStamp = 'report_time'

# SOURCE TUSHARE
sourceTSTableName = 'STOCK_INDEX_QUOTE_TUSHARE'
sourceTSFields = ['date', 'open', 'high', 'low', 'close', 'vol',
                'amount']
sourceTSCodeField = 'code'
sourceTSStockCode = '000011'
sourceTSTimeStamp = 'date'


# TARGET
targetTableName = 'FUND_INDEX'
targetFields = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
targetTimeStamp = 'date'
targetNewTimeStamp = 'time_stamp'
chgDataTypeCol = [ 'open', 'high', 'low', 'close', 'volume', 'amount']

def updateTushare(sql_conn_quant, target_max_timestamp, supposed_date_num):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceTSFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where (`%s` > '%s') and (%s = '%s')" % (
        tmp_fields, sourceTSTableName, sourceTSTimeStamp, target_max_timestamp, sourceTSCodeField, sourceTSStockCode)
    incrm_data = pd.read_sql(sql_statement, sql_conn_quant)

    # drop duplicates and sort values
    incrm_data = incrm_data.drop_duplicates(sourceTSTimeStamp)
    incrm_data = incrm_data.sort_values(sourceTSTimeStamp)

    incrm_data_num = incrm_data[sourceTSTimeStamp].unique().size

    # write data tot target
    if incrm_data_num == supposed_date_num:
        # rename columns
        incrm_data = renameDF(incrm_data, sourceTSFields, targetFields)

        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # add time stamp
        incrm_data[targetNewTimeStamp] = datetime.now()

        incrm_data.to_sql(targetTableName, sql_conn_quant, index=False, if_exists='append')

        return True
    else:
        return False


def updateSpider(sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where (`%s` > '%s') and (%s = '%s')" % (
        tmp_fields, sourceTableName, sourceTimeStamp, target_max_timestamp, sourceCodeField, sourceStockCode)
    incrm_data = pd.read_sql(sql_statement, sql_conn_spider)

    # drop duplicates and sort values
    incrm_data = incrm_data.drop_duplicates(sourceTimeStamp)
    incrm_data = incrm_data.sort_values(sourceTimeStamp)

    incrm_data_num = incrm_data[sourceTimeStamp].unique().size

    # write data to target
    if incrm_data_num == supposed_date_num:
        # rename columns
        incrm_data = renameDF(incrm_data, sourceFields, targetFields)

        # change data type
        incrm_data = chgDFDataType(incrm_data, chgDataTypeCol, 'float')

        # add time stamp
        incrm_data[targetNewTimeStamp] = datetime.now()

        incrm_data.to_sql(targetTableName, sql_conn_quant, index=False, if_exists='append')

        return True
    else:
        return False


def updateFull(quant_engine, spider_engine):
    # fetch data from source
    tmp_fields = list(map(lambda x: '`%s`' % x, sourceFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from `%s` where %s = '%s'" % (tmp_fields, sourceTableName, sourceCodeField, sourceStockCode)
    full_data = pd.read_sql(sql_statement, spider_engine)

    # drop duplicates & sort value
    full_data = full_data.drop_duplicates(sourceTimeStamp)
    full_data = full_data.sort_values(sourceTimeStamp)

    # rename columns
    full_data = renameDF(full_data, sourceFields, targetFields)

    # change data type
    full_data = chgDFDataType(full_data, chgDataTypeCol, 'float')

    # add time stamp
    full_data[targetNewTimeStamp] = datetime.now()

    # write data tot target
    if not full_data.empty:
        full_data.to_sql(targetTableName, quant_engine, index=False, if_exists='replace')

def updateIncrm(quant_engine, spider_engine):
    sql_conn_quant = quant_engine.connect()
    sql_conn_spider = spider_engine.connect()

    # get target latest date
    sql_statement = 'select max(`%s`) from `%s`' % (targetTimeStamp, targetTableName)
    target_max_timestamp = pd.read_sql(sql_statement, sql_conn_quant)
    target_max_timestamp = target_max_timestamp.iloc[0, 0]

    # get calendar
    sql_statement = "select `date` from %s" % calendarTableName
    calendar = pd.read_sql(sql_statement, sql_conn_quant).values.T[0]

    today = datetime.now()
    cur_hour = today.hour
    today = datetime.strftime(today, '%Y-%m-%d')

    supposed_date_num = calendar[(calendar > target_max_timestamp) & (calendar <= today)].size  # supposing number of dates for new data
    if cur_hour < 15:  # before market close, day num - 1
        supposed_date_num -= 1


    if supposed_date_num > 0:  # need to update data
        # update from tushare
        is_successful = updateTushare(sql_conn_quant, target_max_timestamp, supposed_date_num)

        if not is_successful:
            # update from spider
            updateSpider(sql_conn_quant, sql_conn_spider, target_max_timestamp, supposed_date_num)

    sql_conn_quant.close()
    sql_conn_spider.close()


def airflowCallable():
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # updateFull(quant_engine, spider_engine)
    updateIncrm(quant_engine, spider_engine)

if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))

    # updateFull(quant_engine, spider_engine)
    updateIncrm(quant_engine, spider_engine)
