from sqlalchemy import create_engine
import tushare as ts
import os
import sys
from datetime import datetime
import requests

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# ================= update from SHIBOR
# # SOURCE (由于guijinshu更新速度太慢，所以为了及时获取最新的交易日，用更新比较快的shibor)
# sourceTableName = 'guijinshu'
# sourceTableNameIncrm = 'shibor'
# sourceDateField = 'report_date'
# sourceDateFieldIncrm = 'ReportDate'

# TARGET
targetTableName = 'TRADE_CALENDAR'
targetDateField = 'date'


# def updateFull(source_sql_engine, target_sql_engine):
#     sql_statement = "select distinct `%s` from `%s` order by %s asc" % (sourceDateField, sourceTableName, sourceDateField)
#     trade_date = pd.read_sql(sql_statement, source_sql_engine)
#     trade_date = trade_date.rename(columns={sourceDateField: targetDateField})
#
#     trade_date.to_sql(targetTableName, target_sql_engine, index=False, if_exists='replace')
#
# def updateIncrm(source_sql_engine, target_sql_engine):
#     sql_statement = "select max(%s) from %s" % (targetDateField, targetTableName)
#     target_latest_date = pd.read_sql(sql_statement, target_sql_engine)
#     target_latest_date = target_latest_date.iloc[0,0]
#
#     sql_statement = "select distinct `%s` from `%s` where %s > '%s' order by %s asc" % (sourceDateFieldIncrm, sourceTableNameIncrm,
#                                                                     sourceDateFieldIncrm, target_latest_date, sourceDateFieldIncrm)
#     trade_date = pd.read_sql(sql_statement, source_sql_engine)
#     trade_date = trade_date.rename(columns={sourceDateFieldIncrm: targetDateField})
#
#
#     # change date format
#     trade_date = trade_date['date'].apply(lambda x: x[:10])
#
#     trade_date = trade_date.loc[trade_date > target_latest_date]
#
#     if not trade_date.empty:
#         trade_date.to_sql(targetTableName, target_sql_engine, index=False, if_exists='append')

# ================== update from tushare
def updateFromTushare(target_sql_engine):
    ts_token = 'e37ce8e806bbfc9bfcb9ac35e68998c1710e7f0714e8fb5f257cd13c'
    ts_api = ts.pro_api(ts_token)

    # get current year
    current_year = datetime.now().year
    year_end_date = '%d1231' % current_year

    retry_num = 0
    while retry_num < 10:
        try:
            trade_cal = ts_api.trade_cal(start_date='20070101', end_date=year_end_date)
            break
        except requests.exceptions.ReadTimeout:
            retry_num += 1
    if retry_num >= 10:  #  cannot download tradedate within limited number of times, stop updating
        return
    trade_cal.columns = ['exchange', 'date', 'is_open']
    trade_cal = trade_cal.loc[trade_cal['is_open'] == 1, 'date']

    trade_cal = trade_cal.apply(lambda x: '-'.join([x[:4], x[4:6], x[6:]]))

    tmp_conn = target_sql_engine.connect()
    trade_cal.to_sql(targetTableName, tmp_conn, index=False, if_exists='replace')
    tmp_conn.close()


if __name__ == '__main__':
    # spider_engine = create_engine(
    #     'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider))
    #
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # # updateFull(spider_engine, quant_engine)
    # updateIncrm(spider_engine, quant_engine)
    updateFromTushare(quant_engine)


def airflowCallable():
    # spider_engine = create_engine(
    #     'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider))

    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # # updateFull(spider_engine, quant_engine)
    # updateIncrm(spider_engine, quant_engine)

    updateFromTushare(quant_engine)