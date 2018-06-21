import tushare
from sqlalchemy import create_engine
from Utils.DB_config import ConfigSpider, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType
import pandas as pd


checkPeriodStart = '2017-10-10'
checkPeriodEnd = '2017-10-15'

spiderTableName = 'EastMoneyIndex'
spiderDataFields = ['report_time', 'code', 'open', 'high', 'low', 'close', 'turnover', 'amount']
spiderTime = 'report_time'


historyTableName = 'yucezhe_market_index'
historyTableNameHS = 'yucezhe_hs300'
historyDataFields = ['date', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount']
historyDataFieldsHS = ['date', 'hs300_open', 'hs300_high', 'hs300_low', 'hs300_close', 'hs300_volume', 'hs300_money']
historyTime = 'date'

newFields = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
newFieldsHS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
marketToCode = {'SH': '000001', 'SZ':'399001', 'SME':'399005', 'GEB': '399006'}
codeHS300 = '000300'

def checkMarketIndex(spider_engine, quant_engine):
    # fetch data from spider
    tmp_fields = list(map(lambda x: '`%s`'%x, spiderDataFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from %s where (`%s` >= '%s') and (`%s` <= '%s')" % (
        tmp_fields, spiderTableName, spiderTime, checkPeriodStart, spiderTime, checkPeriodEnd)
    spider_data = pd.read_sql(sql_statement, spider_engine)

    # rename columns
    spider_data = renameDF(spider_data, spiderDataFields, newFields)

    # change data type
    spider_data = chgDFDataType(spider_data,['open', 'high', 'low', 'close', 'volume'], 'float')

    # fetch data from quant
    # market index
    tmp_fields = list(map(lambda x: '`%s`'%x, historyDataFields))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from %s where (`%s` >= '%s') and (`%s` <= '%s')" % (
        tmp_fields, historyTableName, historyTime, checkPeriodStart, historyTime, checkPeriodEnd)
    his_data = pd.read_sql(sql_statement, quant_engine)
    # HS300
    tmp_fields = list(map(lambda x: '`%s`' % x, historyDataFieldsHS))
    tmp_fields = ','.join(tmp_fields)
    sql_statement = "select %s from %s where (`%s` >= '%s') and (`%s` <= '%s')" % (
        tmp_fields, historyTableNameHS, historyTime, checkPeriodStart, historyTime, checkPeriodEnd)
    his_data_HS = pd.read_sql(sql_statement, quant_engine)

    # rename columns HS
    his_data_HS = renameDF(his_data_HS, historyDataFieldsHS, newFieldsHS)

    # match market to index code
    his_data['code'] = ''
    for i in marketToCode.items():
        his_data.loc[his_data['market'] == i[0], 'code'] = i[1]

    # match index
    spider_data_HS = spider_data.loc[spider_data['code'] == codeHS300]
    spider_data =spider_data.loc[spider_data['code'].isin(list(marketToCode.values()))]

    # market index
    combine_data = spider_data.merge(his_data, on=['date', 'code'], suffixes=['', '_h'])
    diff_num = {}
    for field in ['open', 'high', 'low', 'close']:
        diff_num[field] = ((combine_data[field] - combine_data[field + '_h']).abs() > 0.00001).sum()

    # HS300
    combine_data = spider_data_HS.merge(his_data_HS, on='date', suffixes=['', '_h'])
    diff_num_hs = {}
    for field in ['open', 'high', 'low', 'close']:
        diff_num_hs[field] = ((combine_data[field] - combine_data[field + '_h']).abs() > 0.00001).sum()

    pass


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    # create source engine
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider))

    checkMarketIndex(spider_engine, quant_engine)