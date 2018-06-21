import os
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from datetime import datetime
import tushare as ts

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant, ConfigSpider2

calendarTableName = 'TRADE_CALENDAR'

rankTableConfig ={
    'fundamental_ttm': 'STOCK_FUNDAMENTAL_TTM',
    'technical': 'DERI_STOCK_TECH_INDICATORS',
    'area': 'STOCK_DESCRIPTION',
    'industry': 'STOCK_INDUSTRY',
    'area_rank': 'DERI_STOCK_RANKING_AREA',
    'industry_rank': 'DERI_STOCK_RANKING_INDUSTRY',
    'market_rank': 'DERI_STOCK_RANKING_MARKET',
    'all_stock_rank': 'DERI_STOCK_RANKING_ALL_STOCKS'
}

categoryIndexTableConfig = {
    'area': 'STOCK_DESCRIPTION',
    'industry': 'STOCK_INDUSTRY',
    'area_index': 'AREA_INDEX_FORWARD_ADJ_QUOTE',
    'industry_index': 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE'
}

stockDescriptionTableConfig = {
    'spider': 'stock_list',
    'quant': 'STOCK_DESCRIPTION'
}

stockExcessiveRetsTableConfig = {
    'stock_technical': 'DERI_STOCK_TECH_INDICATORS',
    'area': 'STOCK_DESCRIPTION',
    'industry': 'STOCK_INDUSTRY',
    'area_exc_ret': 'DERI_AREA_EXCESSIVE_RET',
    'industry_exc_ret': 'DERI_INDUSTRY_EXCESSIVE_RET'
}


#  ===== (fundamental inner merge Category) outer merge (Stock technical inner merge Category) ====
def checkRecodNumForRankingByCategory(sql_config, tot_table_names, str_today):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    # get record num from db
    table_codes = {}
    for (tmp_name, tmp_table_name) in tot_table_names.items():
        if tmp_name == 'area':
            continue
        sql_statement = "select `code` from %s where (date = '%s')" % (tmp_table_name, str_today)

        tmp_codes = pd.read_sql(sql_statement, sql_engine)  # read record num
        table_codes[tmp_name] = tmp_codes['code']

    # area (no date)
    sql_statement = "select `code` from %s" % (tot_table_names['area'])
    tmp_codes = pd.read_sql(sql_statement, sql_engine)  # read record num
    table_codes['area'] = tmp_codes['code']

    # check record num
    error_tables = []

    # industry, area: (fundamental inner merge category) outer merge (technical inner merge category)
    tmp_category = ['area', 'industry']
    for tmp_c in tmp_category:
        tmp_codes_ttm_cat = table_codes['fundamental_ttm'].loc[table_codes['fundamental_ttm'].isin(table_codes[tmp_c])]  # inner merge with category
        tmp_codes_tech_cat = table_codes['technical'].loc[table_codes['technical'].isin(table_codes[tmp_c])]
        tmp_final_codes = tmp_codes_ttm_cat.append(tmp_codes_tech_cat).unique()
        tmp_final_num = tmp_final_codes.size  # ranking data at least larger than this number (outer merge)
        if table_codes[tmp_c + '_rank'].size != tmp_final_num:
            error_tables.append(tmp_c + 'rank')

    # market, all stocks: fundamental outer merge technical
    tmp_category = ['market', 'all_stock']
    tmp_final_codes = table_codes['fundamental_ttm'].append(table_codes['technical']).unique()  # outer merge
    tmp_final_num = tmp_final_codes.size
    for tmp_c in tmp_category:
        if table_codes[tmp_c + '_rank'].size != tmp_final_num:  #  ranking data at least larger than this number (outer merge)
            error_tables.append(tmp_c + '_rank')

    return error_tables

# ===== check stock area/industry index quote record num =====
def checkCategoryIndexRecordNum(sql_config, tot_table_names, str_today):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    # get category num
    category_num = {}

    # total industry num
    sql_statement = "select count(1) from (SELECT DISTINCT industry as uni_ind FROM `%s` where date='%s') as t" % (tot_table_names['industry'], str_today)
    industry_num = pd.read_sql(sql_statement, sql_engine)
    category_num['industry'] = industry_num.iloc[0, 0]

    # total area num
    sql_statement = "select count(1) from (SELECT DISTINCT area as uni_area FROM `%s`) as t" % tot_table_names['area']
    area_num = pd.read_sql(sql_statement, sql_engine)
    category_num['area'] = area_num.iloc[0, 0]

    error_tables = []
    for tmp_cat in ['industry', 'area']:
        sql_statement = "select count(1) from %s where date='%s'" % (tot_table_names[tmp_cat + '_index'], str_today)
        index_num = pd.read_sql(sql_statement, sql_engine)
        index_num = index_num.iloc[0,0]

        if category_num[tmp_cat] != index_num:
            error_tables.append(tot_table_names[tmp_cat + '_index'])

    return error_tables

def checkStockDescription(spider_config, quant_config, table_config):
    spider_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**spider_config))

    sql_statement = "select count(1) from `%s` where timeToMarket!='0'" % table_config['spider']
    record_num_spider = pd.read_sql(sql_statement, spider_engine)
    record_num_spider = record_num_spider.iloc[0, 0]

    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**quant_config))

    sql_statement = "select count(1) from `%s`" % table_config['quant']
    record_num_quant = pd.read_sql(sql_statement, quant_engine)
    record_num_quant = record_num_quant.iloc[0, 0]

    error_tables = []
    if record_num_spider == 0:
        error_tables.append(table_config['spider'])
    if (record_num_quant == 0) or (record_num_quant != record_num_spider):
        error_tables.append(table_config['quant'])

    return error_tables

# check area/industry excessive rise record num
def checkStockExcessiveRetsRecordNum(sql_config, tot_table_names, str_today):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    # get record num from db
    table_codes = {}
    for (tmp_name, tmp_table_name) in tot_table_names.items():
        if tmp_name == 'area':
            continue
        sql_statement = "select `code` from %s where (date = '%s')" % (tmp_table_name, str_today)
        tmp_codes = pd.read_sql(sql_statement, sql_engine)  # read record num
        table_codes[tmp_name] = tmp_codes['code']

    # area (no date)
    sql_statement = "select `code` from %s" % (tot_table_names['area'])
    tmp_codes = pd.read_sql(sql_statement, sql_engine)  # read record num
    table_codes['area'] = tmp_codes['code']

    error_tables = []
    # stock technical inner join category
    for tmp_c in ['area', 'industry']:
        tmp_final_codes = table_codes[tmp_c].loc[table_codes[tmp_c].isin(table_codes['stock_technical'])]
        tmp_num = tmp_final_codes.size
        if table_codes[tmp_c+'_exc_ret'].size != tmp_num:
            error_tables.append(tmp_c+'_exc_ret')

    return error_tables

# check stock quotes record num (by comparing to record num of tushare)
def checkStockQuoteRecordNum(sql_config, table_name, str_today):
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    sql_statement = "select count(1) from `%s` where `date`='%s'" % (table_name, str_today)

    record_num = pd.read_sql(sql_statement, sql_engine)
    record_num = record_num.iloc[0, 0]

    # get today's quotes from tushare
    ts_stock_quote = ts.get_today_all()
    ts_stock_quote = ts_stock_quote.drop_duplicates('code')
    ts_stock_quote = ts_stock_quote.loc[ts_stock_quote['open'] != 0]  # drop stop stocks

    error_table = []
    if record_num < ts_stock_quote.shape[0]: # error if record num is less than tushare
        error_table.append(table_name)
    return error_table

# =====================================================================================================
def getTodayStr(sql_config, shift):
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    # fetch calendar
    sql_statement = "select date from %s" % calendarTableName
    calendar = pd.read_sql(sql_statement, sql_engine)
    calendar = calendar['date'].values

    today = datetime.now()
    str_today = datetime.strftime(today, '%Y-%m-%d')

    if np.sum(calendar == str_today) == 1:  #  today is trade day
        if shift != 0:
            str_today = calendar[np.where(calendar == str_today)[0][0] - shift]
    else:
        str_today = ''

    return str_today


def specialRecordCheckMorning():
    str_today = getTodayStr(ConfigQuant, 1)

    tot_error_tables = []
    if str_today != '':
        # check area/industry/market/all stocks ranking in quant db
        this_error_tables = checkRecodNumForRankingByCategory(ConfigQuant, rankTableConfig, str_today)
        tot_error_tables.extend(this_error_tables)

    return tot_error_tables

def specialRecordCheckAfternoon():
    str_today = getTodayStr(ConfigQuant, 0)

    tot_error_tables = []
    if str_today != '':
        # area/industry quote in quant db
        this_error_tables = checkCategoryIndexRecordNum(ConfigQuant, categoryIndexTableConfig, str_today)
        tot_error_tables.extend(this_error_tables)

        # check STOCK_DESCRIPTION in quant db
        this_error_tables = checkStockDescription(ConfigSpider2, ConfigQuant, stockDescriptionTableConfig)
        tot_error_tables.extend(this_error_tables)

        # check area/industry excessive rise record num (inner join with stock technical)
        this_error_tables = checkStockExcessiveRetsRecordNum(ConfigQuant, stockExcessiveRetsTableConfig, str_today)
        tot_error_tables.extend(this_error_tables)

        # check stock quote num by comparing with tushare
        this_error_tables = checkStockQuoteRecordNum(ConfigQuant, 'STOCK_FORWARD_ADJ_QUOTE', str_today)
        tot_error_tables.extend(this_error_tables)

    return tot_error_tables

if __name__ == '__main__':
    # specialRecordCheckMorning()
    specialRecordCheckAfternoon()