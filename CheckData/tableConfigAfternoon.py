import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant, ConfigSpider2

tableConfig ={
    # middle dag
    'Shibor_spider':{
        'table_name': 'shibor',
        'date_field': 'left(ReportDate,10)',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['left(ReportDate,10)'],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Shibor_quant': {
        'table_name': 'SHIBOR_NEW',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_ON': {
        'table_name': 'DERI_MACRO_SHIBOR_ON',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_1W': {
        'table_name': 'DERI_MACRO_SHIBOR_1W',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_2W': {
        'table_name': 'DERI_MACRO_SHIBOR_2W',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_1M': {
        'table_name': 'DERI_MACRO_SHIBOR_1M',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_3M': {
        'table_name': 'DERI_MACRO_SHIBOR_3M',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_6M': {
        'table_name': 'DERI_MACRO_SHIBOR_6M',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_9M': {
        'table_name': 'DERI_MACRO_SHIBOR_9M',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_shibor_1Y': {
        'table_name': 'DERI_MACRO_SHIBOR_1Y',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },



    # afternoon dag
    'Stock_quotes_spider':{
        'table_name': 'EastMoneyHouFuQuan',
        'date_field': 'report_time',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['report_time', 'code'],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Stock_quotes_quant':{
        'table_name': 'STOCK_FORWARD_ADJ_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_fundamental_spider': {
        'table_name': 'tushare_real_time_market',
        'date_field': 'left(time_stamp, 10)',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['left(time_stamp, 10)', 'code'],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Stock_fundamental_quant': {
        'table_name': 'STOCK_FUNDAMENTAL_BASIC',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Fund_index_spider': {
        'table_name': 'shangzhengjijinzhishu',
        'date_field': 'report_time',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['report_time', 'stock'],
        'condition': "stock=' 000011'",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Fund_index_quant': {
        'table_name': 'FUND_INDEX',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_technical': {
        'table_name': 'DERI_STOCK_TECH_INDICATORS',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_beta': {
        'table_name': 'STOCK_BETA',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_value': {
        'table_name': 'STOCK_VALUE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_day_count': {
        'table_name': 'STOCK_DAY_COUNT',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_fund_index': {
        'table_name': 'DERI_FUND_INDEX_SSE_C',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Stock_market_index_spider': {
        'table_name': 'EastMoneyIndex',
        'date_field': 'report_time',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['report_time', 'code'],
        'condition': "`code` in ('000001', '399001', '399005', '399006')",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Stock_market_index_quant': {
        'table_name': 'STOCK_MARKET_INDEX_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'HS300_spider': {
        'table_name': 'EastMoneyIndex',
        'date_field': 'report_time',
        'date_format': '%Y-%m-%d',
        'distinct_fields': ['report_time', 'code'],
        'condition': "`code`='000300'",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'HS300_quant': {
        'table_name': 'HS300_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_HS300': {
        'table_name': 'DERI_HS300',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Area_index_quote': {
        'table_name': 'AREA_INDEX_FORWARD_ADJ_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Industry_index_quote': {
        'table_name': 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_area': {
        'table_name': 'DERI_AREA_INDEX_FOR_ADJ',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_industry': {
        'table_name': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_market': {
        'table_name': 'DERI_MARKET_INDEX',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_market_excessive': {
        'table_name': 'DERI_MARKET_EXCESSIVE_RET',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_HS300_excessive': {
        'table_name': 'DERI_HS300_EXCESSIVE_RET',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_area_rise': {
        'table_name': 'DERI_AREA_RISE_RATIO',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_industry_rise': {
        'table_name': 'DERI_INDUSTRY_RISE_RATIO',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_market_rise': {
        'table_name': 'DERI_MARKET_RISE_RATIO',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    },
    'Deri_all_stock_rise': {
        'table_name': 'DERI_ALL_STOCK_RISE_RATIO',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 0,
        'db_conn_config': ConfigQuant
    }
}

tableDirection = {
    'Shibor_spider': ['Shibor_quant'],
    'Shibor_quant': ['Deri_shibor_ON', 'Deri_shibor_1W', 'Deri_shibor_2W', 'Deri_shibor_1M', 'Deri_shibor_3M', 'Deri_shibor_6M', 'Deri_shibor_9M', 'Deri_shibor_1Y'],
    'Stock_quotes_spider': ['Stock_quotes_quant'],
    'Stock_fundamental_spider': ['Stock_fundamental_quant'],
    'Fund_index_spider': ['Fund_index_quant'],
    'Stock_quotes_quant': ['Stock_technical', 'Stock_beta', 'Stock_value', 'Stock_day_count', 'Deri_market_excessive', 'Deri_HS300_excessive'],
    'Fund_index_quant': ['Deri_fund_index'],
    'Stock_market_index_spider': ['Stock_market_index_quant'],
    'Stock_market_index_quant': ['Deri_market'],
    'Deri_market': ['Deri_market_rise'],
    'HS300_spider': ['HS300_quant'],
    'HS300_quant': ['Deri_HS300'],
    'Deri_HS300' : ['Deri_all_stock_rise'],
    'Area_index_quote': ['Deri_area', 'Deri_area_rise'],
    'Industry_index_quote': ['Deri_industry', 'Deri_industry_rise']
}