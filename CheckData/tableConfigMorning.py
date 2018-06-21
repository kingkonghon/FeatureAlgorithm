import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant, ConfigSpider2

tableConfig ={
    # evening dag
    'Stock_fundamental_TTM_spider':{
        'table_name': 'XueQiuStockTTM',
        'date_field': 'tradedate',
        'date_format': '%Y%m%d',
        'distinct_fields': ['tradedate', 'stockCode'],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigSpider2
    },
    'Stock_fundamental_TTM_quant': {
        'table_name': 'STOCK_FUNDAMENTAL_TTM',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Precious_metal_spider': {
        'table_name': 'guijinshu',
        'date_field': 'report_date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "contract='AU99.99'",
        'offset_day_num': 1,
        'db_conn_config': ConfigSpider2
    },
    'Precious_metal_quant': {
        'table_name': 'PRECIOUS_METAL',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "code='AU99.99'",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Deri_AU': {
        'table_name': 'DERI_AU',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },



    # morning dag
    'NASDAQ_spider':{
        'table_name': 'BaiduStockIndex',
        'date_field': 'left(time_stamp, 11)',
        'date_format': '%Y年%m月%d日',
        'distinct_fields': [],
        'condition': "stockCode='.IXIC'",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Dow_spider':{
        'table_name': 'BaiduStockIndex',
        'date_field': 'left(time_stamp, 11)',
        'date_format': '%Y年%m月%d日',
        'distinct_fields': [],
        'condition': "stockCode='.DJI'",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'SP500_spider': {
        'table_name': 'BaiduStockIndex',
        'date_field': 'left(time_stamp, 11)',
        'date_format': '%Y年%m月%d日',
        'distinct_fields': [],
        'condition': "stockCode='.INX'",
        'offset_day_num': 0,
        'db_conn_config': ConfigSpider2
    },
    'Brent_spider': {
        'table_name': 'SinaBulunte',
        'date_field': 'reportTime',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigSpider2
    },
    'NASDAQ_quant':{
        'table_name': 'NASDAQ_COMPOSITE_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Dow_quant':{
        'table_name': 'DOW_JONES_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'SP500_quant': {
        'table_name': 'SP500_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Brent_quant': {
        'table_name': 'BRENT_QUOTE',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Deri_Nasdaq': {
        'table_name': 'DERI_NASDAQ',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Deri_SP500': {
        'table_name': 'DERI_SP500',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Deri_Dow': {
        'table_name': 'DERI_DOW_JONES',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
    'Deri_Brent': {
        'table_name': 'DERI_BRENT',
        'date_field': 'date',
        'date_format': '%Y-%m-%d',
        'distinct_fields': [],
        'condition': "",
        'offset_day_num': 1,
        'db_conn_config': ConfigQuant
    },
}

tableDirection = {
    'Stock_fundamental_TTM_spider': ['Stock_fundamental_TTM_quant'],
    'Precious_metal_spider': ['Precious_metal_quant'],
    'Precious_metal_quant': ['Deri_AU'],
    'NASDAQ_spider': ['NASDAQ_quant'],
    'NASDAQ_quant': ['Deri_Nasdaq'],
    'Dow_spider': ['Dow_quant'],
    'Dow_quant': ['Deri_Dow'],
    'SP500_spider': ['SP500_quant'],
    'SP500_quant': ['Deri_SP500'],
    'Brent_spider': ['Brent_quant'],
    'Brent_quant': ['Deri_Brent']
}