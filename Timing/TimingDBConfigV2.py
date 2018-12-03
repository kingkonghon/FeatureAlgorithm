# ================ DB config ============================
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

# ================== Process config =======================


HS300QuoteConf = {
    'table_name': 'HS300_QUOTE',
    'date_field': 'date'
}


allMergeDataConf = [
    {
        'table_name': 'DERI_HS300',
        'fields': ['date', 'MA_DIFF_SIGN_2D', 'MA_DIFF_PLUS_COUNT_2D', 'MA_DIFF_NEGATIVE_COUNT_2D', 'MA_DIFF_CON_PLUS_COUNT_2D', 'MA_DIFF_CON_NEGATIVE_COUNT_2D',
                   'MA_DIFF_SIGN_5D', 'MA_DIFF_PLUS_COUNT_5D', 'MA_DIFF_NEGATIVE_COUNT_5D', 'MA_DIFF_CON_PLUS_COUNT_5D', 'MA_DIFF_CON_NEGATIVE_COUNT_5D',
                   'MA_DIFF_SIGN_10D', 'MA_DIFF_PLUS_COUNT_10D', 'MA_DIFF_NEGATIVE_COUNT_10D', 'MA_DIFF_CON_PLUS_COUNT_10D', 'MA_DIFF_CON_NEGATIVE_COUNT_10D',
                   'MA_DIFF_SIGN_20D', 'MA_DIFF_PLUS_COUNT_20D', 'MA_DIFF_NEGATIVE_COUNT_20D', 'MA_DIFF_CON_PLUS_COUNT_20D', 'MA_DIFF_CON_NEGATIVE_COUNT_20D'
                   ],
        'merge_fields': ['date'],
        'drop_fields': [],
        'rounding_fields':{},
        'discrete_fields': [],
        'one_hot_fields': ['MA_DIFF_PLUS_COUNT_2D', 'MA_DIFF_NEGATIVE_COUNT_2D', 'MA_DIFF_CON_PLUS_COUNT_2D', 'MA_DIFF_CON_NEGATIVE_COUNT_2D',
                           'MA_DIFF_PLUS_COUNT_5D', 'MA_DIFF_NEGATIVE_COUNT_5D', 'MA_DIFF_CON_PLUS_COUNT_5D','MA_DIFF_CON_NEGATIVE_COUNT_5D',
                           'MA_DIFF_PLUS_COUNT_10D', 'MA_DIFF_NEGATIVE_COUNT_10D','MA_DIFF_CON_PLUS_COUNT_10D', 'MA_DIFF_CON_NEGATIVE_COUNT_10D',
                           'MA_DIFF_PLUS_COUNT_20D', 'MA_DIFF_NEGATIVE_COUNT_20D', 'MA_DIFF_CON_PLUS_COUNT_20D', 'MA_DIFF_CON_NEGATIVE_COUNT_20D'],
        'prefix':'',
        'condition': ''
    },

    {
        'table_name': 'DERI_STAT_HS300',
        'fields': ['date', #'PRICE_TO_MA_2D', 'PRICE_TO_MA_5D', 'PRICE_TO_MA_10D', 'PRICE_TO_MA_20D', 'PRICE_TO_MA_60D', 'PRICE_TO_MA_120D', 'PRICE_TO_MA_250D',
                   # 'AMTMA_DIFF_2D', 'AMTMA_DIFF_SIGN_2D', 'AMTMA_DIFF_PLUS_COUNT_2D', 'AMTMA_DIFF_NEGATIVE_COUNT_2D', 'AMTMA_DIFF_CON_PLUS_COUNT_2D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_2D',
                   # 'AMTMA_DIFF_5D', 'AMTMA_DIFF_SIGN_5D', 'AMTMA_DIFF_PLUS_COUNT_5D', 'AMTMA_DIFF_NEGATIVE_COUNT_5D', 'AMTMA_DIFF_CON_PLUS_COUNT_5D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_5D',
                   # 'AMTMA_DIFF_10D', 'AMTMA_DIFF_SIGN_10D', 'AMTMA_DIFF_PLUS_COUNT_10D', 'AMTMA_DIFF_NEGATIVE_COUNT_10D', 'AMTMA_DIFF_CON_PLUS_COUNT_10D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_10D',
                   # 'AMTMA_DIFF_20D', 'AMTMA_DIFF_SIGN_20D', 'AMTMA_DIFF_PLUS_COUNT_20D', 'AMTMA_DIFF_NEGATIVE_COUNT_20D', 'AMTMA_DIFF_CON_PLUS_COUNT_20D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_20D',
                   # 'AMTMA_DIFF_20D', 'AMTMA_DIFF_SIGN_20D', 'AMTMA_DIFF_PLUS_COUNT_20D', 'AMTMA_DIFF_NEGATIVE_COUNT_20D', 'AMTMA_DIFF_CON_PLUS_COUNT_20D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_20D',
                   # 'AMTMA_DIFF_20D', 'AMTMA_DIFF_SIGN_20D', 'AMTMA_DIFF_PLUS_COUNT_20D', 'AMTMA_DIFF_NEGATIVE_COUNT_20D', 'AMTMA_DIFF_CON_PLUS_COUNT_20D', 'AMTMA_DIFF_CON_NEGATIVE_COUNT_20D',
                   # 'AMTMA_DIFF_10D', 'AMTMA_DIFF_SIGN_10D', 'AMTMA_DIFF_PLUS_COUNT_10D', 'AMTMA_DIFF_NEGATIVE_COUNT_10D', 'AMTMA_DIFF_CON_PLUS_COUNT_10D','AMTMA_DIFF_CON_NEGATIVE_COUNT_10D',
                   # 'AMTSTD_10D', 'AMTSTD_20D', 'AMTSTD_30D', 'AMTSTD_60D', 'AMTSTD_120D', 'AMTSTD_250D', ''
                   'AMOUNT_RATIO_2D', 'AMOUNT_RATIO_5D', 'AMOUNT_RATIO_10D', 'AMOUNT_RATIO_20D', 'AMOUNT_RATIO_250D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'rounding_fields': {'AMOUNT_RATIO_2D': 1, 'AMOUNT_RATIO_5D':1, 'AMOUNT_RATIO_10D':1, 'AMOUNT_RATIO_20D':1, 'AMOUNT_RATIO_250D':1},
        'discrete_fields': [],
        'one_hot_fields': ['AMOUNT_RATIO_2D', 'AMOUNT_RATIO_5D', 'AMOUNT_RATIO_10D', 'AMOUNT_RATIO_20D', 'AMOUNT_RATIO_250D'],
        'prefix':'',
        'condition': ''
    },
    {
        'table_name': 'FINANCE_CASH_AND_SECURITY',
        'fields': ['date', 'OUTSTANDING_CASH_TO_FREE_CAP'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'rounding_fields':{'OUTSTANDING_CASH_TO_FREE_CAP':1},
        'discrete_fields': [],
        'one_hot_fields': ['OUTSTANDING_CASH_TO_FREE_CAP'],
        'prefix':'',
        'condition': ''
    },

    {
        'table_name': 'MACROECONOMIC_TUSHARE',
        'fields': ['date', 'm2_yoy', 'ppi'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'rounding_fields':{'m2_yoy':0, 'ppi':1},
        'discrete_fields': [],
        'one_hot_fields': ['m2_yoy', 'ppi'],
        'prefix':'',
        'condition': ''
    },

    {
        'table_name': 'DONGFANGCAIFU_CPI',
        'fields': ['date', 'oncountrywide_YOY'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'rounding_fields':{'oncountrywide_YOY':2},
        'discrete_fields': [],
        'one_hot_fields': ['oncountrywide_YOY'],
        'prefix':'',
        'condition': ''
    },
]