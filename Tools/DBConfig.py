# ================ DB config ============================
ConfigQuant = {
    'host':'10.46.228.175',
    'user':'alg',
     'password':'Alg#824',
     'db':'quant',
     #'port': 3306,
     'charset':'utf8'
}

# ================== Process config =======================


stockQuoteConf = {
    'table_name': 'STOCK_FORWARD_ADJ_QUOTE',
    'fields': ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
}


allMergeDataConf = [
    {
        'table_name': 'STOCK_INDUSTRY',
        'fields': ['date', 'code', 'industry'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix':'',
        'condition': ''
    },

    {
        'table_name': 'STOCK_FUNDAMENTAL_TTM',
        'fields': ['date', 'code', 'PE_TTM', 'PS_TTM'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'STOCK_FUNDAMENTAL_BASIC',
        'fields': ['date', 'code', 'PB', 'TOT_MRK_CAP', 'FREE_MRK_CAP'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_TECH_INDICATORS',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['TURNOVER_ROLLING_SUM_1D', 'time_stamp'],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_AREA',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'AREA',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_INDUSTRY',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'INDUSTRY',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_MARKET',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'MARKET',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_ALL_STOCKS',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'ALL',
        'condition': ''
    },

    {
        'table_name': 'DERI_AREA_EXCESSIVE_RET',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'AREA',
        'condition': ''
    },

    {
        'table_name': 'DERI_INDUSTRY_EXCESSIVE_RET',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix':'INDUSTRY',
        'condition': ''
    },

    {
        'table_name': 'DERI_MARKET_EXCESSIVE_RET',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix': 'MARKET',
        'condition': ''
    },

    {
        'table_name': 'DERI_HS300_EXCESSIVE_RET',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix': 'HS300',
        'condition': ''
    },

    {
        'table_name': 'STOCK_BETA',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': ['time_stamp'],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'STOCK_DAY_COUNT',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'STOCK_VALUE',
        'fields': '*',
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'PRECIOUS_METAL',
        'fields': ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'AU',
        'condition': "`code`='AU99.99'"
    },

    {
        'table_name': 'FUND_INDEX',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'FUND',
        'condition': ""
    },

    {
        'table_name': 'DERI_AU',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'AU',
        'condition': ""
    },

    {
        'table_name': 'DERI_FUND_INDEX_SSE_C',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'FUND',
        'condition': ""
    },

    {
        'table_name': 'BRENT_QUOTE',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'BRENT',
        'condition': ""
    },

    {
        'table_name': 'SHIBOR_NEW',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR',
        'condition': ""
    },

    {
        'table_name': 'NASDAQ_COMPOSITE_QUOTE',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['change', 'time_stamp'],
        'prefix': 'NASDAQ',
        'condition': ""
    },

    {
        'table_name': 'SP500_QUOTE',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['change', 'time_stamp'],
        'prefix': 'SP500',
        'condition': ""
    },

    {
        'table_name': 'DOW_JONES_QUOTE',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['change', 'time_stamp'],
        'prefix': 'DOW',
        'condition': ""
    },

    {
        'table_name': 'DERI_BRENT',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'BRENT',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_ON',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_ON',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_1W',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_1D',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_2W',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_2D',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_1M',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_1M',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_3M',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_3M',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_6M',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_6M',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_9M',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_9M',
        'condition': ""
    },

    {
        'table_name': 'DERI_MACRO_SHIBOR_1Y',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SHIBOR_1Y',
        'condition': ""
    },

    {
        'table_name': 'DERI_NASDAQ',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'NASDAQ',
        'condition': ""
    },

    {
        'table_name': 'DERI_SP500',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'SP500',
        'condition': ""
    },

    {
        'table_name': 'DERI_DOW_JONES',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'DOW',
        'condition': ""
    },

    {
        'table_name': 'DONGFANGCAIFU_CPI',
        'fields': ['date', 'oncountrywide_YOY', 'oncountrywide_MOM'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'CPI',
        'condition': ""
    },
]


# merge  after categories are already merged
allMergeDataConf2 = [
    {
        'table_name': 'DERI_ALL_STOCK_RISE_RATIO',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'ALL',
        'condition': ""
    },

    {
        'table_name': 'DERI_AREA_RISE_RATIO',
        'fields': '*',
        'merge_fields': ['date', 'area'],
        'drop_fields': ['time_stamp'],
        'prefix': 'AREA',
        'condition': ""
    },

    {
        'table_name': 'DERI_INDUSTRY_RISE_RATIO',
        'fields': '*',
        'merge_fields': ['date', 'industry'],
        'drop_fields': ['time_stamp'],
        'prefix': 'INDUSTRY',
        'condition': ""
    },

    {
        'table_name': 'DERI_MARKET_RISE_RATIO',
        'fields': '*',
        'merge_fields': ['date', 'market'],
        'drop_fields': ['time_stamp'],
        'prefix': 'MARKET',
        'condition': ""
    },

    {
        'table_name': 'AREA_INDEX_FORWARD_ADJ_QUOTE',
        'fields': '*',
        'merge_fields': ['date', 'area'],
        'drop_fields': ['time_stamp'],
        'prefix': 'AREA',
        'condition': ""
    },

    {
        'table_name': 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE',
        'fields': '*',
        'merge_fields': ['date', 'industry'],
        'drop_fields': ['time_stamp'],
        'prefix': 'INDUSTRY',
        'condition': ""
    },

    {
        'table_name': 'STOCK_MARKET_INDEX_QUOTE',
        'fields': '*',
        'merge_fields': ['date', 'market'],
        'drop_fields': ['time_stamp'],
        'prefix': 'MARKET',
        'condition': ""
    },

    {
        'table_name': 'HS300_QUOTE',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'HS300',
        'condition': ""
    },

    {
        'table_name': 'DERI_AREA_INDEX_FOR_ADJ',
        'fields': '*',
        'merge_fields': ['date', 'area'],
        'drop_fields': ['time_stamp'],
        'prefix': 'AREA',
        'condition': ""
    },

    {
        'table_name': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'fields': '*',
        'merge_fields': ['date', 'industry'],
        'drop_fields': ['time_stamp'],
        'prefix': 'INDUSTRY',
        'condition': ""
    },

    {
        'table_name': 'DERI_MARKET_INDEX',
        'fields': '*',
        'merge_fields': ['date', 'market'],
        'drop_fields': ['TURNOVER_ROLLING_SUM_1D', 'TURNOVER_ROLLING_SUM_2D', 'TURNOVER_ROLLING_SUM_5D',
                        'TURNOVER_ROLLING_SUM_10D', 'TURNOVER_ROLLING_SUM_20D', 'TURNOVER_ROLLING_SUM_30D',
                        'TURNOVER_ROLLING_SUM_60D', 'time_stamp'],
        'prefix': 'MARKET',
        'condition': ""
    },

    {
        'table_name': 'DERI_HS300',
        'fields': '*',
        'merge_fields': ['date'],
        'drop_fields': ['time_stamp'],
        'prefix': 'HS300',
        'condition': ""
    },
]

areaConf = {
    'table_name': 'STOCK_DESCRIPTION',
    'fields': ['code', 'area'],
    'merge_fields': ['code'],
}