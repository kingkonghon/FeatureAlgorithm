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
        'fields': ['date', 'code', 'RET_2D', 'RET_5D', 'RET_20D', 'RET_60D', 'STD_5D', 'STD_20D', 'STD_60D', 'TURNOVER_ROLLING_SUM_5D',
                   'TURNOVER_ROLLING_SUM_60D', 'OBV_DIFF_5D', 'OBV_DIFF_20D', 'EMA_5D', 'EMA_20D', 'RSI_21D', 'MACD_12_26_14D'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': '',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_INDUSTRY',
        'fields': ['date', 'code', 'RET_5D_RANK', 'RET_20D_RANK'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix':'INDUSTRY',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_MARKET',
        'fields': ['date', 'code', 'RET_5D_RANK', 'RET_20D_RANK'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix':'MARKET',
        'condition': ''
    },

    {
        'table_name': 'DERI_STOCK_RANKING_ALL_STOCKS',
        'fields': ['date', 'code', 'RET_5D_RANK', 'RET_20D_RANK'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix':'ALL',
        'condition': ''
    },

    {
        'table_name': 'DERI_INDUSTRY_EXCESSIVE_RET',
        'fields': ['date', 'code', 'EXCESSIVE_RET_5D', 'EXCESSIVE_RET_20D'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix':'INDUSTRY',
        'condition': ''
    },

    {
        'table_name': 'DERI_MARKET_EXCESSIVE_RET',
        'fields': ['date', 'code', 'EXCESSIVE_RET_5D', 'EXCESSIVE_RET_20D'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': 'MARKET',
        'condition': ''
    },

    {
        'table_name': 'DERI_HS300_EXCESSIVE_RET',
        'fields': ['date', 'code', 'EXCESSIVE_RET_5D', 'EXCESSIVE_RET_20D'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
        'prefix': 'HS300',
        'condition': ''
    },

    {
        'table_name': 'STOCK_BETA',
        'fields': ['date', 'code', 'beta_5', 'beta_20'],
        'merge_fields': ['date', 'code'],
        'drop_fields': [],
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
        'table_name': 'DERI_AU',
        'fields': ['date', 'RET_1D', 'RET_5D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'AU',
        'condition': ""
    },

    {
        'table_name': 'SHIBOR_NEW',
        'fields': ['date', 'O_N', '1M', '6M', '1Y'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'SHIBOR',
        'condition': ""
    },

    {
        'table_name': 'DERI_NASDAQ',
        'fields': ['date', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'NASDAQ',
        'condition': ""
    },

    {
        'table_name': 'DERI_SP500',
        'fields': ['date', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'SP500',
        'condition': ""
    },

    {
        'table_name': 'DERI_DOW_JONES',
        'fields': ['date', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date'],
        'drop_fields': [],
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
        'fields': ['date', 'RATIO_RET_1D', 'RATIO_RET_5D', 'RATIO_RET_20D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'ALL',
        'condition': ""
    },

    {
        'table_name': 'DERI_INDUSTRY_RISE_RATIO',
        'fields': ['date', 'industry', 'RATIO_RET_1D', 'RATIO_RET_5D', 'RATIO_RET_20D'],
        'merge_fields': ['date', 'industry'],
        'drop_fields': [],
        'prefix': 'INDUSTRY',
        'condition': ""
    },

    {
        'table_name': 'DERI_MARKET_RISE_RATIO',
        'fields': ['date', 'market', 'RATIO_RET_1D', 'RATIO_RET_5D', 'RATIO_RET_20D'],
        'merge_fields': ['date', 'market'],
        'drop_fields': [],
        'prefix': 'MARKET',
        'condition': ""
    },

    {
        'table_name': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'fields': ['date', 'industry', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date', 'industry'],
        'drop_fields': [],
        'prefix': 'INDUSTRY',
        'condition': ""
    },

    {
        'table_name': 'DERI_MARKET_INDEX',
        'fields': ['date', 'market', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date', 'market'],
        'drop_fields': [],
        'prefix': 'MARKET',
        'condition': ""
    },

    {
        'table_name': 'DERI_HS300',
        'fields': ['date', 'RET_1D', 'RET_5D', 'RET_20D'],
        'merge_fields': ['date'],
        'drop_fields': [],
        'prefix': 'HS300',
        'condition': ""
    },
]

areaConf = {
    'table_name': 'STOCK_DESCRIPTION',
    'fields': ['code', 'area'],
    'merge_fields': ['code'],
}