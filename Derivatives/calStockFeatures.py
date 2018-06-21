from Mapping.StockTechnicalIndicatorMapping import StockTechnicalIndicatorMapping
from Mapping.ExcessReturnMapping import ExcessReturnMapping
from Mapping.StockOtherFeaturesMapping import StockOtherFeaturesMapping
import time

ConTech = {
    'sourceTableName': 'STOCK_FORWARD_ADJ_QUOTE',
    'codeField': 'code',
    'dateField': 'date',
    'openField': 'open',
    'highField': 'high',
    'lowField': 'low',
    'closeField': 'close',
    'volumeField': 'volume',
    'turnoverField': 'turnover',
    'lags': [1,2,5,10,20,30,60],
    'targetTableName': 'DERI_STOCK_TECH_INDICATORS',
    'condition': '',
    'chunkSize': 10,
    'isMultiProcess': True
}

ConExc = [
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_AREA_INDEX_FOR_ADJ',
        'sourceCategoryTableName': 'STOCK_DESCRIPTION',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'area',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,  # not every stock has area (merge)
        'targetTableName': 'DERI_AREA_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'sourceCategoryTableName': 'STOCK_INDUSTRY',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'industry',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,  # not every stock has industry (merge)
        'targetTableName': 'DERI_INDUSTRY_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_MARKET_INDEX',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'market',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_MARKET_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_HS300',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'HS300',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_HS300_EXCESSIVE_RET',
        'chunkSize': 10
    }
]

ConOthers = [
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_DESCRIPTION',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'area',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,   # not every stock has area field
        'targetTableName': 'DERI_STOCK_RANKING_AREA',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_INDUSTRY',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'industry',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,   # not every stock has industry field
        'targetTableName': 'DERI_STOCK_RANKING_INDUSTRY',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'market',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_STOCK_RANKING_MARKET',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'all',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_STOCK_RANKING_ALL_STOCKS',
        'chunkSize': 10
    },
]

# afternoon tasks
def airflowCallableStockTechnicalInicators():
    features = StockTechnicalIndicatorMapping(**ConTech)
    features.run()

def airflowCallableStockExcessiveReturnArea():
    con  = ConExc[0]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnIndustry():
    con  = ConExc[1]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnMarket():
    con  = ConExc[2]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnHS300():
    con  = ConExc[3]
    features = ExcessReturnMapping(**con)
    features.run()

# evening task
def airflowCallableRankingArea():
    con = ConOthers[0]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingIndustry():
    con = ConOthers[1]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingMarket():
    con = ConOthers[2]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingAllStocks():
    con = ConOthers[3]
    features = StockOtherFeaturesMapping(**con)
    features.run()

if __name__ == '__main__':
    start_time = time.clock()

    # features = StockTechnicalIndicatorMapping(**ConTech)
    # features.run()

    # for i in ConExc:
    #     features = ExcessReturnMapping(**i)
    #     features.run()
    #

    for i in ConOthers:
        features = StockOtherFeaturesMapping(**i)
        features.run()
    airflowCallableRankingAllStocks()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)