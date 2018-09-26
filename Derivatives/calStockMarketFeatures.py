from Mapping.StockMarketFeaturesMapping import StockMarketFeaturesMapping
import time
from multiprocessing import Pool

CON = [
    {
        'sourceStockTechIndiTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_DESCRIPTION',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'area',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'targetTableName': 'DERI_AREA_RISE_RATIO',
        'chunkSize': 10
    },
    {
        'sourceStockTechIndiTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_INDUSTRY',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'industry',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'targetTableName': 'DERI_INDUSTRY_RISE_RATIO',
        'chunkSize': 10
    },
    {
        'sourceStockTechIndiTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'market',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'targetTableName': 'DERI_MARKET_RISE_RATIO',
        'chunkSize': 10
    },
    {
        'sourceStockTechIndiTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'all',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'targetTableName': 'DERI_ALL_STOCK_RISE_RATIO',
        'chunkSize': 10
    }
]

def airflowCallableArea():
    con = CON[0]
    features = StockMarketFeaturesMapping(**con)
    features.run()

def airflowCallableIndustry():
    con = CON[1]
    features = StockMarketFeaturesMapping(**con)
    features.run()

def airflowCallableMarket():
    con = CON[2]
    features = StockMarketFeaturesMapping(**con)
    features.run()

def airflowCallableAllStock():
    con = CON[3]
    features = StockMarketFeaturesMapping(**con)
    features.run()


if __name__ == '__main__':
    start_time = time.clock()
    # airflowCallableAllStock()
    # pool = Pool(processes=4)
    # tot_res = []
    for i in CON:
        features = StockMarketFeaturesMapping(**i)
        features.run()
        # res = pool.apply_async(features.run)
        # tot_res.append(res)

    # for (i, res) in enumerate(tot_res):
    #     res.get()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)