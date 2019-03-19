import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Mapping.StockTechnicalIndicatorMapping import StockTechnicalIndicatorMapping
from Mapping.NonStockFeaturesMapping import NonStockFeaturesMapping
from Mapping.StockStationaryTechIndicatorMapping import StockStationaryTechnicalIndicatorMapping
from Mapping.NonStockStationaryFeaturesMapping import NonStockStationaryFeaturesMapping
import time

CONF = [
    {
        'sourceTableName': 'AREA_INDEX_FORWARD_ADJ_QUOTE',
        'codeField': 'area',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'turnoverField': 'turnover',
        'lags': [1,2,5,10,20,30,60],
        'targetTableName': 'DERI_AREA_INDEX_FOR_ADJ',
        'condition': '',
        'chunkSize': 10,
        'isMultiProcess': False
    },
    {
        'sourceTableName': 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE',
        'codeField': 'industry',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'turnoverField': 'turnover',
        'lags': [1,2,5,10,20,30,60],
        'targetTableName': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'condition': '',
        'chunkSize': 10,
        'isMultiProcess': False
    },
    {
        'sourceTableName': 'STOCK_MARKET_INDEX_QUOTE',
        'codeField': 'market',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'amountField': 'amount',
        'turnoverField': '',
        'lags': [1,2,5,10,20,30,60],
        'targetTableName': 'DERI_MARKET_INDEX',
        'alignFlag': False,
        'condition': "",
        'chunkSize': 10,
        'isMultiProcess': False
    },
]

ConHS300 = {
        'sourceTableName': 'HS300_QUOTE',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'amountField': 'amount',
        'turnoverField': '',
        'lags': [1,2,5,10,20,30,60],
        'targetTableName': 'DERI_HS300',
        'condition': ""
}

#===========================  stationary ========================
CONFStationary = [
    {
        'sourceTableName': 'AREA_INDEX_FORWARD_ADJ_QUOTE',
        'codeField': 'area',
        'dateField': 'date',
        'closeField': 'close',
        'amountField': 'amount',
        'lags': [1,2,5,10,20,30,60,120,250],
        'targetTableName': 'DERI_STAT_AREA_INDEX_FOR_ADJ',
        'condition': '',
        'chunkSize': 50,
        'isMultiProcess': False,
        'processNum': 4
    },
    {
        'sourceTableName': 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE',
        'codeField': 'industry',
        'dateField': 'date',
        'closeField': 'close',
        'amountField': 'amount',
        'lags': [1,2,5,10,20,30,60,120,250],
        'targetTableName': 'DERI_STAT_INDUSTRY_INDEX_FOR_ADJ',
        'condition': '',
        'chunkSize': 50,
        'isMultiProcess': False,
        'processNum': 4
    },
    {
        'sourceTableName': 'STOCK_MARKET_INDEX_QUOTE',
        'codeField': 'market',
        'dateField': 'date',
        'closeField': 'close',
        'amountField': 'amount',
        'lags': [1, 2, 5, 10, 20, 30, 60, 120, 250],
        'targetTableName': 'DERI_STAT_MARKET_INDEX',
        'condition': '',
        'chunkSize': 50,
        'isMultiProcess': False,
        'processNum': 4
    },
]

ConHS300Stationary = {
        'sourceTableName': 'HS300_QUOTE',
        'dateField': 'date',
        'closeField': 'close',
        'amountField': 'amount',
        'lags': [1, 2, 5, 10, 20, 30, 60, 120, 250],
        'targetTableName': 'DERI_STAT_HS300',
        'condition': '',
}


def airflowCallableStockIndexTechnical():
    for i in CONF:
        features = StockTechnicalIndicatorMapping(**i)
        features.run()


def airflowCallableHS300Technical():
    features = NonStockFeaturesMapping(**ConHS300)
    features.run()


def airflowCallableStockIndexStationaryTechnical():
    for i in CONFStationary:
        features = StockStationaryTechnicalIndicatorMapping(**i)
        features.run()

def airflowCallableHS300StationaryTechnical():
    features = NonStockStationaryFeaturesMapping(**ConHS300Stationary)
    features.run()


if __name__ == '__main__':
    start_time = time.clock()
    for i in CONF:
        features = StockTechnicalIndicatorMapping(**i)
        features.run()

    features = NonStockFeaturesMapping(**ConHS300)
    features.run()

    for i in CONFStationary:
        features = StockStationaryTechnicalIndicatorMapping(**i)
        features.run()

    features = NonStockStationaryFeaturesMapping(**ConHS300Stationary)
    features.run()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)