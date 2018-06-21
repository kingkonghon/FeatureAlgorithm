from Mapping.StockTechnicalIndicatorMapping import StockTechnicalIndicatorMapping
from Mapping.NonStockFeaturesMapping import NonStockFeaturesMapping
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

def airflowCallableStockIndexTechnical():
    for i in CONF:
        features = StockTechnicalIndicatorMapping(**i)
        features.run()


def airflowCallableHS300Technical():
    features = NonStockFeaturesMapping(**ConHS300)
    features.run()

if __name__ == '__main__':
    start_time = time.clock()
    for i in CONF:
        features = StockTechnicalIndicatorMapping(**i)
        features.run()

    features = NonStockFeaturesMapping(**ConHS300)
    features.run()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)