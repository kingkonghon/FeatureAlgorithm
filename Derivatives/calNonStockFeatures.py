from Mapping.NonStockFeaturesMapping import NonStockFeaturesMapping
import time

CONF = [
    {
        'sourceTableName': 'PRECIOUS_METAL',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': False,
        'targetTableName': 'DERI_AU',
        'condition': "`code` = 'Au99.99'"
    },
    {
        'sourceTableName': 'FUND_INDEX',
        'dateField': 'date',
        'openField': 'open',
        'highField': 'high',
        'lowField': 'low',
        'closeField': 'close',
        'volumeField': 'volume',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_FUND_INDEX_SSE_C',
        'condition': ""
    }
]

def airflowCallablePreciousMetal():
    features = NonStockFeaturesMapping(**CONF[0])
    features.run()

def airflowCallableFundIndex():
    features = NonStockFeaturesMapping(**CONF[1])
    features.run()


if __name__ == '__main__':
    # start_time = time.clock()
    # for i in CONF:
    #     features = NonStockFeaturesMapping(**i)
    #     features.run()
    #
    # eclapsed = time.clock() - start_time
    # print("Time eclapsed", eclapsed)
    # airflowCallableFundIndex()
    airflowCallablePreciousMetal()