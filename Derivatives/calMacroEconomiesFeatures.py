from Mapping.MacroEconomiesMapping import MacroEconomiesMapping
import time

CONF = [
    {
        'sourceTableName': 'BRENT_QUOTE',
        'dateField': 'date',
        'valueField': 'close',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_BRENT',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': 'O_N',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_ON',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '1D',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_1W',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '2D',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_2W',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '1M',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_1M',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '3M',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_3M',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '6M',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_6M',
        'condition': ''
    },
{
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '9M',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_9M',
        'condition': ''
    },
    {
        'sourceTableName': 'SHIBOR_NEW',
        'dateField': 'date',
        'valueField': '1Y',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_MACRO_SHIBOR_1Y',
        'condition': ''
    }
]

ConUSIndex = [
    {
        'sourceTableName': 'NASDAQ_COMPOSITE_QUOTE',
        'dateField': 'date',
        'valueField': 'close',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_NASDAQ',
        'condition': ''
    },
    {
        'sourceTableName': 'SP500_QUOTE',
        'dateField': 'date',
        'valueField': 'close',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_SP500',
        'condition': ''
    },
    {
        'sourceTableName': 'DOW_JONES_QUOTE',
        'dateField': 'date',
        'valueField': 'close',
        'lags': [1,2,5,10,20,30,60],
        'alignFlag': True,
        'targetTableName': 'DERI_DOW_JONES',
        'condition': ''
    }
]

def airflowCallableOil():
    conf = CONF[0]
    features = MacroEconomiesMapping(**conf)
    features.run()


def airflowCallableShibor():
    conf = CONF[1:]
    for i in conf:
        features = MacroEconomiesMapping(**i)
        features.run()

def airflowCaCallableUSStockIndex():
    for i in ConUSIndex:
        features = MacroEconomiesMapping(**i)
        features.run()

if __name__ == '__main__':
    start_time = time.clock()
    # for i in CONF:
    #     features = MacroEconomiesMapping(**i)
    #     features.run()

    # for i in ConUSIndex:
    #     features = MacroEconomiesMapping(**i)
    #     features.run()

    airflowCallableShibor()
    # airflowCallableOil()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)
