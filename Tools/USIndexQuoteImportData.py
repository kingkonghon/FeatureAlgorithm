import pandas as pd
import os
import sys
from sqlalchemy import create_engine
from datetime import  datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

fileName = ['DowJones.csv', 'SP500.csv', 'Nasdaq.csv']
indexCode = ['.DJI', '.INX', '.IXIC']
targetTableName = ['DOW_JONES_QUOTE', 'SP500_QUOTE', 'NASDAQ_COMPOSITE_QUOTE']

col_dict = {r'日期':'date', r'最新': 'close', r'开盘':'open', r'高':'high', r'低':'low', r'交易量':'volume',
            r'百分比变化':'change'}

chgDataTypeColName = ['open', 'high', 'low', 'close']
finalCols = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']

def changeColDataType(data, cols):
    for col in cols:
        data.loc[:, col] = data[col].apply(lambda x: float(x.replace(',','')))
    return  data

def changeDateFormaat(ori_date):
    dt_date = ori_date.apply(lambda x: datetime.strptime(x, r'%Y年%m月%d日'))
    str_date = dt_date.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    return str_date

if __name__ == '__main__':
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    for (name,code,table) in zip(fileName, indexCode, targetTableName):
        data = pd.read_csv(name)
        data = data.rename(columns=col_dict)

        # change data type
        data.loc[data['volume'] == '-', 'volume'] = '0'
        data.loc[:, 'volume'] = data['volume'].apply(
            lambda x: float(x[:-1]) * 100000000 if x[-1] == u'B' else (
                float(x[:-1]) * 10000 if x[-1] == u'M' else float(x)))
        data.loc[:,'change'] = data['change'].apply(lambda x: float(x.strip('%')) / 100.)
        data = changeColDataType(data, chgDataTypeColName)
        data.loc[:, 'date'] = changeDateFormaat(data['date'])

        # add index code
        # data['code'] = code

        # rearrange data
        data = data[finalCols]
        data = data.sort_values('date')

        # add timestamp
        data['time_stamp'] = datetime.now()

        # dump data into db
        data.to_sql(table, quant_engine, index=False, if_exists='replace')

