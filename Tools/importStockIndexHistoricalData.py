# -*- coding:utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import tushare as ts
import h5py
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider, ConfigQuant, ConfigSpider2
from Utils.ProcessFunc import renameDF, chgDFDataType

#============================= import amount from H5, other fields from tushare  ====================================

# H5 data file name
H5DataFileNameDict = {
    'open':'LZ_CN_STKA_INDXQUOTE_OPEN.h5',
    'high':'LZ_CN_STKA_INDXQUOTE_HIGH.h5',
    'low':'LZ_CN_STKA_INDXQUOTE_LOW.h5',
    'close':'LZ_CN_STKA_INDXQUOTE_CLOSE.h5',
    'volume':'LZ_CN_STKA_INDXQUOTE_VOLUME.h5',
    'amount':'LZ_CN_STKA_INDXQUOTE_AMOUNT.h5'
}

# TARGET
targetTableNameMarketIndex = 'STOCK_MARKET_INDEX_QUOTE'
targetTableNameHS = 'HS300_QUOTE'
targetFields = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
marketToCode = {'SH': '000001', 'SZ':'399001', 'SMEB':'399005', 'GEB': '399006'}
codeHS300 = '000300'

targetNewTimeStamp = 'time_stamp'

def read_h5_data(file_path, file_name, trunk_date, start_date='2007-01-01'):
    full_path = os.path.join(file_path, file_name)
    h5_file = h5py.File(full_path)

    # dataset
    ds_data = h5_file['data']
    ds_code = h5_file['header']
    ds_tradedates = h5_file['date']

    data = ds_data[...]
    code = ds_code[...]
    tradedates = ds_tradedates[...]

    h5_file.close()

    code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), code)))  # remove the market signs
    tradedates = list(map(lambda x: str(x), tradedates))
    tradedates = np.array(list(map(lambda x: '-'.join([x[0:4], x[4:6], x[6:]]), tradedates)))

    # trunk data
    data = data[(tradedates <= trunk_date) & (tradedates >= start_date)]
    tradedates = tradedates[(tradedates <= trunk_date) & (tradedates >= start_date)]

    return data, code, tradedates

def combineQuoteFields(all_quote_fields, index_code, trade_dates):
    tmp_data = pd.DataFrame({'date': trade_dates})
    for (field_name, field_data) in all_quote_fields.items():
        tmp_field = field_data[['date', index_code]]
        tmp_field = tmp_field.rename(columns={index_code: field_name})
        tmp_data = tmp_data.merge(tmp_field, on='date')

    tmp_data = tmp_data[targetFields]
    tmp_data = tmp_data.loc[~tmp_data['close'].isnull()]  # drop nan data
    return tmp_data

def getIndexQuote(all_quote_fields, trade_dates, end_date, start_date='2007-01-01'):
    market_index_quote = pd.DataFrame()
    for (market, index_code) in marketToCode.items():
        # get '399001.SZ' from tushare
        if market == 'SZ':
            tmp_data = ts.get_k_data(code=index_code, ktype='D', autype=None, index=True, start=start_date, end=end_date)
            tmp_data.loc[:, 'amount'] = np.nan
            tmp_data = tmp_data[targetFields]
        else:
            tmp_data = combineQuoteFields(all_quote_fields, index_code, trade_dates)

        tmp_data.loc[:, 'market'] = market
        market_index_quote = market_index_quote.append(tmp_data)

    HS300_quote = combineQuoteFields(all_quote_fields, codeHS300, trade_dates)

    return market_index_quote, HS300_quote


if __name__ == '__main__':
    # create target engine
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    folder_path = ''
    end_date = '2016-01-20'
    all_quote_fields = {}
    for (field, file_name) in H5DataFileNameDict.items():
        tmp_index_quote_field, index_code, trade_dates = read_h5_data(folder_path, file_name, end_date)

        df_tmp_index_quote_field = pd.DataFrame(tmp_index_quote_field, columns=index_code)
        df_tmp_index_quote_field.loc[:, 'date'] = trade_dates

        all_quote_fields[field] = df_tmp_index_quote_field

    market_index_quote, HS300_quote = getIndexQuote(all_quote_fields, trade_dates, end_date)

    # add time stamp
    market_index_quote.loc[:, targetNewTimeStamp] = datetime.now()
    HS300_quote.loc[:,targetNewTimeStamp] = datetime.now()

    # dump data into DB
    market_index_quote.to_sql(targetTableNameMarketIndex, quant_engine, index=False, if_exists='replace')

    HS300_quote.to_sql(targetTableNameHS, quant_engine, index=False, if_exists='replace')
