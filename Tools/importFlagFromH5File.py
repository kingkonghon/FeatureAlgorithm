import pandas as pd
import os
import sys
from sqlalchemy import create_engine
from datetime import  datetime
import h5py
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigSpider, ConfigQuant
from Utils.ProcessFunc import renameDF, chgDFDataType

h5FileName = ['LZ_CN_STKA_SLCIND_ST_FLAG.h5', 'LZ_CN_STKA_INDEX_CSI500WEIGHT.h5', 'LZ_CN_STKA_INDEX_HS300WEIGHT.h5', 'LZ_CN_STKA_SLCIND_TRADEDAYCOUNT.h5']
valueFields = ['ST_FLAG', 'CSI500_FLAG', 'HS300_FLAG', 'LISTDATE_COUNT']

targetTableName = 'STOCK_FLAG'


def read_h5_data(file_path, file_name, trunk_date):
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

    # trunk data
    data = data[tradedates <= trunk_date]
    tradedates = tradedates[tradedates <= trunk_date]

    code = np.array(list(map(lambda x: x[-6:].decode('utf-8'), code)))  # remove the market signs
    tradedates = list(map(lambda x: str(x), tradedates))
    tradedates = np.array(list(map(lambda x: '-'.join([x[0:4], x[4:6], x[6:]]), tradedates)))
    return data, code, tradedates

def unpivot2DData(data, code, tradedates, value_name):
    tmp_df = pd.DataFrame(data, columns=code)
    tmp_df['date'] = tradedates
    new_data = pd.melt(tmp_df, id_vars='date', var_name='code', value_name=value_name) # unpivot
    new_data = new_data.loc[~new_data[value_name].isnull()] # drop nan
    return new_data

if __name__ == '__main__':
    quant_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    file_path = ''

    chunk_size = 10000

    start_date = '2007-01-01'
    trunk_date = 20180619 # warning:!! the last row is nan

    all_data = pd.DataFrame([])
    for (file_name, field) in zip(h5FileName, valueFields):
        tmp_data, code, tradedates = read_h5_data(file_path, file_name,trunk_date)
        if field in ['CSI500_FLAG', 'HS300_FLAG']:
            tmp_data[~np.isnan(tmp_data)] = 1  #  set stocks with weight as 1
            tmp_data[np.isnan(tmp_data)] = 0  # set others as 0
        tmp_data = unpivot2DData(tmp_data, code, tradedates, field)

        if all_data.empty:
            all_data = tmp_data.copy()
        else:
            all_data = all_data.merge(tmp_data, on=['date', 'code'], how='outer')

    # trim data
    all_data = all_data.loc[all_data['date'] >= start_date]

    # drop records that are not listed or delisted (listday_count == 0)
    all_data = all_data.loc[all_data['LISTDATE_COUNT'] > 0]
    all_data = all_data.drop('LISTDATE_COUNT', axis=1)

    # add timestamp
    all_data['time_stamp'] = datetime.now()

    # dump data into db chunk by chunk
    write_method = 'replace'
    for i in range(int(all_data.shape[0] / chunk_size) + 1):
        tmp_data = all_data.iloc[i*chunk_size : (i+1)*chunk_size]
        tmp_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
        write_method = 'append'

