import h5py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Utils.DBOperation import writeDB
from Utils.DB_config import ConfigQuant

def read_h5_industry_code(file_path, file_name):
    file_path = os.path.join(file_path, file_name)
    h5_file = h5py.File(file_path)

    # dataset
    ds_data = h5_file['data']
    data = ds_data[...]

    h5_file.close()

    data = data.T[0]
    data = list(map(lambda x: x.decode('utf-8'), data)) # convert to chinese
    data = list(map(lambda x: x[:-4], data)) # remove substring
    data = pd.DataFrame(data, columns=['industry'])
    data['indcode'] = range(1, data.shape[0]+1)
    return data

def read_h5_industry(file_path, file_name):
    file_path = os.path.join(file_path, file_name)
    h5_file = h5py.File(file_path)

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
    return data, code, tradedates


def write_industry_to_db(data_indcode, data_industry, code, tradedates, chunk_size, target_table_name, db_config):
    method = 'replace'
    for i in range(int(tradedates.size / chunk_size) + 1):
        tmp_data = data_industry[i*chunk_size : (i+1)*chunk_size]
        tmp_tradedates = tradedates[i*chunk_size : (i+1)*chunk_size]

        # unpivot data
        tmp_data = pd.DataFrame(tmp_data, columns=code)
        tmp_data['date'] = tmp_tradedates
        new_data = pd.melt(tmp_data, id_vars='date', var_name='code', value_name='indcode')

        new_data = new_data.loc[new_data['indcode'] != 0]
        new_data = new_data.merge(data_indcode, on=['indcode'])
        new_data = new_data.drop('indcode', axis=1)

        new_data = new_data.sort_values('date')

        new_data['time_stamp'] = datetime.strptime(tradedates[-1], '%Y-%m-%d') + timedelta(days=1, hours=5) # adjust time stamp to concur with spider
        writeDB(target_table_name, new_data, db_config, method=method)
        method = 'append' # except the first time is 'replace


if __name__ == '__main__':
    file_path = ''
    file_name_indcode = 'LZ_CN_STKA_INDUCODE_SW.h5'
    file_name_industry = 'LZ_CN_STKA_INDU_SW.h5'
    chunk_size = 100
    target_table_name = 'STOCK_INDUSTRY'

    # read industry code dictionary
    data_indcode = read_h5_industry_code(file_path, file_name_indcode)
    # read stock industry code
    data_industry, code, tradedates = read_h5_industry(file_path, file_name_industry)

    # write sql
    write_industry_to_db(data_indcode, data_industry, code, tradedates, chunk_size, target_table_name, ConfigQuant)
    pass