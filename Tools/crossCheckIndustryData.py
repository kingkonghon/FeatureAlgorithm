import h5py
from sqlalchemy import create_engine
from Utils.DB_config import ConfigSpider2
from Utils.DBOperation import getDataFromSQL
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


sqlTableName = 'sws_research'
sqlFields = {
    'code':'stock',
    'date':'time_stamp',
    'industry':'industry_name'
}
filePath = ''
h5FileNameInd = 'LZ_CN_STKA_INDU_SW.h5'
h5FileNameIndDict = 'LZ_CN_STKA_INDUCODE_SW.h5'



def read_h5_industry_dict(file_path, file_name):
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



if __name__ == '__main__':
    # read h5 file
    industry, code, tradedates = read_h5_industry(filePath, h5FileNameInd)
    industry_dict = read_h5_industry_dict(filePath, h5FileNameIndDict)

    # nearest_chg_idx = 0
    # for i in range(tradedates.size-1, 1, -1):
    #     tmp_num = np.sum(industry[i] != industry[i-1])
    #     if tmp_num != 0:
    #         nearest_chg_idx = i
    #         break
    # nearest_chg_date = tradedates[nearest_chg_idx]

    # create sql engine
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigSpider2))
    sql_statement = 'select `%s`, `%s`, `%s` from %s' % (
        sqlFields['date'], sqlFields['code'], sqlFields['industry'], sqlTableName)
    basic_data = pd.read_sql(sql_statement, sql_engine)
    basic_data = basic_data.rename(columns={sqlFields['code']: 'code', sqlFields['date']: 'date' , sqlFields['industry']: 'industry'})
    basic_data = basic_data.drop_duplicates(['date', 'code'])
    basic_data.loc[:, 'industry'] = basic_data['industry'].apply(lambda x: x.strip())
    basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: datetime.strptime(x, u'%Y年%m月%d日 %H:%M:%S') - timedelta(days=1))
    basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: '%s-%s-%s' % (x[0:4], x[5:7], x[8:10]))
    basic_data = basic_data.merge(industry_dict, on='industry')
    basic_data = basic_data.drop_duplicates(['date', 'code'])

    basic_data_pivot = basic_data.pivot_table('indcode', 'date', 'code', fill_value=0)
    basic_data_pivot = basic_data_pivot.astype('int')

    # nearest_chg_idx = 0
    # for i in range(basic_data_pivot.shape[0] - 1, 1, -1):
    #     tmp_num = (basic_data_pivot.iloc[i] != basic_data_pivot.iloc[i - 1]).sum()
    #     if tmp_num != 0:
    #         nearest_chg_idx = i
    #         break
    # nearest_chg_date = basic_data_pivot.index[nearest_chg_idx]

    h5_data = pd.DataFrame(industry, index=tradedates, columns=code)
    common_code = np.array(list(set(code) & set(basic_data_pivot.columns)))
    common_date = np.array(list(set(tradedates) & set(basic_data_pivot.index)))
    common_date = np.sort(common_date)

    basic_data_pivot = basic_data_pivot.loc[common_date, common_code].values
    h5_data = h5_data.loc[common_date, common_code].values
    diff_num = np.sum(h5_data != basic_data_pivot, axis=1)

    pass
