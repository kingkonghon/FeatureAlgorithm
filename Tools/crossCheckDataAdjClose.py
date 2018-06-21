import h5py
from sqlalchemy import create_engine
from Utils.DB_config import ConfigQuant
from Utils.DBOperation import getDataFromSQL
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


sqlTableName = 'yucezhe_market_overviewdata'
sqlFields = {
    'code':'code',
    'date':'date',
    'adj_close':'pre_close'
}
startDate = '2017-10-01'


filePath = ''
h5FileNameAdjFactor = 'LZ_CN_STKA_CMFTR_CUM_FACTOR.h5'
h5FileNameIndClosePrice = 'LZ_CN_STKA_QUOTE_TCLOSE.h5'


def read_h5(file_path, file_name):
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
    adj_factor, code, tradedates = read_h5(filePath, h5FileNameAdjFactor)
    close_price, code, tradedates = read_h5(filePath, h5FileNameIndClosePrice)

    adj_close_price = adj_factor * close_price

    # create sql engine
    sql_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    sql_statement = "select `%s`, `%s`, `%s` from %s where %s >= '%s'" % (
        sqlFields['date'], sqlFields['code'], sqlFields['adj_close'], sqlTableName, sqlFields['date'], startDate)
    basic_data = pd.read_sql(sql_statement, sql_engine)
    basic_data = basic_data.rename(columns={sqlFields['code']: 'code', sqlFields['date']: 'date' , sqlFields['adj_close']: 'adj_close'})
    basic_data = basic_data.drop_duplicates(['date', 'code'])
    # basic_data.loc[:, 'adj_close'] = basic_data['adj_close'].apply(lambda x: x.strip())
    # basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: datetime.strptime(x, u'%Y年%m月%d日 %H:%M:%S') - timedelta(days=1))
    # basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    # basic_data.loc[:, 'date'] = basic_data['date'].apply(lambda x: '%s-%s-%s' % (x[0:4], x[5:7], x[8:10]))

    basic_data_pivot = basic_data.pivot_table('adj_close', 'date', 'code', fill_value=np.nan)

    ori_data = pd.DataFrame(adj_close_price, index=tradedates, columns=code)
    common_code = np.array(list(set(code) & set(basic_data_pivot.columns)))
    common_date = np.array(list(set(tradedates) & set(basic_data_pivot.index)))

    basic_data_pivot = basic_data_pivot.loc[common_date, common_code].values
    tmp_ind = ori_data.loc[common_date, common_code].values
    diff_num = np.sum(tmp_ind != basic_data_pivot, axis=1)

    pass
