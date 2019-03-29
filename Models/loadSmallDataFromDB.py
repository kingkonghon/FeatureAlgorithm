from multiprocessing import Pool
import pandas as pd
import time
from sqlalchemy import create_engine
from sqlalchemy.pool import SingletonThreadPool
from SelectedDBConfigStationary import ConfigQuant, stockQuoteConf, allMergeDataConf, areaConf, allMergeDataConf2
import time
import os

# ========================  version features  ===============================
#  srceen st, listed less than 60 days stocks
#  convert industry, area, market into dummy variables
#
# =================== SW industry code ===========================
SW_industry = [
    '农林牧渔',
    '采掘',
    '化工',
    '钢铁',
    '有色金属',
    '电子',
    '家用电器',
    '食品饮料',
    '纺织服装',
    '轻工制造',
    '医药生物',
    '公用事业',
    '交通运输',
    '房地产',
    '金融服务',
    '商业贸易',
    '休闲服务',
    '综合',
    '建筑材料',
    '建筑装饰',
    '电气设备',
    '国防军工',
    '计算机',
    '传媒',
    '通信',
    '银行',
    '非银金融',
    '汽车',
    '机械设备'
]

Market = [
    'SH',
    'SZ',
    'SMEB',
    'GEB'
]

Area = [
    '湖北',
    '江苏',
    '河南',
    '甘肃',
    '山西',
    '天津',
    '广东',
    '浙江',
    '福建',
    '北京',
    '安徽',
    '山东',
    '青海',
    '河北',
    '深圳',
    '上海',
    '四川',
    '湖南',
    '辽宁',
    '广西',
    '陕西',
    '吉林',
    '宁夏',
    '新疆',
    '云南',
    '重庆',
    '黑龙江',
    '内蒙',
    '海南',
    '贵州',
    '江西',
    '西藏'
]

STConfig = {
    'db_config': ConfigQuant,
    'table_name': 'STOCK_FLAG',
    'field_name': 'ST_FLAG'
}


def mergeDataWithoutDate(tot_data, sql_conn, table_name, fields, merge_fields):
    t_start = time.time()

    field_str = list(map(lambda x: '`%s`' % x, fields))
    field_str = ','.join(field_str)

    sql_statement = "select %s from %s" % (field_str, table_name)
    data = pd.read_sql(sql_statement, sql_conn)

    t_eclipsed = time.time() - t_start
    print('%s sql time consumed %f' % (table_name, t_eclipsed))

    tot_data = tot_data.merge(data, how='left', on=merge_fields)
    return tot_data


def getStockMarket(tot_data):
    tot_data.loc[:, 'market'] = tot_data['code'].apply(lambda x: 'SH' if x[:2] == '60' else (
        'SMEB' if x[:3] == '002' else (
            'GEB' if x[:3] == '300' else 'SZ')))
    return tot_data


def pivotStockCategory(tot_data, sw_industry, market_name, area_name):
    # industry
    tot_categories = ['industry', 'area', 'market']
    category_subset = [sw_industry, area_name, market_name]

    for c_name, c_subset in zip(tot_categories, category_subset):
        category = tot_data[c_name]
        for tmp_c in c_subset:
            tot_data.loc[:, tmp_c] = category.apply(lambda x: 1 if x == tmp_c else 0)

        tot_data = tot_data.drop(c_name, axis=1)

    return tot_data

def screenUnwantedStocks(stock_data, st_config, start_date, end_date, sql_conn):
    # list date num <= 60 days
    stock_data = stock_data.loc[stock_data['LIST_DAYNUM'] > 60]

    # ST or &ST
    tot_fields = ['code', 'date', st_config['field_name']]
    str_tot_fields = list(map(lambda x: '`%s`'%x, tot_fields))
    str_tot_fields = ','.join(str_tot_fields)

    t_start = time.time()

    sql_statement = "select %s from `%s` where (`date` between '%s' and '%s')" % (str_tot_fields,
                                                                                    st_config['table_name'], start_date, end_date)
    st_stock_flag = pd.read_sql(sql_statement, sql_conn)  # load st flag data from DB

    t_eclipsed = time.time() - t_start
    print('%s sql time consumed %f' % (st_config['table_name'], t_eclipsed))

    stock_data = stock_data.merge(st_stock_flag, how='left', on=['date', 'code'])

    stock_data = stock_data.loc[stock_data[st_config['field_name']] == 0]
    # drop st flag column (useless because drop all 0)
    stock_data = stock_data.drop(st_config['field_name'], axis=1)

    return stock_data

class basicDataClass:
    def __init__(self, fields, start_date, end_date, sql_conn, table_name):
        # Process.__init__(self)
        # self.output_pipe = output_pipe
        # self.final_output_pipe = final_output_pipe
        # self.merge_num = merge_num
        self.sql_conn = sql_conn
        self.fields = fields
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        self.data = []

    def load_data(self):
        t_start = time.time()

        # fetch data from db
        str_fields = list(map(lambda x: '`%s`' % x, self.fields))
        str_fields = ','.join(str_fields)
        sql_statement = "select %s from %s where `date` between '%s' and '%s'" % (
            str_fields, self.table_name, self.start_date, self.end_date)
        data = pd.read_sql(sql_statement, self.sql_conn)
        data = data.drop_duplicates(['date', 'code'])

        t_eclipsed = time.time() - t_start
        print('%s, sql time consumed %f' % (self.table_name, t_eclipsed))

        # close copy of output pipe
        # self.output_pipe.close()

        # special data that need to be merge
        data = mergeDataWithoutDate(tot_data=data, sql_conn=self.sql_conn, **areaConf)
        data = getStockMarket(data)

        self.data = data

    def mergeData(self, merge_dict):
        # merge other data receive from Pipe
        if not self.data.empty:
            # print('ready to receive data...')
            # for i in range(self.merge_num):
            #     try:
            #         print(i)
            # merge_dict = self.input_pipe.recv()
            # print(merge_dict['table_name'])
            # if len(merge_dict['data']) == 0:
            #     print('empty:', merge_dict['table_name'])
            # else:
            t_start = time.time()

            tmp_data = merge_dict['data']
            tmp_data = tmp_data.drop_duplicates(merge_dict['merge_cols'])
            self.data = self.data.merge(tmp_data, how='left', on=merge_dict['merge_cols'])

            t_eclipsed = time.time() - t_start
            print('%s, merging time consumed %f' % (merge_dict['table_name'], t_eclipsed))
            # except EOFError:
            #     break
            # self.final_output_pipe.send(data)
            # self.final_output_pipe.close()
            # print('finish')


class mergeDataClass:
    def __init__(self, start_date, end_date, fields, merge_fields, drop_fields, condition, prefix,
                 table_name):
        self.fields = fields
        self.merge_fields = merge_fields
        self.drop_fields = drop_fields
        self.condition = condition
        self.prefix = prefix
        self.table_name = table_name
        self.start_date = start_date
        self.end_date = end_date
        # self.drop_nan_ratio = 0.7

    def load_data(self):
        global sql_adapter

        t_start = time.time()

        sql_conn = sql_adapter.get_connection(os.getpid())

        if type(self.fields).__name__ == 'list':
            str_fields = list(map(lambda x: '`%s`' % x, self.fields))
            str_fields = ','.join(str_fields)
        else:
            str_fields = self.fields  # select *

        sql_statement = "select %s from %s where (`date` between '%s' and '%s')" % (
            str_fields, self.table_name, self.start_date, self.end_date)
        if self.condition != '':
            sql_statement = "%s and (%s)" % (sql_statement, self.condition)
        data = pd.read_sql(sql_statement, sql_conn)
        # self.sql_conn.close()

        t_eclipsed = time.time() - t_start
        print('%s, sql time consumed %f' % (self.table_name, t_eclipsed))
        t_start = time.time()

        # # drop feature if mostly nan
        # tmp_nan_ratio = data.isnull().sum(axis=0) / data.shape[0]
        # tmp_drop_cols = tmp_nan_ratio[tmp_nan_ratio > self.drop_nan_ratio].index.tolist()
        # if len(tmp_drop_cols) > 0:
        #     print('drop nan columns:', tmp_drop_cols)
        #     data = data.drop(tmp_drop_cols, axis=1)

        # drop columns
        if self.drop_fields != []:
            data = data.drop(self.drop_fields, axis=1)

        tmp_dtypes = data.dtypes
        tmp_dtypes = tmp_dtypes[tmp_dtypes == 'O']
        if (tmp_dtypes.size >= 3) and (self.table_name != 'STOCK_INDUSTRY'):
            print('after drop:', self.table_name)
            print(sql_statement)
            print(tmp_dtypes)

        # rename columns
        if self.prefix != '':
            tmp_col_names = list(data.columns)
            for tmp_col in self.merge_fields:
                tmp_col_names.remove(tmp_col)
            # tmp_col_names.remove('date')
            # tmp_col_names.remove('code')
            tmp_new_col_names = list(map(lambda x: '%s_%s' % (self.prefix, x), tmp_col_names))
            rename_dict = {}
            for field_pair in zip(tmp_col_names, tmp_new_col_names):
                rename_dict[field_pair[0]] = field_pair[1]
            data = data.rename(columns=rename_dict)

        tmp_dtypes = data.dtypes
        tmp_dtypes = tmp_dtypes[tmp_dtypes == 'O']
        if (tmp_dtypes.size >= 3) and (self.table_name != 'STOCK_INDUSTRY'):
            print('after rename:')
            print(tmp_dtypes)

        # if not data.empty:
        merge_dict = {'data': data, 'merge_cols': self.merge_fields, 'table_name': self.table_name}
        # self.output_pipe.send(merge_dict)
        # else:
        # print('empty!!', self.table_name)
        # self.output_pipe.close()
        # print('sent', self.table_name)

        t_eclipsed = time.time() - t_start
        print('%s, processing time consumed %f' % (self.table_name, t_eclipsed))

        return merge_dict


class sqlConnClass:
    def __init__(self, sql_config):
        self.engines = {}
        self.sql_config = sql_config

    def get_connection(self, pid):
        if pid in self.engines.keys():
            return self.engines[pid]  # engine belong to this worker already exists
        else:
            self.engines[pid] = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**self.sql_config), poolclass=SingletonThreadPool)
            print('create engine for pid:', pid)
            return self.engines[pid]

    def dispose(self):
        for pid in self.engines.keys():
            self.engines[pid].dispose()
        self.engines = {}

# def receiveMergeData(tot_data, input_pipe, merge_num, final_output_pipe):
#     print('ready to receive the second wave of data...')
#     for i in range(merge_num):
#         try:
#             print(i)
#             merge_dict = input_pipe.recv()
#             print(merge_dict['table_name'])
#             if len(merge_dict['data']) == 0:
#                 print('empty:', merge_dict['table_name'])
#             else:
#                 tot_data = tot_data.merge(merge_dict['data'], how='left', on=merge_dict['merge_cols'])
#         except EOFError:
#             print('end of second wave')
#             break
#     final_output_pipe.send(tot_data)
#     final_output_pipe.close()
#     print('finish')
# =================== load data function =================

def loadData(start_date, end_date):
    global sql_adapter
    sql_adapter = sqlConnClass(ConfigQuant)
    engine_main = sql_adapter.get_connection(os.getpid())

    # sql_engine = create_engine(
    #     'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    # sql_conn = sql_engine.connect()

    # stock quotes
    process_basic_data = basicDataClass(start_date=start_date, end_date=end_date, sql_conn = engine_main,
                                        **stockQuoteConf)
    process_basic_data.load_data()

    # ============= the first wave of data merge ================
    pool = Pool(processes=3)
    process_merge_data = []
    # data to be merged
    # print(len(allMergeDataConf),"to be merged")
    for conf in allMergeDataConf:
        # tmp_process = mergeDataClass(output_pipe=output_pipe, sql_config=ConfigQuant, start_date=start_date, end_date=end_date, **conf)
        # tmp_process.start()
        tmp_ins = mergeDataClass(start_date=start_date, end_date=end_date, **conf)
        tmp_process = pool.apply_async(tmp_ins.load_data)
        process_merge_data.append(tmp_process)

    for tmp_process in process_merge_data:
        t_start = time.time()

        merge_dict = tmp_process.get()
        process_basic_data.mergeData(merge_dict)

        t_eclipsed = time.time() - t_start
        print('%d process consumed', t_eclipsed)

    # output_pipe.close()
    # final_output_pipe.close()
    #
    # try:
    #     final_data = final_input_pipe.recv()
    # except EOFError:
    #     pass
    # print(process_basic_data.data)
    # print('end of the first wave')
    # pool.terminate()
    # ========== the second wave of merge ===========
    process_merge_data = []
    for conf in allMergeDataConf2:
        tmp_ins = mergeDataClass(start_date=start_date, end_date=end_date, **conf)
        tmp_process = pool.apply_async(tmp_ins.load_data)
        # tmp_process.start()
        process_merge_data.append(tmp_process)

    for tmp_process in process_merge_data:
        t_start = time.time()

        merge_dict = tmp_process.get()
        process_basic_data.mergeData(merge_dict)

        t_eclipsed = time.time() - t_start
        print('%d process consumed', t_eclipsed)

    tmp_d_types = process_basic_data.data.dtypes
    print('data object columns:')
    print(tmp_d_types[tmp_d_types == 'O'])
    # print('ready for final')
    #
    # # output_pipe2.close()
    # # final_output_pipe2.close()
    #
    # try:
    #     final_data = final_input_pipe2.recv()
    # except EOFError:
    #     pass
    # print('final data shape:', process_basic_data.data.shape)
    # pass
    # pool.terminate()

    # ====== screen data ======
    t_start = time.time()

    process_basic_data.data = screenUnwantedStocks(process_basic_data.data, STConfig, start_date, end_date, engine_main)
    process_basic_data.data = process_basic_data.data.reset_index()
    process_basic_data.data = process_basic_data.data.drop('index', axis=1)

    t_eclipsed = time.time() - t_start
    print('screen data time consumed', t_eclipsed)

    # ====== convert industry to dummy variables ======
    t_start = time.time()

    chunk_size = 50000
    chunk_num = int(process_basic_data.data.shape[0] / chunk_size)
    if chunk_num * chunk_size < process_basic_data.data.shape[0]:
        chunk_num += 1

    # pool = Pool(processes=3)
    tot_child_process = []
    for i in range(chunk_num):
        tmp_data = process_basic_data.data.iloc[i * chunk_size: (i + 1) * chunk_size]
        tmp_process = pool.apply_async(pivotStockCategory, (tmp_data, SW_industry, Market, Area))
        tot_child_process.append(tmp_process)

    tot_data = pd.DataFrame([])
    for tmp_process in tot_child_process:
        tmp_data = tmp_process.get()
        tot_data = tot_data.append(tmp_data)

    t_eclipsed = time.time() - t_start
    print('one-hot time consumed', t_eclipsed)

    # return process_basic_data.data
    pool.terminate()
    sql_adapter.dispose()

    print('data shape:',tot_data.shape)
    print('data time range: %s - %s' % (tot_data['date'].min(), tot_data['date'].max()))
    return tot_data


if __name__ == '__main__':
    tic = time.time()
    start_date = '2018-12-11'
    end_date = '2018-12-11'

    sql_adapter = None
    final_data = loadData(start_date, end_date)

    toc =time.time()
    print('time eclipsed:', toc-tic)

    print(final_data)