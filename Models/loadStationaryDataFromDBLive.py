from multiprocessing import Process, Pipe, Pool
import pandas as pd
import time
from sqlalchemy import create_engine
from SelectedDBConfigStationary import ConfigQuant, stockQuoteConf, allMergeDataConf, areaConf, allMergeDataConf2

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


def mergeDataWithoutDate(tot_data, sql_config, table_name, fields, merge_fields):
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    field_str = list(map(lambda x: '`%s`' % x, fields))
    field_str = ','.join(field_str)

    sql_statement = "select %s from %s" % (field_str, table_name)
    tmp_conn = sql_engine.connect()
    data = pd.read_sql(sql_statement, tmp_conn)
    tmp_conn.close()

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

def screenUnwantedStocks(stock_data, st_config):
    # list date num <= 60 days
    stock_data = stock_data.loc[stock_data['LIST_DAYNUM'] > 60]

    # ST or &ST
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**st_config['db_config']))

    tot_fields = ['code', 'date', st_config['field_name']]
    str_tot_fields = list(map(lambda x: '`%s`'%x, tot_fields))
    str_tot_fields = ','.join(str_tot_fields)

    sql_statement = "select %s from `%s` where `date`=( select max(`date`) from %s)" % (str_tot_fields,
                                                                                    st_config['table_name'], st_config['table_name'])
    sql_conn = sql_engine.connect()
    st_stock_flag = pd.read_sql(sql_statement, sql_conn)  # load st flag data from DB
    sql_conn.close()
    st_stock_flag = st_stock_flag.drop('date', axis=1)  # do not merge on date any more
    stock_data = stock_data.merge(st_stock_flag, how='left', on='code')  # merge on code

    stock_data = stock_data.loc[stock_data[st_config['field_name']] == 0]
    # drop st flag column (useless because drop all 0)
    stock_data = stock_data.drop(st_config['field_name'], axis=1)

    return stock_data

class basicDataClass:
    def __init__(self, fields, table_name, sql_config):
        # Process.__init__(self)
        # self.output_pipe = output_pipe
        # self.final_output_pipe = final_output_pipe
        # self.merge_num = merge_num
        self.fields = fields
        self.table_name = table_name
        self.sql_config = sql_config
        self.data = []
        self.trade_date = ''

    def run(self):
        sql_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**self.sql_config))

        # fetch data from db
        str_fields = list(map(lambda x: '`%s`' % x, self.fields))
        str_fields = ','.join(str_fields)
        sql_statement = "select %s from %s where `date`=(select max(`date`) from %s)" % (
            str_fields, self.table_name, self.table_name)  # get the newest data
        sql_conn = sql_engine.connect()
        data = pd.read_sql(sql_statement, sql_conn)
        sql_conn.close()

        data = data.drop_duplicates(['date', 'code'])

        self.trade_date = data['date'].unique()
        if self.trade_date.size != 1:
            print('load basic data error: more than one day')
        else:
            self.trade_date = self.trade_date[0]

        # close copy of output pipe
        # self.output_pipe.close()

        # special data that need to be merge
        data = mergeDataWithoutDate(tot_data=data, sql_config=ConfigQuant, **areaConf)
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
            tmp_data = merge_dict['data']
            tmp_data = tmp_data.drop_duplicates(merge_dict['merge_cols'])   # data of the latest records
            tmp_data.loc[:, 'date'] = self.trade_date
            self.data = self.data.merge(tmp_data, how='left', on=merge_dict['merge_cols'])
            # except EOFError:
            #     break
            # self.final_output_pipe.send(data)
            # self.final_output_pipe.close()
            # print('finish')


class mergeDataClass:
    def __init__(self, fields, merge_fields, drop_fields, condition, prefix, table_name, sql_config):
        # Process.__init__(self)
        # self.output_pipe = output_pipe
        self.fields = fields
        self.merge_fields = merge_fields
        self.drop_fields = drop_fields
        self.condition = condition
        self.prefix = prefix
        self.table_name = table_name
        self.sql_config = sql_config

    def run(self):
        sql_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**self.sql_config))

        sql_conn = sql_engine.connect()

        if type(self.fields).__name__ == 'list':
            str_fields = list(map(lambda x: '`%s`' % x, self.fields))
            str_fields = ','.join(str_fields)
        else:
            str_fields = self.fields  # select *

        sql_statement = "select %s from %s where `date`=(select max(`date`) from %s)" % (
            str_fields, self.table_name, self.table_name)
        if self.condition != '':
            sql_statement = "%s and (%s)" % (sql_statement, self.condition)
        data = pd.read_sql(sql_statement, sql_conn)
        sql_conn.close()

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
        return merge_dict


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

def loadData():
    # ============= the first wave of data merge ================
    pool = Pool(processes=4)
    process_merge_data = []

    # stock quotes
    process_basic_data = basicDataClass(sql_config=ConfigQuant, **stockQuoteConf)
    # process_basic_data.start()
    # tmp_process = pool.apply_async(process_basic_data.run)
    # process_merge_data.append(tmp_process)
    process_basic_data.run()

    # data to be merged
    # print(len(allMergeDataConf),"to be merged")
    for conf in allMergeDataConf:
        # tmp_process = mergeDataClass(output_pipe=output_pipe, sql_config=ConfigQuant, start_date=start_date, end_date=end_date, **conf)
        # tmp_process.start()
        tmp_ins = mergeDataClass(sql_config=ConfigQuant, **conf)
        tmp_process = pool.apply_async(tmp_ins.run)
        process_merge_data.append(tmp_process)

    for tmp_process in process_merge_data:
        merge_dict = tmp_process.get()
        process_basic_data.mergeData(merge_dict)

    # output_pipe.close()
    # final_output_pipe.close()
    #
    # try:
    #     final_data = final_input_pipe.recv()
    # except EOFError:
    #     pass
    # print(process_basic_data.data)
    # print('end of the first wave')
    pool.terminate()
    # ========== the second wave of merge ===========
    input_pipe2, output_pipe2 = Pipe(True)
    final_input_pipe2, final_output_pipe2 = Pipe(True)

    pool = Pool(processes=4)
    # receiver process
    # receiver_process = Process(target=receiveMergeData, args=(final_data, input_pipe, final_output_pipe))
    # receiver_process.start()
    # receiver_process = pool.apply_async(receiveMergeData, (final_data, input_pipe2, len(allMergeDataConf2), final_output_pipe2))

    # data to be merge (send loaded data to receiver process to be merged)
    # print(len(allMergeDataConf2), 'to be merged')
    process_merge_data = []
    for conf in allMergeDataConf2:
        tmp_ins = mergeDataClass(sql_config=ConfigQuant, **conf)
        tmp_process = pool.apply_async(tmp_ins.run)
        # tmp_process.start()
        process_merge_data.append(tmp_process)

    for tmp_process in process_merge_data:
        merge_dict = tmp_process.get()
        process_basic_data.mergeData(merge_dict)

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
    pool.terminate()

    # ====== screen data ======
    process_basic_data.data = screenUnwantedStocks(process_basic_data.data, STConfig)
    process_basic_data.data = process_basic_data.data.reset_index()
    process_basic_data.data = process_basic_data.data.drop('index', axis=1)

    # ====== convert industry to dummy variables ======
    chunk_size = 50000
    chunk_num = int(process_basic_data.data.shape[0] / chunk_size)
    if chunk_num * chunk_size < process_basic_data.data.shape[0]:
        chunk_num += 1

    pool = Pool(processes=4)
    tot_child_process = []
    for i in range(chunk_num):
        tmp_data = process_basic_data.data.iloc[i * chunk_size: (i + 1) * chunk_size]
        tmp_process = pool.apply_async(pivotStockCategory, (tmp_data, SW_industry, Market, Area))
        tot_child_process.append(tmp_process)

    tot_data = pd.DataFrame([])
    for tmp_process in tot_child_process:
        tmp_data = tmp_process.get()
        tot_data = tot_data.append(tmp_data)

    # return process_basic_data.data
    pool.terminate()

    print('data shape:',tot_data.shape)
    return tot_data


if __name__ == '__main__':
    # start_date = '2013-11-01'
    # end_date = '2013-11-04'

    final_data = loadData()

    print(final_data)