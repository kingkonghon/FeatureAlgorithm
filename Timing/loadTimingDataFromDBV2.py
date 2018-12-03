from multiprocessing import Process, Pipe, Pool
import pandas as pd
import time
from sqlalchemy import create_engine
from TimingDBConfigV2 import ConfigQuant, HS300QuoteConf, allMergeDataConf

# ========================  version features  ===============================
#  convert industry, area, market into dummy variables
#

# =================== SW industry code ===========================


def mergeDataWithoutDate(tot_data, sql_config, table_name, fields, merge_fields):
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    field_str = list(map(lambda x: '`%s`' % x, fields))
    field_str = ','.join(field_str)

    sql_statement = "select %s from %s" % (field_str, table_name)
    data = pd.read_sql(sql_statement, sql_engine)

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

def loadHS300Date(sql_config, hs300_table_config, start_date, end_date):
    sql_engine = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**sql_config))

    sql_statement = "select %s from %s where %s between '%s' and '%s'" % (hs300_table_config['date_field'],
                        hs300_table_config['table_name'], hs300_table_config['date_field'], start_date, end_date)
    hs300_date = pd.read_sql(sql_statement, sql_engine)

    return hs300_date

def oneHotEncoding(data, cols):
    for tmp_col in cols:
        tmp_uni_vals = data[tmp_col].astype('float').unique()
        for tmp_val in tmp_uni_vals:
            tmp_new_col = '%s:%.0f' % (tmp_col, tmp_val)
            data.loc[:, tmp_new_col] = data[tmp_col].apply(lambda x: x == tmp_val)

    return data


class mergeDataClass:
    def __init__(self, fields, merge_fields, one_hot_fields, drop_fields, discrete_fields, rounding_fields, condition, prefix, start_date, end_date,
                 table_name, sql_config):
        # Process.__init__(self)
        # self.output_pipe = output_pipe
        self.fields = fields
        self.merge_fields = merge_fields
        self.drop_fields = drop_fields
        self.rounding_fields = rounding_fields
        self.one_hot_fields = one_hot_fields
        self.discrete_fields = discrete_fields
        self.condition = condition
        self.prefix = prefix
        self.table_name = table_name
        self.sql_config = sql_config
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        sql_engine = create_engine(
            'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**self.sql_config))

        if type(self.fields).__name__ == 'list':
            str_fields = list(map(lambda x: '`%s`' % x, self.fields))
            str_fields = ','.join(str_fields)
        else:
            str_fields = self.fields  # select *

        sql_statement = "select %s from %s where (`date` between '%s' and '%s')" % (
            str_fields, self.table_name, self.start_date, self.end_date)
        if self.condition != '':
            sql_statement = "%s and (%s)" % (sql_statement, self.condition)
        data = pd.read_sql(sql_statement, sql_engine)

        # drop columns
        if self.drop_fields != []:
            data = data.drop(self.drop_fields, axis=1)

        # rounding real-value features
        if self.rounding_fields != {}:
            for tmp_field, tmp_rounding_num in self.rounding_fields.items():
                data.loc[:, tmp_field] = data[tmp_field].apply(lambda x: round(round(x, tmp_rounding_num) * pow(10, tmp_rounding_num), 0))

        # convert features into one-hot encoding
        if self.one_hot_fields != []:
            data = oneHotEncoding(data, self.one_hot_fields)

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

        # if not data.empty:
        # merge_dict = {'data': data, 'merge_cols': self.merge_fields, 'table_name': self.table_name}
        # self.output_pipe.send(merge_dict)
        # else:
        # print('empty!!', self.table_name)
        # self.output_pipe.close()
        # print('sent', self.table_name)
        return data


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
    # ============= the first wave of data merge ================
    hs300_data = loadHS300Date(ConfigQuant, HS300QuoteConf,  start_date, end_date)

    # data to be merged
    pool = Pool(processes=3)
    process_merge_data = []
    for conf in allMergeDataConf:
        # tmp_process = mergeDataClass(output_pipe=output_pipe, sql_config=ConfigQuant, start_date=start_date, end_date=end_date, **conf)
        # tmp_process.start()
        tmp_ins = mergeDataClass(sql_config=ConfigQuant, start_date=start_date,
                                 end_date=end_date, **conf)
        tmp_process = pool.apply_async(tmp_ins.run)
        process_merge_data.append(tmp_process)

    for tmp_process in process_merge_data:
        merge_data = tmp_process.get()
        hs300_data = hs300_data.merge(merge_data, on=HS300QuoteConf['date_field'], how='left')

    # return process_basic_data.data
    return hs300_data


if __name__ == '__main__':
    start_date = '2013-04-11'
    end_date = '2013-04-20'

    final_data = loadData(start_date, end_date)

    print(final_data)