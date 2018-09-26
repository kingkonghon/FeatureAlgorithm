import tushare as ts
import pandas as pd
import os
import sys
from time import sleep
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np
import h5py
from urllib.error import HTTPError
import pymysql

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant

targetTableName = 'STOCK_FUNDAMENTAL_TUSHARE'

factorRenameDict={
    'distrib': 'split_and_dividend',
    'epcf': 'cfo_per_share',
    'net_profit_ratio': 'parent_net_profit_margin',
    'gross_profit_rate': 'gross_profit_margin',
    'business_income': 'revenue',
    'bips': 'sales_per_share',
    'arturnover': 'acc_rece_turover',
    'arturndays': 'acc_rece_turover_days',
    'inventory_days': 'inventory_turnover_days',
    'currentasset_days': 'cur_asst_turnover_days',
    'mbrg': 'revenue_yoy',
    'nav': 'consolidated_equity_yoy',
    'targ': 'asset_yoy',
    'seg': 'parent_equity_yoy',
    'icratio': 'interest_coverage',
    'sheqratio': 'equity_ratio',  # equity_ratio = equity / asset = 1 - asset_ratio
    'adratio': 'asset_ratio',
    'cf_sales': 'cfo_to_revenue_ratio',
    'rateofreturn': 'cfo_to_asset_ratio',
    'cf_liabilities': 'cfo_liab_ratio',
    'net_profits': 'parent_net_profits',
    'profits_yoy': 'parent_net_profits_yoy',  #  checked
    'currentratio': 'current_ratio',   # checked
    'quickratio': 'quick_ratio',  # checked
    'cashratio': 'cash_ratio'  ## ???? not matched with Wind
}

h5FilePath = 'D:\FeatureAlgorithm\Data\LZ_CN_STKA_BAL_COMBO_ACTUAL_ANN_DT_RT.h5'

def loadWindReportData():
    # open file
    h5_file = h5py.File(h5FilePath, 'r')

    # load data from h5 file
    wind_data = h5_file['data'][...]
    wind_tradedates = h5_file['date'][...]
    wind_codes = h5_file['header'][...]

    # format data
    wind_tradedates = list(map(lambda x: str(x), wind_tradedates))
    wind_tradedates = np.array(list(map(lambda x: '%s-%s-%s' % (x[:4], x[4:6], x[6:]), wind_tradedates)))
    wind_codes = list(map(lambda x: x.decode('utf-8'), wind_codes))
    wind_codes = np.array(list(map(lambda x: x.split('.')[1], wind_codes)))

    return wind_tradedates, wind_codes, wind_data


def getWindReportDate(wind_tradedates, wind_codes, wind_data, ori_codes, current_year, current_season):
    # fiscal report date
    fiscal_report_dates = {
        1: '03-31',
        2: '06-30',
        3: '09-30',
        4: '12-31'
    }

    # find the first trade date after the fiscal trade date
    target_report_date = '%d-%s' % (current_year, fiscal_report_dates[current_season])
    pos_v = np.where(wind_tradedates >= target_report_date)[0][0]

    # find report release dates
    release_dates = []
    for i, tmp_code in enumerate(ori_codes):
        try:
            tmp_pos_h = np.where(wind_codes == tmp_code)[0][0]
            tmp_date = wind_data[pos_v, tmp_pos_h]
            tmp_date = tmp_date.decode('utf-8')
            if tmp_date == 'None' or tmp_date == 'nan':
                tmp_date = np.nan
            else:
                tmp_date = '%s-%s-%s' % (tmp_date[:4], tmp_date[4:6], tmp_date[6:])
        except IndexError: # cannot be found in data
            tmp_date = np.nan
        release_dates.append(tmp_date)

    return release_dates


def getDataFromTushare(func, year, season):
    failed_retry_secs = 30
    success_retry_secs = 10

    stable_time_num = 0
    total_stable_time_num = 5 # set multiple download attempts because it can be unstable
    data = pd.DataFrame([])
    while True:
        try:
            tmp_data = func(year, season)
            print('\n')
            tmp_data = tmp_data.drop_duplicates('code')

            # check if there are new records or new non-nan values at this attempt
            old_record_num = data.shape[0]
            is_filled_nan = False  # set a flag to record if there are new non-nan values
            if old_record_num == 0:
                data = tmp_data.copy()
            else:
                # check if nan values in previous download now are non-nan
                tmp_merge_data = data.merge(tmp_data, on='code', how='left', suffixes=['_l', '_r'])
                tmp_merge_data.index = data.index

                tmp_check_cols = tmp_data.columns.tolist()
                tmp_check_cols = [x for x in tmp_check_cols if x not in
                                  ['code', 'fiscal_year', 'fiscal_season', 'time_stamp',
                                   'report_date']]  # remove column that doesn't need to check

                for tmp_col in tmp_check_cols:  # check each column to see if there are nans that can be filled
                    l_tmp_col = tmp_col + '_l'
                    r_tmp_col = tmp_col + '_r'

                    tmp_idx = tmp_merge_data[l_tmp_col].isnull() & (~tmp_merge_data[r_tmp_col].isnull())
                    if tmp_idx.sum() > 0:  # there exists newly downloaded, non-nan data, to fill existing data
                        is_filled_nan = True  # need to rewrite database
                        data.loc[tmp_idx, tmp_col] = tmp_merge_data.loc[tmp_idx, r_tmp_col]  # fill nan

                # append new data, and drop duplicates
                data = data.append(tmp_data)
                data = data.drop_duplicates('code')

            new_record_num = data.shape[0]
            if (new_record_num == old_record_num) and ~is_filled_nan:  # if no new record is downloaded and no nan values are updated
                stable_time_num += 1
            if stable_time_num >= total_stable_time_num: # if no new record and no new non-nan values after some times, complete
                break

            sleep(success_retry_secs)
        except HTTPError:
            print('download data failed, retry after %d secs' % failed_retry_secs)
            sleep(failed_retry_secs)

    return data


def mergeData(left_data, right_data):
    right_data = right_data.drop('name', axis=1) # drop the name column (duplicated)
    right_data = right_data.drop_duplicates('code') # drop duplicates

    common_col = [x for x in left_data.columns if x in right_data.columns]
    common_col.remove('code')

    merge_data = left_data.merge(right_data, on='code', how='left', suffixes=['_x', '_y'])

    # compare common data: 1. use left data to fill nan in right data, 2. delete left data column, 3. rename right column name
    rename_col_dict = {}
    col_to_del = []
    for tmp_col in common_col:
        l_tmp_col = tmp_col + '_x'
        r_tmp_col = tmp_col + '_y'

        tmp_com_data = merge_data[[l_tmp_col, r_tmp_col]]
        tmp_filled_idx = tmp_com_data[r_tmp_col].isnull() & (~tmp_com_data[l_tmp_col].isnull())

        # find discrepancy
        tmp_discrepancy = (tmp_com_data[l_tmp_col] - tmp_com_data[r_tmp_col]).abs()
        tmp_discrepancy = tmp_discrepancy[tmp_discrepancy > 0.01]
        if not tmp_discrepancy.empty:
            print('found discrepancy: %s (size: %d)'% (tmp_col, tmp_discrepancy.shape[0]))

            # columns that left data are more reliable, use left to rewrite right
            if tmp_col in ['eps', 'profits_yoy']:
                tmp_com_data.loc[tmp_discrepancy.index, r_tmp_col] = tmp_com_data.loc[tmp_discrepancy.index, l_tmp_col]

        # fill right nan with left non-nan value
        if tmp_filled_idx.sum() > 0: # check if there is data in right nan but non-nan in left
            tmp_com_data.loc[tmp_filled_idx, r_tmp_col] = tmp_com_data.loc[tmp_filled_idx, l_tmp_col]
            merge_data.loc[:, r_tmp_col] = tmp_com_data[r_tmp_col] # copy the changed value back to the original data

        # record left columns to delete, and right colums to rename
        col_to_del.append(l_tmp_col)
        rename_col_dict[r_tmp_col] = tmp_col

    # delete left columns
    merge_data = merge_data.drop(col_to_del, axis=1)

    # rename right columns
    merge_data = merge_data.rename(rename_col_dict, axis=1)

    return merge_data

def downloadReportFull(start_year, start_season, end_year, end_season, write_method):
    # connect to database
    quant_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

    current_year = start_year
    current_season = start_season

    # load wind financial report release dates
    wind_tradedates, wind_codes, wind_release_dates = loadWindReportData()

    dump_record_num = 50000 # write data to sql every N records
    tot_data = pd.DataFrame([])
    # write_method = 'replace'
    while True:
        print('%d_s%d'%(current_year, current_season))
        # main report
        tmp_data = getDataFromTushare(ts.get_report_data, current_year, current_season)
        tmp_data = tmp_data.drop('name', axis=1) # drop stock name
        tmp_data = tmp_data.rename({'roe': 'roe_weighted'}, axis=1) # roe is weighted roe
        chunk_data = tmp_data.copy()

        # use wind report release dates to replace tushare data
        chunk_data.loc[:, 'report_date'] = getWindReportDate(wind_tradedates, wind_codes, wind_release_dates,
                                                             chunk_data['code'], current_year, current_season)

        # profit
        tmp_data = getDataFromTushare(ts.get_profit_data, current_year, current_season)
        tmp_data.loc[:, 'net_profits'] = tmp_data * 100  # profit unit not consistent (million to wan yuan)
        tmp_data.loc[tmp_data['net_profits'] == 0, 'net_profits'] = np.nan # profit impossible to be 0
        tmp_data.loc[:, 'business_income'] = tmp_data * 100  # revenue unit not consistent (million to wan yuan)
        tmp_data = tmp_data.rename({'roe': 'roe_diluted'}, axis=1)
        chunk_data = mergeData(chunk_data, tmp_data)

        # operation
        tmp_data = getDataFromTushare(ts.get_operation_data, current_year, current_season)
        chunk_data = mergeData(chunk_data, tmp_data)

        # growth
        tmp_data = getDataFromTushare(ts.get_growth_data, current_year, current_season)
        tmp_data = tmp_data.rename({'nprg': 'profits_yoy', 'epsg': 'eps_yoy'}, axis=1)
        chunk_data = mergeData(chunk_data, tmp_data)

        # solvency
        tmp_data = getDataFromTushare(ts.get_debtpaying_data, current_year, current_season)
        tmp_data.loc[:, 'currentratio'] = (tmp_data['currentratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        tmp_data.loc[:, 'quickratio'] = (tmp_data['quickratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        tmp_data.loc[:, 'cashratio'] = (tmp_data['cashratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        tmp_data.loc[:, 'icratio'] = (tmp_data['icratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        tmp_data.loc[:, 'sheqratio'] = (tmp_data['sheqratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        tmp_data.loc[:, 'adratio'] = (tmp_data['adratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
        chunk_data = mergeData(chunk_data, tmp_data)

        # cash flow
        tmp_data = getDataFromTushare(ts.get_cashflow_data, current_year, current_season)
        tmp_data = tmp_data.drop(['cf_nm', 'cashflowratio'], axis=1) # drop columns unclear
        chunk_data = mergeData(chunk_data, tmp_data)

        # change column names
        chunk_data = chunk_data.rename(factorRenameDict, axis=1)

        # add year & season into the data buffer
        chunk_data.loc[:, 'fiscal_year'] = current_year
        chunk_data.loc[:, 'fiscal_season'] = current_season

        # store this season's data into the total data buffer
        tot_data = tot_data.append(chunk_data)

        # dump data to database if record num exceeds the threshold
        if tot_data.shape[0] > dump_record_num:
            tot_data.loc[:, 'time_stamp'] = datetime.now()
            quant_engine = create_engine(
                'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant)) # connect again, in case disconnected
            tot_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method)
            write_method = 'append'
            tot_data = pd.DataFrame([])

        # move on to the next season
        current_season += 1
        if current_year == end_year and current_season > end_season:  # finish downloading
            tot_data.loc[:, 'time_stamp'] = datetime.now()
            quant_engine = create_engine(
                'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))  # connect again, in case disconnected
            tot_data.to_sql(targetTableName, quant_engine, index=False, if_exists=write_method) # write the last chunk of data into sql
            write_method = 'append'
            break

        if current_season > 4: # move on to next year
            current_season = 1
            current_year += 1


def downloadReportIncrm():
    # fiscal report date
    fiscal_report_dates = {
        1: '03-31',
        2: '06-30',
        3: '09-30',
        4: '12-31'
    }

    today_dt = datetime.today()
    end_year = today_dt.year
    today_str = datetime.strftime(today_dt, '%Y-%m-%d')

    # find current year and season
    tmp_check_date = '%d-%s' % (end_year, fiscal_report_dates[1])
    if today_str <= tmp_check_date:
        end_year -= 1
        end_season = 4 # not pass the end of 1st season, so it is the 4th season of last year
    else:
        for tmp_season in range(2,5):
            tmp_check_date = '%d-%s' % (end_year, fiscal_report_dates[tmp_season])
            if today_str <= tmp_check_date:
                end_season = tmp_season - 1 # not pass the end of the nth season
                break

    # look backwards for 4 seasons, to check if there are records missed from previous download
    start_year = end_year - 1
    start_season = end_season

    current_year = start_year
    current_season = start_season

    write_method = 'append'
    while True:
        print('%d_s%d' % (current_year, current_season))
        # main report
        tmp_data = getDataFromTushare(ts.get_report_data, current_year, current_season)
        tmp_data = tmp_data.drop('name', axis=1)  # drop stock name
        tmp_data = tmp_data.rename({'roe': 'roe_weighted'}, axis=1)  # roe is weighted roe
        chunk_data = tmp_data.copy()

        tmp_report_period = '%d-%s' % (current_year, fiscal_report_dates[current_season]) # report period

        if not chunk_data.empty:
            # use guessed year and tushare's report date (month & day) to form a complete report date
            tmp_report_dates = chunk_data['report_date'].apply(lambda x: '%d-%s' % (current_year, x)) # complete report dates

            # use some basic principles to adjust guessed report date
            # principle 1: report date exceed today, guessed year - 1
            tmp_idx = tmp_report_dates > today_str
            if tmp_idx.sum() > 0:
                tmp_report_dates[tmp_idx] = chunk_data.loc[tmp_idx, 'report_date'].apply(lambda x: '%d-%s' % (current_year-1, x))
            # principle 2: report date earlier than the report period, guessed year + 1
            tmp_idx = tmp_report_dates < tmp_report_period
            if tmp_idx.sum() > 0:
                tmp_report_dates[tmp_idx] = chunk_data.loc[tmp_idx, 'report_date'].apply(lambda x: '%d-%s' % (current_year+1, x))

            # go back to principle 1, if there still have violation, error, set report date as nan
            tmp_idx = tmp_report_dates > today_str
            if tmp_idx.sum() > 0:
                print('invalid report dates! (today:%s)' % today_str)
                print(tmp_report_dates[tmp_idx])
                tmp_report_dates[tmp_idx] = np.nan

            chunk_data.loc[:, 'report_date'] = tmp_report_dates

            # profit
            tmp_data = getDataFromTushare(ts.get_profit_data, current_year, current_season)
            tmp_data.loc[:, 'net_profits'] = tmp_data * 100  # profit unit not consistent (million to wan yuan)
            tmp_data.loc[tmp_data['net_profits'] == 0, 'net_profits'] = np.nan  # profit impossible to be 0
            tmp_data.loc[:, 'business_income'] = tmp_data * 100  # revenue unit not consistent (million to wan yuan)
            tmp_data = tmp_data.rename({'roe': 'roe_diluted'}, axis=1)
            chunk_data = mergeData(chunk_data, tmp_data)

            # operation
            tmp_data = getDataFromTushare(ts.get_operation_data, current_year, current_season)
            chunk_data = mergeData(chunk_data, tmp_data)

            # growth
            tmp_data = getDataFromTushare(ts.get_growth_data, current_year, current_season)
            tmp_data = tmp_data.rename({'nprg': 'profits_yoy', 'epsg': 'eps_yoy'}, axis=1)
            chunk_data = mergeData(chunk_data, tmp_data)

            # solvency
            tmp_data = getDataFromTushare(ts.get_debtpaying_data, current_year, current_season)
            tmp_data.loc[:, 'currentratio'] = (tmp_data['currentratio'].apply(lambda x: x if x != '--' else np.nan)).astype(
                'float')
            tmp_data.loc[:, 'quickratio'] = (tmp_data['quickratio'].apply(lambda x: x if x != '--' else np.nan)).astype(
                'float')
            tmp_data.loc[:, 'cashratio'] = (tmp_data['cashratio'].apply(lambda x: x if x != '--' else np.nan)).astype(
                'float')
            tmp_data.loc[:, 'icratio'] = (tmp_data['icratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
            tmp_data.loc[:, 'sheqratio'] = (tmp_data['sheqratio'].apply(lambda x: x if x != '--' else np.nan)).astype(
                'float')
            tmp_data.loc[:, 'adratio'] = (tmp_data['adratio'].apply(lambda x: x if x != '--' else np.nan)).astype('float')
            chunk_data = mergeData(chunk_data, tmp_data)

            # cash flow
            tmp_data = getDataFromTushare(ts.get_cashflow_data, current_year, current_season)
            tmp_data = tmp_data.drop(['cf_nm', 'cashflowratio'], axis=1)  # drop columns unclear
            chunk_data = mergeData(chunk_data, tmp_data)

            # change column names
            chunk_data = chunk_data.rename(factorRenameDict, axis=1)

            # add year & season into the data buffer
            chunk_data.loc[:, 'fiscal_year'] = current_year
            chunk_data.loc[:, 'fiscal_season'] = current_season

            # add time stamp
            chunk_data.loc[:, 'time_stamp'] = datetime.now()

            # connect to database
            quant_engine = create_engine(
                'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

            # load existing data from database
            sql_statement = "select * from %s where fiscal_year = %d and fiscal_season = %d" % (targetTableName, current_year, current_season)
            existing_data = pd.read_sql(sql_statement, quant_engine)

            # check if database contains thia season's data
            if not existing_data.empty:
                # fill nan if new download data is not nan
                is_rewrite_sql = False  # flag to indicate whether rewrite existing data in database

                # merge existing data with newly download data
                tmp_merge_data = existing_data.merge(chunk_data, on='code', how='left', suffixes=['_l','_r'])
                tmp_merge_data.index = existing_data.index

                tmp_check_cols = chunk_data.columns.tolist()
                for tmp_col in ['code', 'fiscal_year', 'fiscal_season', 'time_stamp']:
                    tmp_check_cols.remove(tmp_col) # remove column that doesn't need to check

                code_to_modify = []
                for tmp_col in tmp_check_cols:  # check each column to see if there are nans that can be filled
                    l_tmp_col = tmp_col + '_l'
                    r_tmp_col = tmp_col + '_r'

                    tmp_idx = tmp_merge_data[l_tmp_col].isnull() & (~tmp_merge_data[r_tmp_col].isnull())
                    if tmp_idx.sum() > 0: # there exists newly downloaded, non-nan data, to fill existing data
                        is_rewrite_sql = True  # need to rewrite database
                        existing_data.loc[tmp_idx, tmp_col] = tmp_merge_data.loc[tmp_idx, r_tmp_col]  # fill nan
                        tmp_modify_code = existing_data.loc[tmp_idx, 'code'].tolist()
                        code_to_modify.extend(tmp_modify_code)

                # delete obselete data, and insert new data
                if is_rewrite_sql:
                    code_to_modify = list(set(code_to_modify))
                    tmp_code_str = list(map(lambda x: "'%s'" % x, code_to_modify))
                    tmp_code_str = ','.join(tmp_code_str)
                    tmp_conn = pymysql.connect(**ConfigQuant)
                    with tmp_conn.cursor() as tmp_cur:
                        sql_statement = "delete from %s where fiscal_year = %d and fiscal_season = %d and `code` in (%s)" % (
                                targetTableName, current_year, current_season, tmp_code_str)  # avoid duplicated data
                        tmp_num = tmp_cur.execute(sql_statement)
                        print('(%d s%d) delete previous record num: %d' % (current_year, current_season, tmp_num))
                    tmp_conn.commit()
                    tmp_conn.close()

                    # create the engine again, in case the connection has lost
                    quant_engine = create_engine(
                        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

                    modifying_data = existing_data.loc[existing_data['code'].isin(code_to_modify)]  # existing_data has already been modified
                    modifying_data.loc[:, 'time_stamp'] = datetime.now() # renew time stamp (for the purpose of subsequent renewal in derivative data)
                    modifying_data.to_sql(targetTableName, quant_engine, index=None, if_exists=write_method)
                    print('(%d s%d) renew record num: %d' % (current_year, current_season, modifying_data.shape[0]))

                # if there is new records downloaded, write new data to database
                new_data = chunk_data.loc[~chunk_data['code'].isin(existing_data['code'])]
                new_data = new_data.drop_duplicates('code')
                if not new_data.empty:
                    # create the engine again, in case the connection has lost
                    quant_engine = create_engine(
                        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

                    new_data.to_sql(targetTableName, quant_engine, index=None, if_exists=write_method)
                    print('(%d s%d) brand new record num: %d' % (current_year, current_season, new_data.shape[0]))

            else:  # database does not contain this season's data, put the whole newly download data into DB
                # create the engine again, in case the connection has lost
                quant_engine = create_engine(
                    'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))

                chunk_data.to_sql(targetTableName, quant_engine, index=None, if_exists=write_method) # write the whole newly downloaded data
                print('(%d s%d) brand new record num: %d' % (current_year, current_season, chunk_data.shape[0]))

            # move on to the next season
            current_season += 1
            if current_year == end_year and current_season > end_season:  # finish downloading
                break

            if current_season > 4:  # move on to next year
                current_season = 1
                current_year += 1


def airflowCallable():
    downloadReportIncrm()


if __name__ == '__main__':
    start_year = 2018
    start_season = 2
    end_year = 2018
    end_season = 2
    write_method = 'append'

    # downloadReportFull(start_year, start_season, end_year, end_season, write_method)

    downloadReportIncrm()