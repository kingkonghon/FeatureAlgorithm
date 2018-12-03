# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre, deleteObsoleteDataFromDB
from Utils.Algorithms import getFinancialReportAccountYOY, getFinancialReportAccountTTM, getStockCashDividend, expandDataFromSeasonToDaily


class StockFundamentalTushareMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.calendarTableName = kwargs.get('calendarTableName')
        self.codeField = kwargs.get("codeField")
        self.dateField = kwargs.get('dateField')
        self.rawFields = kwargs.get('rawFields')
        self.yearField = kwargs.get('yearField')
        self.seasonField = kwargs.get('seasonField')
        self.releaseDateField = kwargs.get('releaseDateField')
        self.splitDividendField = kwargs.get('splitDividendField')
        self.valueForYOYFields = kwargs.get('valueForYOYFields')
        self.valueForTTMFields = kwargs.get('valueForTTMFields')
        self.originalYOYFields = kwargs.get('originalYOYFields')
        self.ratioFields = kwargs.get('ratioFields')
        self.timeStampField = kwargs.get('timeStampField')
        # self.lagYearNum = kwargs.get('lagYearNum')
        self.targetTableName = kwargs.get('targetTableName')
        self.condition = kwargs.get('condition')
        self.chunkSize = kwargs.get('chunkSize')
        self.isMultiProcess = kwargs.get('isMultiProcess')
        self.processNum = kwargs.get('processNum')
        self.reportPeriodField = 'REPORT_PERIOD'
        self.latestTimeStamp = 'latest_time_stamp'
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.timeStampField, [252], self.condition)

        # sql statement
        # tmp_field = list(map(lambda x: "`%s`" % x, tmp_field))
        # fields = ",".join(tmp_field)
        season_end_date = {
            1: '03-31',
            2: '06-30',
            3: '09-30',
            4: '12-31'
        }
        today_dt = datetime.today()
        today_str = datetime.strftime(today_dt, '%Y-%m-%d')
        current_year = today_dt.year
        if today_str < '%d-%s' % (current_year, season_end_date[1]):
            current_season = 4
        else:
            for tmp_season in range(1,4):
                if (today_str >= '%d-%s' % (current_year, season_end_date[tmp_season])) and (today_str < '%d-%s' % (current_year, season_end_date[tmp_season + 1])):
                    current_season = tmp_season
                    break
        current_period = current_year * 100 + current_season
        last_year_period = (current_year - 2) * 100 + current_season # load 2 years' old data in order to calculate ttm and yoy (last year's yoy need the data of 2 years ago), and might have nan filled by new downloads
        if is_full == 1:  # 全量
            self.state = "SELECT * FROM %s where " % self.sourceTableName
            if self.condition != '':
                self.state = self.state + self.condition + ' and ' # add search condition
            self.if_exist = 'replace'
        # elif is_full == 0:  # 增量
        else:  # even if source do not have new data, still update target
            self.last_update_date = datetime.strftime(last_record_date, '%Y-%m-%d')
            self.state = "SELECT * FROM %s where (`%s` * 100 + `%s` >= '%d') and (`%s` * 100 + `%s` <= '%d') " % (
                self.sourceTableName, self.yearField, self.seasonField, last_year_period, self.yearField, self.seasonField, current_period)
            if self.condition != '':
                self.state = self.state + ' and ' + self.condition  # add search condition
            self.if_exist = 'append'
        # else: # 不需要跑
        #     self.state = ''

    def run(self):
        self.prepareData()
        if self.state == '': # already the latest data
            return
        elif self.last_update_date != '':
            self.runIncrm()
        else:
            self.runFull()

    def runFull(self):
        # get total code list
        tmp_state = 'select distinct %s from %s' % (self.codeField, self.sourceTableName)
        code_list = readDB(tmp_state, ConfigQuant).values
        code_list = code_list.T[0]

        # get trade date list from calendar
        tmp_state = 'select `date` from %s' % self.calendarTableName
        tradedates = readDB(tmp_state, ConfigQuant)['date']

        # calculate num of loop
        loop_num = int(code_list.size / self.chunkSize)
        if code_list.size > loop_num * self.chunkSize:
            loop_num = loop_num + 1

        if self.isMultiProcess: # use multi processing
            # register pool
            pool = Pool(processes=self.processNum)
            # fetch and process data from sql by chunk
            for i in range(loop_num):
                tmp_code = code_list[i * self.chunkSize:(i + 1) * self.chunkSize]
                tmp_code_str = list(map(lambda x: "'%s'" % x, tmp_code))
                tmp_range = ','.join(tmp_code_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.codeField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO.loc[:, self.reportPeriodField] = dataO[self.yearField] * 100 + dataO[self.seasonField]  # combine report year and season
                dataO = dataO.drop_duplicates([self.codeField, self.reportPeriodField])
                dataO = dataO.sort_values(self.reportPeriodField)  # sort by report period

                # process chunk data
                pool_results = []
                for code in tmp_code:
                    tmp_data = dataO.loc[dataO[self.codeField] == code]  # dataO already sorted by date

                    if tmp_data.empty:
                        continue

                    # multiprocessing
                    tmp_procs = pool.apply_async(self.coreComputation, (code, tmp_data, tradedates))
                    pool_results.append(tmp_procs)

                # get result from the process pool
                data_tot_result = pd.DataFrame([])
                for tmp_procs in pool_results:
                    data_result = tmp_procs.get()
                    data_tot_result = data_tot_result.append(data_result)

                # add timestamp
                data_tot_result['time_stamp'] = datetime.now()

                if data_tot_result.empty:
                    continue

                # dump chunk data into sql
                writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
                self.if_exist = 'append'

            pool.close()
        else:  # not multiprocess
            # fetch and process data from sql by chunk
            for i in range(loop_num):
                tmp_code = code_list[i*self.chunkSize:(i+1)*self.chunkSize]
                tmp_code_str = list(map(lambda x:"'%s'"%x, tmp_code))
                tmp_range = ','.join(tmp_code_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.codeField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO.loc[:, self.reportPeriodField] = dataO[self.yearField] * 100 + dataO[
                    self.seasonField]  # combine report year and season
                dataO = dataO.drop_duplicates([self.codeField, self.reportPeriodField])
                # dataO = dataO.sort_values(self.reportPeriodField)  # sort by report period

                data_tot_result = pd.DataFrame([])
                for code in tmp_code:
                    tmp_data = dataO.loc[dataO[self.codeField] == code] # dataO already sorted by date
                    tmp_data = tmp_data.sort_values(self.reportPeriodField) # sort by report period

                    if tmp_data.empty:
                        continue

                    data_result = self.coreComputation(code, tmp_data, tradedates)

                    data_tot_result = data_tot_result.append(data_result)

                # add timestamp
                data_tot_result['time_stamp'] = datetime.now()

                if data_tot_result.empty:
                    continue

                # dump chunk data into sql
                writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
                self.if_exist = 'append'


    def runIncrm(self):
        # fetch and process all incremental data from sql
        dataO = readDB(self.state, ConfigQuant)
        dataO.loc[:, self.reportPeriodField] = dataO[self.yearField] * 100 + dataO[self.seasonField]
        dataO = dataO.drop_duplicates([self.codeField, self.reportPeriodField])
        # dataO = dataO.sort_values(self.reportPeriodField) # sort by date

        # get calendar
        tmp_state = "select `date` from `%s`;" % self.calendarTableName
        trade_calendar = readDB(tmp_state, ConfigQuant)
        trade_calendar = trade_calendar['date']

        # get latest time stamp for each stock in the target table
        tmp_state = "select `%s`, max(`%s`) as %s from `%s` group by `%s`" % (
            self.codeField, self.timeStampField, self.latestTimeStamp, self.targetTableName, self.codeField)
        target_latest_time_stamp = readDB(tmp_state, ConfigQuant)
        target_latest_time_stamp = target_latest_time_stamp.set_index(self.codeField)

        # get the latest trade date of the data in target table
        tmp_state = "select max(`%s`) from `%s`" % (self.dateField, self.targetTableName)
        target_latest_trade_date = readDB(tmp_state, ConfigQuant)
        target_latest_trade_date = target_latest_trade_date.iloc[0, 0]

        # process incremental data
        code_list = dataO[self.codeField].unique()

        data_tot_result = pd.DataFrame([])
        # use multiprocessing to improve computation hours
        if self.isMultiProcess:
            pool = Pool(processes=self.processNum)
            pool_results = []
            pool_data_first_date = []
            no_update_code_list = []
            # build pool
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code]
                tmp_data = tmp_data.sort_values(self.reportPeriodField)  #  sorted by report period

                if tmp_data.empty:
                    continue

                # find the latest time stamp, and compare it with the raw data, delete obsolete data if there exists
                is_new_data = False
                try:
                    tmp_target_latest_time_stamp = target_latest_time_stamp.loc[code, self.latestTimeStamp]
                    tmp_data_source_new = tmp_data.loc[tmp_data[self.timeStampField] >= tmp_target_latest_time_stamp]
                    tmp_data_source_unexpanded = tmp_data.loc[tmp_data[self.releaseDateField] > target_latest_trade_date]
                    tmp_data_source_new = tmp_data_source_new.append(tmp_data_source_unexpanded)
                    tmp_data_source_new = tmp_data_source_new.drop_duplicates([self.codeField, self.yearField, self.seasonField])

                    tmp_data_source_new = tmp_data_source_new.loc[~tmp_data_source_new[self.releaseDateField].isnull()]

                    if not tmp_data_source_new.empty: # obsolete data
                        data_new_first_data = tmp_data_source_new[self.releaseDateField].min()  # find the earliest report in new update data

                        if type(data_new_first_data).__name__ == 'str':  # else data_new_first_data is nan
                            is_new_data = True
                            deleteObsoleteDataFromDB(code, data_new_first_data, self.dateField, self.codeField,
                                                     self.targetTableName, ConfigQuant)  # delete obsolet data not earlier than the eariliest report date
                except KeyError:
                    is_new_data = True
                    data_new_first_data = '2007-01-01' # this stock code is new to the target table

                if is_new_data: # have values updated or completely new
                    tmp_result = pool.apply_async(self.coreComputation, (code, tmp_data, trade_calendar))
                    pool_results.append(tmp_result)
                    data_new_first_data = min(data_new_first_data, trade_calendar[trade_calendar > target_latest_trade_date].iloc[0])
                    pool_data_first_date.append(data_new_first_data)
                else:  # no new data from source table to update target table
                    no_update_code_list.append(code)

            # get result from the pool
            for tmp_result, tmp_first_date in zip(pool_results, pool_data_first_date):
                data_result = tmp_result.get()
                data_result = data_result.loc[data_result[self.dateField] >= tmp_first_date]  # slice data, from the earliest report release date
                print('%s regenerate %d data' % (code, data_result.shape[0]))
                data_tot_result = data_tot_result.append(data_result)

            # replicate the latest records for those not updated codes
            replicate_records = self.replicateLatestRecord(no_update_code_list, trade_calendar, target_latest_trade_date)
            data_tot_result = data_tot_result.append(replicate_records)

        else: # single process
            no_update_code_list = []
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code]
                tmp_data = tmp_data.sort_values(self.reportPeriodField)  # sorted by report period

                if tmp_data.empty:
                    continue

                # find the latest time stamp, and compare it with the raw data, delete obsolete data if there exists
                has_new_data = False
                try:
                    tmp_target_latest_time_stamp = target_latest_time_stamp.loc[code, self.latestTimeStamp]
                    tmp_data_source_new = tmp_data[tmp_data[self.timeStampField] >= tmp_target_latest_time_stamp]
                    tmp_data_source_unexpanded = tmp_data[tmp_data[self.releaseDateField] > target_latest_trade_date]
                    tmp_data_source_new = tmp_data_source_new.append(tmp_data_source_unexpanded)
                    tmp_data_source_new = tmp_data_source_new.drop_duplicates([self.codeField, self.yearField, self.seasonField])

                    if not tmp_data_source_new.empty:  # obsolete data
                        tmp_data_new_first_data = tmp_data_source_new[
                            self.releaseDateField].min()  # find the earliest report in new update data
                        if type(tmp_data_new_first_data).__name__ == 'str': # else tmp_data_new_first_data is nan
                            has_new_data = True
                            deleteObsoleteDataFromDB(code, tmp_data_new_first_data, self.dateField, self.codeField,
                                                     self.targetTableName, ConfigQuant)  # delete obsolet data later than the eariliest report date
                except KeyError:
                    has_new_data = True  # this stock code is new to the target table
                    tmp_data_new_first_data = '2007-01-01'

                if has_new_data:
                    data_result = self.coreComputation(code, tmp_data, trade_calendar)
                    tmp_data_new_first_data = min(tmp_data_new_first_data, trade_calendar[trade_calendar > target_latest_trade_date].iloc[0])
                    data_result = data_result.loc[data_result[self.dateField] >= tmp_data_new_first_data]
                    print('%s regenerate %d data' % (code, data_result.shape[0]))
                    data_tot_result = data_tot_result.append(data_result)
                else: # no new data from source table to update target table
                    no_update_code_list.append(code)

            # replicate latest records for those not updated codes
            replicate_records = self.replicateLatestRecord(no_update_code_list, trade_calendar, target_latest_trade_date)
            data_tot_result = data_tot_result.append(replicate_records)

        if not data_tot_result.empty:
            # add timestamp
            data_tot_result['time_stamp'] = datetime.now()

            # dump chunk data into sql
            writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'


    def coreComputation(self, tmp_code, tmp_data, tradedates):
        data_result = tmp_data.copy()

        # calculate yearly increment percent (yoy)
        # yoy_cols = ['parent_net_profits_yoy', 'eps_yoy', 'bvps_yoy', 'revenue_yoy', 'sales_per_share_yoy', 'cfo_per_share_yoy']
        yoy_cols = list(map(lambda x: x + '_yoy', self.valueForYOYFields))

        ts_data_cols = tmp_data.columns.tolist()
        for tmp_yoy_col, tmp_ori_col in zip(yoy_cols, self.valueForYOYFields):
            tmp_yoy = getFinancialReportAccountYOY(tmp_data, tmp_ori_col, self.reportPeriodField)  # calculate derivative from original

            if tmp_yoy_col in ts_data_cols:  # if tushare data already has a yoy column, fill nan
                tmp_yoy = pd.Series(tmp_yoy, index=tmp_data.index)

                tmp_idx = (~tmp_yoy.isnull()) & (tmp_data[tmp_yoy_col].isnull())  # fill nan by calculated data
                if tmp_idx.sum() > 0:
                    data_result.loc[tmp_idx, tmp_yoy_col] = tmp_yoy[tmp_idx]
            else: # if tushare data do not have this yoy data, create a new column
                data_result.loc[:, tmp_yoy_col] = tmp_yoy

        # calculate dividend
        dvd_col_name = 'dividend'
        tmp_data.loc[:, dvd_col_name] = np.nan
        tmp_idx = ~tmp_data[self.splitDividendField].isnull()
        tmp_data.loc[tmp_idx, dvd_col_name] = tmp_data.loc[tmp_idx, self.splitDividendField].apply(lambda x: getStockCashDividend(x))
        tmp_data.loc[:, dvd_col_name] = tmp_data[dvd_col_name].astype('float')

        # calculate TTM values
        # ttm_cols = ['eps_ttm', 'cfo_per_share_ttm', 'parent_net_profits_ttm', 'sales_per_share_ttm', 'dividend_ttm']
        original_data_for_ttm_cols = self.valueForTTMFields.copy()
        # original_data_for_ttm_cols.append(dvd_col_name)  #
        ttm_cols = list(map(lambda x: x + '_ttm', original_data_for_ttm_cols))
        for tmp_ttm_col, tmp_ori_col in zip(ttm_cols, original_data_for_ttm_cols):
            data_result.loc[:, tmp_ttm_col] = getFinancialReportAccountTTM(tmp_data, tmp_ori_col, self.reportPeriodField, self.yearField, self.seasonField)  # calculate derivative from original

        # select columns to expand to daily data
        col_to_expand = [self.codeField, self.yearField, self.seasonField, self.releaseDateField]
        col_to_expand.extend(yoy_cols)
        col_to_expand.extend(ttm_cols)
        col_to_expand.extend(self.ratioFields)
        data_result = data_result[col_to_expand]

        # proliferate seasonal data into daily data
        today_str = datetime.strftime(datetime.now(), '%Y-%m-%d')
        daily_data_result = expandDataFromSeasonToDaily(tmp_code, data_result, tradedates, self.releaseDateField, today_str)

        # drop duplicated daily data
        daily_data_result = self.dropDailyRecordDuplicates(daily_data_result)

        return daily_data_result

    def replicateLatestRecord(self, no_update_code_list, trade_calendar, target_latest_trade_date):
        data_tot_result = pd.DataFrame([])

        # copy the latest records for those not updated codes
        today_str = datetime.strftime(datetime.now(), '%Y-%m-%d')
        expand_dates = trade_calendar[(trade_calendar > target_latest_trade_date) & (trade_calendar <= today_str)]
        if not expand_dates.empty:  # today is not the latest record date in target table
            tmp_not_updated_codes = list(map(lambda x: "'%s'" % x, no_update_code_list))
            tmp_not_updated_codes = ','.join(tmp_not_updated_codes)

            # get the latest record from target table
            tmp_state = "select * from `%s` where `%s` in (%s) and `%s` = '%s'" % (self.targetTableName,
                                                                                   self.codeField,
                                                                                   tmp_not_updated_codes,
                                                                                   self.dateField, target_latest_trade_date)
            tmp_latest_records = readDB(tmp_state, ConfigQuant)

            # expand the latest record up to today
            for tmp_idx in tmp_latest_records.index:
                tmp_record = tmp_latest_records.loc[tmp_idx]
                tmp_expand_records = pd.DataFrame([])
                tmp_expand_records = tmp_expand_records.append([tmp_record] * expand_dates.size)
                tmp_expand_records.loc[:, self.dateField] = expand_dates.tolist()
                data_tot_result = data_tot_result.append(tmp_expand_records)

            data_tot_result.loc[:, self.timeStampField] = datetime.now()

        return data_tot_result

    def dropDailyRecordDuplicates(self, daily_records):
        # seperate all duplicated data of one stock
        tmp_idx = daily_records[self.dateField].duplicated(keep=False)
        duplicated_daily_data = daily_records.loc[tmp_idx]
        distinct_daily_data = daily_records.loc[~tmp_idx]

        if not duplicated_daily_data.empty:
            duplicated_daily_data.loc[:, self.reportPeriodField] = duplicated_daily_data[self.yearField] * 100 + duplicated_daily_data[self.seasonField]

            # get the latest report period of all duplicated group
            max_report_period = duplicated_daily_data.groupby(self.dateField)[self.reportPeriodField].max()

            # use date and max report period to draw unique data from duplicated data
            new_distinct_daily_data = pd.DataFrame([])
            for tmp_date in max_report_period.index:
                tmp_uni_data = duplicated_daily_data.loc[(duplicated_daily_data[self.dateField] == tmp_date) &
                                (duplicated_daily_data[self.reportPeriodField] == max_report_period[tmp_date])]

                new_distinct_daily_data = new_distinct_daily_data.append(tmp_uni_data)

            # combine new distinct daily with the original distinct daily data
            new_distinct_daily_data = new_distinct_daily_data.drop(self.reportPeriodField, axis=1)
            distinct_daily_data = distinct_daily_data.append(new_distinct_daily_data)

        return distinct_daily_data