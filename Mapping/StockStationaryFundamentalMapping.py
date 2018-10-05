# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicatorTimeSeries, amountTechnicalIndicators


class StockStationaryFundamentalIndicatorMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.calendarTableName = kwargs.get('calendarTableName')
        self.dateField = kwargs.get("dateField")
        self.codeField = kwargs.get("codeField")
        self.valueFields = kwargs.get('valueFields')
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.condition = kwargs.get('condition')
        self.chunkSize = kwargs.get('chunkSize')
        self.isMultiProcess = kwargs.get('isMultiProcess')
        self.processNum = kwargs.get('processNum')
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''
        self.init_date = '2007-01-01'

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.dateField, self.lags, self.condition)

        # sql statement
        tmp_field = [self.dateField, self.codeField]
        tmp_field.extend(self.valueFields)
        tmp_field = list(map(lambda x: "`%s`" % x, tmp_field))
        fields = ",".join(tmp_field)
        if is_full == 1:  # 全量
            self.state = "SELECT %s FROM %s where " % (fields, self.sourceTableName)
            if self.condition != '':
                self.state = self.state + self.condition + ' and ' # add search condition
            self.if_exist = 'replace'
        elif is_full == 0:  # 增量
            self.last_update_date = last_record_date
            self.state = "SELECT %s FROM %s where `%s` > '%s' " % (
                fields, self.sourceTableName, self.dateField, start_fetch_date)
            if self.condition != '':
                self.state = self.state + ' and ' + self.condition  # add search condition
            self.if_exist = 'append'
        else: # 不需要跑
            self.state = ''

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

        # get total trade dates
        tmp_state = "select `%s` from %s" % (self.dateField, self.calendarTableName)
        trade_calendar = readDB(tmp_state, ConfigQuant)
        trade_calendar = trade_calendar[self.dateField].values
        trade_calendar = trade_calendar[trade_calendar >= self.init_date]

        # calculate num of loop for time series calculation
        code_chunk_num = int(code_list.size / self.chunkSize)
        if code_list.size > code_chunk_num * self.chunkSize:
            code_chunk_num = code_chunk_num + 1

        # calculate num of loop for time horizon calculation
        date_chunk_num = int(trade_calendar.size / self.chunkSize)
        if trade_calendar.size > date_chunk_num * self.chunkSize:
            date_chunk_num = date_chunk_num + 1

        if self.isMultiProcess: # use multi processing
            # register pool
            pool = Pool(processes=self.processNum)

            # calculate time series derivative features
            data_tot_result_time_series = pd.DataFrame([])
            for i in range(code_chunk_num):
                tmp_code = code_list[i * self.chunkSize:(i + 1) * self.chunkSize]
                tmp_code_str = list(map(lambda x: "'%s'" % x, tmp_code))
                tmp_range = ','.join(tmp_code_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.codeField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO = dataO.drop_duplicates([self.dateField, self.codeField])
                # dataO = dataO.sort_values(self.dateField)  # sort by date

                # process chunk data
                pool_results = []
                for code in tmp_code:
                    tmp_data = dataO.loc[dataO[self.codeField] == code]  # dataO already sorted by date
                    tmp_data = tmp_data.sort_values(self.dateField)

                    if tmp_data.empty:
                        continue

                    # multiprocessing
                    tmp_procs = pool.apply_async(self.coreComputationTimeSeries, (tmp_data,))
                    pool_results.append(tmp_procs)
                    # data_result = self.coreComputationFull(tmp_data)

                # get result from the process pool
                for tmp_procs in pool_results:
                    data_result = tmp_procs.get()
                    data_tot_result_time_series = data_tot_result_time_series.append(data_result)

            # calculate time horizon derivative features
            data_tot_result_time_horizon = pd.DataFrame([])
            for i in range(date_chunk_num):
                tmp_date = trade_calendar[i * self.chunkSize:(i + 1) * self.chunkSize]
                tmp_date_str = list(map(lambda x: "'%s'" % x, tmp_date))
                tmp_range = ','.join(tmp_date_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.dateField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO = dataO.drop_duplicates([self.dateField, self.codeField])
                # dataO = dataO.sort_values(self.dateField)  # sort by date

                # process chunk data
                pool_results = []
                for single_date in tmp_date:
                    tmp_data = dataO.loc[dataO[self.dateField] == single_date]  # dataO already sorted by date

                    if tmp_data.empty:
                        continue

                    # multiprocessing
                    tmp_procs = pool.apply_async(self.coreComputationTimeHorizon, (tmp_data,))
                    pool_results.append(tmp_procs)
                    # data_result = self.coreComputationFull(tmp_data)

                # get result from the process pool
                for tmp_procs in pool_results:
                    data_result = tmp_procs.get()
                    data_tot_result_time_horizon = data_tot_result_time_horizon.append(data_result)

            # combine time series and time horizon features
            data_tot_result = data_tot_result_time_series.merge(data_tot_result_time_horizon, on=[self.dateField, self.codeField], how='outer')

            # add timestamp
            data_tot_result.loc[:, 'time_stamp'] = datetime.now()

            # dump all data into database
            writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'

            pool.close()
        else:
            # calculate time series derivative features
            data_tot_result_time_series = pd.DataFrame([])
            for i in range(code_chunk_num):
                tmp_code = code_list[i * self.chunkSize:(i + 1) * self.chunkSize]
                tmp_code_str = list(map(lambda x: "'%s'" % x, tmp_code))
                tmp_range = ','.join(tmp_code_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.codeField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO = dataO.drop_duplicates([self.dateField, self.codeField])
                # dataO = dataO.sort_values(self.dateField)  # sort by date

                # process chunk data
                for code in tmp_code:
                    tmp_data = dataO.loc[dataO[self.codeField] == code]  # dataO already sorted by date
                    tmp_data = tmp_data.sort_values(self.dateField)

                    if tmp_data.empty:
                        continue

                    data_result = self.coreComputationTimeSeries(tmp_data)
                    data_tot_result_time_series = data_tot_result_time_series.append(data_result)

            # calculate time horizon derivative features
            data_tot_result_time_horizon = pd.DataFrame([])
            for i in range(date_chunk_num):
                tmp_date = trade_calendar[i * self.chunkSize:(i + 1) * self.chunkSize]
                tmp_date_str = list(map(lambda x: "'%s'" % x, tmp_date))
                tmp_range = ','.join(tmp_date_str)
                tmp_state = self.state + "`%s` in (%s)" % (self.dateField, tmp_range)
                dataO = readDB(tmp_state, ConfigQuant)
                dataO = dataO.drop_duplicates([self.dateField, self.codeField])
                # dataO = dataO.sort_values(self.dateField)  # sort by date

                # process chunk data
                for single_date in tmp_date:
                    tmp_data = dataO.loc[dataO[self.dateField] == single_date]  # dataO already sorted by date

                    if tmp_data.empty:
                        continue

                    # multiprocessing
                    data_result = self.coreComputationTimeHorizon(tmp_data)
                    data_tot_result_time_horizon = data_tot_result_time_horizon.append(data_result)

            # combine time series and time horizon features
            data_tot_result = data_tot_result_time_series.merge(data_tot_result_time_horizon,
                                                                on=[self.dateField, self.codeField], how='outer')

            # add timestamp
            data_tot_result.loc[:, 'time_stamp'] = datetime.now()

            # dump chunk data into database
            writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'


    def runIncrm(self):
        # fetch and process all incremental data from sql
        dataO = readDB(self.state, ConfigQuant)
        dataO = dataO.drop_duplicates([self.dateField, self.codeField])
        # dataO = dataO.sort_values(self.dateField) # sort by date

        # process incremental data
        code_list = dataO[self.codeField].unique()
        trade_calendar = dataO[self.dateField].unique()
        trade_calendar = trade_calendar[trade_calendar > self.last_update_date]

        # ==== calculate features
        if self.isMultiProcess:
            # build pool
            pool = Pool(processes=self.processNum)

            # calculate time series features
            data_tot_result_time_series = pd.DataFrame([])
            pool_results = []
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code]
                tmp_data = tmp_data.sort_values(self.dateField) # sort by date

                if tmp_data.empty:
                    continue

                tmp_result = pool.apply_async(self.coreComputationTimeSeries, (tmp_data, ))
                pool_results.append(tmp_result)

            # get result from the pool (time series)
            for tmp_result in pool_results:
                data_result = tmp_result.get()
                data_tot_result_time_series = data_tot_result_time_series.append(data_result)

            # calculate time horizon features
            data_tot_result_time_horizon = pd.DataFrame([])
            pool_results = []
            for trade_date in trade_calendar:
                tmp_data = dataO.loc[dataO[self.dateField] == trade_date]  # no need to sort date

                if tmp_data.empty:
                    continue

                tmp_result = pool.apply_async(self.coreComputationTimeHorizon, (tmp_data,))
                pool_results.append(tmp_result)

            # get result from the pool (time horizon)
            for tmp_result in pool_results:
                data_result = tmp_result.get()
                data_tot_result_time_horizon = data_tot_result_time_horizon.append(data_result)

            pool.terminate()

        # single process
        else:
            # calculate time series features
            data_tot_result_time_series = pd.DataFrame([])
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code]
                tmp_data = tmp_data.sort_values(self.dateField) # sorted by date

                if tmp_data.empty:
                    continue

                data_result = self.coreComputationTimeSeries(tmp_data)
                data_tot_result_time_series = data_tot_result_time_series.append(data_result)

            # calculate time horizon features
            data_tot_result_time_horizon = pd.DataFrame([])
            for trade_date in trade_calendar:
                tmp_data = dataO.loc[dataO[self.dateField] == trade_date]  # dataO already sorted by date

                if tmp_data.empty:
                    continue

                data_result = self.coreComputationTimeHorizon(tmp_data)
                data_tot_result_time_horizon = data_tot_result_time_horizon.append(data_result)

        # write to sql (both multiprocess and single process)
        data_tot_result = data_tot_result_time_series.merge(data_tot_result_time_horizon, on=[self.dateField, self.codeField], how='outer') # combine time series and time horizon features
        data_tot_result = data_tot_result.loc[data_tot_result[self.dateField] > self.last_update_date] # truncate data

        if not data_tot_result.empty:
            # add timestamp
            data_tot_result['time_stamp'] = datetime.now()

            # dump chunk data into sql
            writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)

    def coreComputationTimeSeries(self, tmp_data):
        data_result = tmp_data[[self.dateField, self.codeField]]

        # loop over fields
        for tmp_field in self.valueFields:
            # current_value / MA
            tmp_df = priceTechnicalIndicatorTimeSeries(tmp_data[tmp_field], self.lags, tmp_field)
            data_result = data_result.join(tmp_df)

        return data_result

    def coreComputationTimeHorizon(self, tmp_data):
        data_result = tmp_data[[self.dateField, self.codeField]]

        # loop over fields
        for tmp_field in self.valueFields:
            tmp_value = np.log(tmp_data[tmp_field])  # take log for the distribution to be more normal
            tmp_value = tmp_value.replace([np.inf, -np.inf], np.nan)
            # (value - mean) / std
            tmp_norm_value = (tmp_value - tmp_value.mean(skipna=True)) / tmp_value.std(skipna=True)
            tmp_norm_value.name = 'NORM_LOG_' + tmp_field

            data_result = data_result.join(tmp_norm_value)

        return data_result