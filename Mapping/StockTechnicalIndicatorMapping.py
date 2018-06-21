# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicator, priceTechnicalIndicatorOHLCV, volumeTechnicalIndicators, priceTechnicalIndicatorRollingSum


class StockTechnicalIndicatorMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.dateField = kwargs.get("dateField")
        self.codeField = kwargs.get("codeField")
        self.openField = kwargs.get("openField")
        self.highField = kwargs.get("highField")
        self.lowField = kwargs.get("lowField")
        self.closeField = kwargs.get("closeField")
        self.volumeField = kwargs.get('volumeField')
        self.turnoverField = kwargs.get('turnoverField')
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.condition = kwargs.get('condition')
        self.chunkSize = kwargs.get('chunkSize')
        self.isMultiProcess = kwargs.get('isMultiProcess')
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.dateField, self.lags, self.condition)

        # sql statement
        tmp_field = [self.dateField, self.codeField, self.openField, self.highField, self.lowField, self.closeField, self.volumeField]
        if self.turnoverField != '':
            tmp_field.append(self.turnoverField)
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

        # calculate num of loop
        loop_num = int(code_list.size / self.chunkSize)
        if code_list.size > loop_num * self.chunkSize:
            loop_num = loop_num + 1

        # fetch and process data from sql by chunk
        for i in range(loop_num):
            tmp_code = code_list[i*self.chunkSize:(i+1)*self.chunkSize]
            tmp_code_str = list(map(lambda x:"'%s'"%x, tmp_code))
            tmp_range = ','.join(tmp_code_str)
            tmp_state = self.state + "`%s` in (%s)" % (self.codeField, tmp_range)
            dataO = readDB(tmp_state, ConfigQuant)
            dataO = dataO.drop_duplicates([self.dateField, self.codeField])
            dataO = dataO.sort_values(self.dateField) # sort by date
            # dataO = dataO.fillna(method='ffill') # **** fill with pre-close, not simply forward fill
            for field in [self.openField, self.highField, self.lowField, self.closeField, self.volumeField]:
                dataO.loc[:, field] = dataO[field].astype('float') # change data type
            if self.turnoverField == '': # no turnover field, fill with all nan
                self.turnoverField = 'turnover'
                dataO[self.turnoverField] = np.nan

            # process chunk data
            data_tot_result = pd.DataFrame([])
            for code in tmp_code:
                tmp_data = dataO.loc[dataO[self.codeField] == code] # dataO already sorted by date

                if tmp_data.empty:
                    continue

                data_result = tmp_data[[self.dateField, self.codeField]]

                # MA, MA_DIFF...
                tmp_df = priceTechnicalIndicator(tmp_data[self.closeField], self.lags, '')
                data_result = data_result.join(tmp_df)

                # ADX, CCI ...
                tmp_df = priceTechnicalIndicatorOHLCV(tmp_data[self.openField], tmp_data[self.highField], tmp_data[self.lowField],
                                                      tmp_data[self.closeField], tmp_data[self.volumeField])
                data_result = data_result.join(tmp_df)

                # VMA, VMA_DIFF..., OBV, VOLUME_RATIO...
                tmp_df = volumeTechnicalIndicators(tmp_data[self.closeField], tmp_data[self.volumeField], self.lags, 'V')
                data_result = data_result.join(tmp_df)

                # turnover rolling sum, amplitude
                tmp_df = priceTechnicalIndicatorRollingSum(tmp_data[self.openField], tmp_data[self.highField],
                                                           tmp_data[self.lowField],
                                                           tmp_data[self.closeField], tmp_data[self.volumeField],
                                                           tmp_data[self.turnoverField], self.lags)
                data_result = data_result.join(tmp_df)

                data_tot_result = data_tot_result.append(data_result)

            # trim data for increment
            if self.last_update_date != '':
                data_tot_result = data_tot_result.loc[data_tot_result[self.dateField] > self.last_update_date]

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
        dataO = dataO.drop_duplicates([self.dateField, self.codeField])
        dataO = dataO.sort_values(self.dateField) # sort by date
        # dataO = dataO.fillna(method='ffill') # **** fill with pre-close, not simply forward fill
        # for field in [self.openField, self.highField, self.lowField, self.closeField, self.volumeField]:
        #     dataO.loc[:, field] = dataO[field].astype('float') # change data type
        if self.turnoverField == '': # no turnover field, fill with all nan
            self.turnoverField = 'turnover'
            dataO[self.turnoverField] = np.nan

        # process incremental data
        code_list = dataO[self.codeField].unique()

        data_tot_result = pd.DataFrame([])
        # use multiprocessing to improve computation hours
        if self.isMultiProcess:
            pool = Pool(processes=4)
            pool_results = []
            # build pool
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code] # dataO already sorted by date

                if tmp_data.empty:
                    continue

                tmp_result = pool.apply_async(self.coreComputation, (tmp_data, ))
                pool_results.append(tmp_result)

            # get result from the pool
            for tmp_result in pool_results:
                data_result = tmp_result.get()
                data_tot_result = data_tot_result.append(data_result)
        else:
            for code in code_list:
                tmp_data = dataO.loc[dataO[self.codeField] == code] # dataO already sorted by date

                if tmp_data.empty:
                    continue

                data_result = self.coreComputation(tmp_data)
                data_tot_result = data_tot_result.append(data_result)

                # data_result = tmp_data[[self.dateField, self.codeField]]
                #
                # # MA, MA_DIFF...
                # tmp_df = priceTechnicalIndicator(tmp_data[self.closeField], self.lags, '')
                # data_result = data_result.join(tmp_df)
                #
                # # ADX, CCI ...
                # tmp_df = priceTechnicalIndicatorOHLCV(tmp_data[self.openField], tmp_data[self.highField], tmp_data[self.lowField],
                #                                       tmp_data[self.closeField], tmp_data[self.volumeField])
                # data_result = data_result.join(tmp_df)
                #
                # # OBV...
                # tmp_df = volumeTechnicalIndicators(tmp_data[self.closeField], tmp_data[self.volumeField], self.lags, 'V')
                # data_result = data_result.join(tmp_df)
                #
                # # turnover rolling sum, amplitude
                # tmp_df = priceTechnicalIndicatorRollingSum(tmp_data[self.openField], tmp_data[self.highField],
                #                                            tmp_data[self.lowField],
                #                                            tmp_data[self.closeField], tmp_data[self.volumeField],
                #                                            tmp_data[self.turnoverField], self.lags)
                # data_result = data_result.join(tmp_df)
                #
                # # trim data for increment
                # data_result = data_result.loc[data_result[self.dateField] > self.last_update_date]
                #
                # data_tot_result = data_tot_result.append(data_result)


        # if self.last_update_date != '':
        #     data_tot_result = data_tot_result.loc[data_tot_result[self.dateField] > self.last_update_date]

        if not data_tot_result.empty:
            # add timestamp
            data_tot_result['time_stamp'] = datetime.now()

            # dump chunk data into sql
            writeDB(self.targetTableName, data_tot_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'

    def coreComputation(self, tmp_data):
        data_result = tmp_data[[self.dateField, self.codeField]]

        # MA, MA_DIFF...
        tmp_df = priceTechnicalIndicator(tmp_data[self.closeField], self.lags, '')
        data_result = data_result.join(tmp_df)

        # ADX, CCI ...
        tmp_df = priceTechnicalIndicatorOHLCV(tmp_data[self.openField], tmp_data[self.highField],
                                              tmp_data[self.lowField],
                                              tmp_data[self.closeField], tmp_data[self.volumeField])
        data_result = data_result.join(tmp_df)

        # OBV...
        tmp_df = volumeTechnicalIndicators(tmp_data[self.closeField], tmp_data[self.volumeField], self.lags, 'V')
        data_result = data_result.join(tmp_df)

        # turnover rolling sum, amplitude
        tmp_df = priceTechnicalIndicatorRollingSum(tmp_data[self.openField], tmp_data[self.highField],
                                                   tmp_data[self.lowField],
                                                   tmp_data[self.closeField], tmp_data[self.volumeField],
                                                   tmp_data[self.turnoverField], self.lags)
        data_result = data_result.join(tmp_df)

        data_result = data_result.loc[data_result[self.dateField] > self.last_update_date]
        return data_result