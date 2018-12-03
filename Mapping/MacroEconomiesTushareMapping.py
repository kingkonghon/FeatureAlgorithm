# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicatorTimeSeries


class MacroEconomiesTushareMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.dateField = kwargs.get("dateField")
        self.yearField = kwargs.get("yearField")
        self.monthField = kwargs.get('monthField')
        self.seasonField = kwargs.get('seasonField')
        self.sourceFields = kwargs.get("sourceFields")
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.targetFields = kwargs.get('targetFields')
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.dateField, self.lags, '')

        # sql statement
        fields = ','.join([self.dateField, self.yearField, self.monthField, self.seasonField])
        fields = fields + "," + ",".join(self.sourceFields)
        if is_full == 1:  # 全量
            self.state = "SELECT %s FROM %s" % (fields, self.sourceTableName)
            self.if_exist = 'replace'
        elif is_full == 0:  # 增量
            self.last_update_date = last_record_date
            self.state = "SELECT %s FROM %s where `%s` > '%s'" % (
                fields, self.sourceTableName, self.dateField, start_fetch_date)
            self.if_exist = 'append'
        else: # 不需要跑
            self.state = ''

    def run(self):
        self.prepareData()
        if self.state == '': # already the latest data
            return

        # fetch data from sql
        dataO = readDB(self.state, ConfigQuant)
        dataO = dataO.drop_duplicates([self.dateField])
        dataO = dataO.sort_values(self.dateField)   # sorted by date

        # process data
        data_result = dataO[[self.dateField, self.yearField, self.seasonField, self.monthField]]
        tmp_ratio = self.getRatio(dataO, self.dateField, self.yearField, self.monthField, 'm0', 'm0_mom')
        data_result = data_result.join(tmp_ratio, how='left')

        tmp_ratio = self.getRatio(dataO, self.dateField, self.yearField, self.monthField, 'm1', 'm1_mom')
        data_result = data_result.join(tmp_ratio, how='left')

        tmp_ratio = self.getRatio(dataO, self.dateField, self.yearField, self.monthField, 'm2', 'm2_mom')
        data_result = data_result.join(tmp_ratio, how='left')

        tmp_ratio = self.getRatio(dataO, self.dateField, self.yearField, self.monthField, 'ppi', 'ppi_mom')
        data_result = data_result.join(tmp_ratio, how='left')

        tmp_ratio = self.getRatio(dataO, self.dateField, self.yearField, self.yearField, 'ppi', 'ppi_yoy')
        data_result = data_result.join(tmp_ratio, how='left')

        # rearrage data columns
        tmp_cols = [self.dateField, self.yearField, self.seasonField, self.monthField] + self.targetFields
        data_result = data_result[tmp_cols]

        # trim data for increment
        if self.last_update_date != '':
            data_result = data_result.loc[data_result[self.dateField] > self.last_update_date]

        # add timestamp
        data_result['time_stamp'] = datetime.now()

        # dump data into sql
        writeDB(self.targetTableName, data_result, ConfigQuant, self.if_exist)
        # self.if_exist = 'append'

    def getRatio(self, data, date_col, year_col, frequency_col, source_col, ratio_col):
        if year_col != frequency_col:
            tmp_data = data[[year_col, frequency_col, source_col]]
            tmp_data = tmp_data.drop_duplicates([year_col, frequency_col])  # already sorted, keep the first
        else:
            tmp_data = data[[frequency_col, source_col]]  # yearly ratio
            tmp_data = tmp_data.drop_duplicates(frequency_col)  # already sorted

        tmp_data.loc[:, ratio_col] = tmp_data[source_col] / tmp_data[source_col].shift(1) - 1

        data_result = data[[date_col]]
        data_result = data_result.join(tmp_data[ratio_col], how='left')
        data_result.loc[:, ratio_col] = data_result[ratio_col].fillna(method='ffill')  # join monthly data, fill daily

        return data_result[ratio_col]


