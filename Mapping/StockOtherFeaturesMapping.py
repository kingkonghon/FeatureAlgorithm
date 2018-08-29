# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicator, priceTechnicalIndicatorOHLCV, volumeTechnicalIndicators,\
    priceTechnicalIndicatorRollingSum, priceOtherIndicatorRanking


class StockOtherFeaturesMapping:
    def __init__(self, **kwargs):
        self.sourceBasicTableName = kwargs.get("sourceBasicTableName")
        self.sourceTechnicalTableName = kwargs.get("sourceTechnicalTableName")
        self.sourceCategoryTableName = kwargs.get('sourceCategoryTableName')
        self.dateField = kwargs.get("dateField")
        self.codeField = kwargs.get("codeField")
        self.techSeriesFieldWithLag = kwargs.get('techSeriesFieldWithLag')
        self.techSeriesFieldWithoutLag = kwargs.get('techSeriesFieldWithoutLag')
        self.techSeriesField = []  # combine with and without lag
        self.basicSeriesField = kwargs.get('basicSeriesField')
        self.reverseSeriesField = kwargs.get('reverseSeriesField')
        self.categoryField = kwargs.get('categoryField')
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.chunkSize = kwargs.get('chunkSize')
        self.alignFlag = kwargs.get('alignFlag')
        self.basicState = ''
        self.techState = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceBasicTableName,
                                                                   self.targetTableName, self.dateField, self.lags, '', self.alignFlag)
        if is_full == 2: # 删除了一天重跑
            is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceBasicTableName,
                                                                       self.targetTableName, self.dateField, self.lags,
                                                                       '', self.alignFlag)

        # sql statement
        tmp_basic_field = [self.dateField, self.codeField] # basic data field
        tmp_basic_field.extend(self.basicSeriesField)
        tmp_basic_field = list(map(lambda x: "`%s`" % x, tmp_basic_field))
        tmp_tech_field = [] # technical data field
        for field in self.techSeriesFieldWithLag:
            tmp_sub_tech_field = list(map(lambda x:'%s_%dD' % (field, x), self.lags))
            tmp_tech_field.extend(tmp_sub_tech_field)
        self.techSeriesField = tmp_tech_field
        self.techSeriesField.extend(self.techSeriesFieldWithoutLag)
        tmp_tech_field = [self.dateField, self.codeField]
        tmp_tech_field.extend(self.techSeriesField)
        tmp_tech_field = list(map(lambda x: "`%s`" % x, tmp_tech_field))
        basic_field_str = ",".join(tmp_basic_field)
        tech_field_str = ",".join(tmp_tech_field)
        if is_full == 1:  # 全量
            self.basicState = "SELECT %s FROM %s where " % (basic_field_str, self.sourceBasicTableName)
            self.techState = "SELECT %s FROM %s where " % (tech_field_str, self.sourceTechnicalTableName)
            self.if_exist = 'replace'
        elif is_full == 0:  # 增量
            self.basicState = "SELECT %s FROM %s where `%s` > '%s' and " % (
                basic_field_str, self.sourceBasicTableName, self.dateField, last_record_date)
            self.techState = "SELECT %s FROM %s where `%s` > '%s' and " % (
                tech_field_str, self.sourceTechnicalTableName, self.dateField, last_record_date)
            self.last_update_date = last_record_date
            self.if_exist = 'append'
        else: # 不需要跑
            self.basicState = ''
            self.techState = ''

    def run(self, startDate='2007-01-01'):
        self.prepareData()
        if self.basicState == '': # already the latest data
            return

        # get total date list (horizontally)
        tmp_state = 'select distinct %s from %s' % (self.dateField, self.sourceBasicTableName)
        date_list = readDB(tmp_state, ConfigQuant).values
        date_list = date_list.T[0]
        if self.last_update_date == '':
            date_list = date_list[date_list > startDate]
        else:
            date_list = date_list[date_list > self.last_update_date]

        #  get area (if category is area) ******************  area is the same for all dates, that means using future data!!! need to be modified
        if self.categoryField == 'area':
            tmp_state = "SELECT `%s`, `%s`  FROM `%s`;" % (
            self.codeField, self.categoryField, self.sourceCategoryTableName)
            stock_area = readDB(tmp_state, ConfigQuant)
        else:
            stock_area = None

        # calculate num of loop
        loop_num = int(date_list.size / self.chunkSize)
        if date_list.size > loop_num * self.chunkSize:
            loop_num = loop_num + 1

        # fetch and process data from sql by chunk
        for i in range(loop_num):
            tmp_date = date_list[i*self.chunkSize:(i+1)*self.chunkSize]
            tmp_date_str = list(map(lambda x:"'%s'"%x, tmp_date))
            tmp_range = ','.join(tmp_date_str)
            # read basic data
            tmp_state = self.basicState + "`%s` in (%s)" % (self.dateField, tmp_range)
            dataBasicO = readDB(tmp_state, ConfigQuant)
            for field in self.reverseSeriesField:
                dataBasicO[field] = 1. / dataBasicO[field]
            # read technical data
            tmp_state = self.techState + "`%s` in (%s)" % (self.dateField, tmp_range)
            dataTechO = readDB(tmp_state, ConfigQuant)
            for tmp_type in ['dataBasicO', 'dataTechO']:
                exec("{0} = {0}.drop_duplicates([self.dateField, self.codeField])".format(tmp_type))
                exec("{0} = {0}.sort_values(self.dateField)".format(tmp_type)) # sort by date
            # dataBasicO = dataBasicO.fillna(method='ffill')


            # for field in self.basicSeriesField:
            #     dataBasicO.loc[:, field] = dataBasicO[field].astype('float') # change data type

            # process chunk data
            data_result = pd.DataFrame([])

            # Fundamental data
            uni_dates = dataBasicO[self.dateField].unique()
            dataBasicO = self.getStockCategory(dataBasicO, uni_dates, stock_area)  # merge with stock category
            tmp_df = priceOtherIndicatorRanking(dataBasicO, uni_dates, self.dateField, self.codeField, self.basicSeriesField, self.categoryField)  # ranking
            data_result = pd.concat([data_result, tmp_df])

            # Technical data
            uni_dates = dataTechO[self.dateField].unique()
            dataTechO = self.getStockCategory(dataTechO, uni_dates, stock_area)  # merge with stock category
            tmp_df = priceOtherIndicatorRanking(dataTechO, uni_dates, self.dateField, self.codeField, self.techSeriesField, self.categoryField) # ranking
            data_result = data_result.merge(tmp_df, on=[self.dateField, self.codeField], how='outer')

            # add timestamp
            data_result['time_stamp'] = datetime.now()

            # dump chunk data into sql
            writeDB(self.targetTableName, data_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'


    def getStockCategory(self, stock_data, uni_dates, stock_area):
        if self.categoryField == 'industry':
            date_range = list(map(lambda x: "'%s'" % x, uni_dates))
            date_range = ','.join(date_range)
            tmp_state = 'select `%s`, `%s`, `%s` from %s where %s in (%s)' % (self.dateField, self.codeField,
                                                                              self.categoryField, self.sourceCategoryTableName,
                                                                              self.dateField, date_range)
            category = readDB(tmp_state, ConfigQuant)
            stock_data = stock_data.merge(category, on=[self.dateField, self.codeField])
        elif self.categoryField == 'area':
            stock_data = stock_data.merge(stock_area, on=self.codeField)
        elif self.categoryField == 'market':
            stock_data[self.categoryField] = stock_data[self.codeField].apply(lambda x: 'SH' if x[:2] == '60' else (
                'SMEB' if x[:3] == '002' else (
                    'GEB' if x[:3] == '300' else 'SZ')))
        elif self.categoryField == 'all':
            pass

        return stock_data