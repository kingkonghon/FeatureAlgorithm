# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicator, priceTechnicalIndicatorOHLCV, volumeTechnicalIndicators,\
    priceTechnicalIndicatorRollingSum, priceOtherIndicatorRanking


class ExcessReturnMapping:
    def __init__(self, **kwargs):
        self.sourceStockTableName = kwargs.get("sourceStockTableName")
        self.sourceIndexTableName = kwargs.get("sourceIndexTableName")
        self.sourceCategoryTableName = kwargs.get('sourceCategoryTableName')
        self.dateField = kwargs.get("dateField")
        self.codeField = kwargs.get("codeField")
        self.categoryField = kwargs.get('categoryField')
        self.retSeriesField = kwargs.get('retSeriesField')
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.chunkSize = kwargs.get('chunkSize')
        self.stockState = ''
        self.indexState = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceStockTableName,
                                                                   self.targetTableName, self.dateField, self.lags,  '')

        # sql statement
        tmp_ret_field = [] # technical data field
        for field in self.retSeriesField:
            tmp_sub_ret_field = list(map(lambda x:'%s_%dD' % (field, x), self.lags))
            tmp_ret_field.extend(tmp_sub_ret_field)
        self.retSeriesField = tmp_ret_field
        tmp_stock_field = [self.dateField, self.codeField]
        if self.categoryField == 'HS300':
            tmp_index_field = [self.dateField]
        else:
            tmp_index_field = [self.dateField, self.categoryField]
        tmp_stock_field.extend(self.retSeriesField)
        tmp_index_field.extend(self.retSeriesField)
        stock_field_str = list(map(lambda x: "`%s`" % x, tmp_stock_field))
        index_field_str = list(map(lambda x: "`%s`" % x, tmp_index_field))
        stock_field_str = ",".join(stock_field_str)
        index_field_str = ",".join(index_field_str)
        if is_full == 1:  # 全量
            self.stockState = "SELECT %s FROM %s where " % (stock_field_str, self.sourceStockTableName)
            self.indexState = "SELECT %s FROM %s where " % (index_field_str, self.sourceIndexTableName)
            self.if_exist = 'replace'
        elif is_full == 0:  # 增量
            self.stockState = "SELECT %s FROM %s where `%s` > '%s' and " % (
                stock_field_str, self.sourceStockTableName, self.dateField, last_record_date)
            self.indexState = "SELECT %s FROM %s where `%s` > '%s' and " % (
                index_field_str, self.sourceIndexTableName, self.dateField, last_record_date)
            self.last_update_date = last_record_date
            self.if_exist = 'append'
        else: # 不需要跑
            self.stockState = ''
            self.indexState = ''

    def run(self, startDate='2007-01-01'):
        self.prepareData()
        if self.stockState == '': # already the latest data
            return

        # get total date list (horizontally)
        tmp_state = 'select distinct %s from %s' % (self.dateField, self.sourceStockTableName)
        date_list = readDB(tmp_state, ConfigQuant).values
        date_list = date_list.T[0]
        if self.last_update_date == '':
            date_list = date_list[date_list > startDate]
        else:
            date_list = date_list[date_list > self.last_update_date]

        #  ******************  area is the same for all dates, that means using future data!!! need to be modified
        if self.categoryField == 'area':
            tmp_state = "SELECT `%s`, `%s`  FROM `%s`;" % (self.codeField, self.categoryField, self.sourceCategoryTableName)
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
            # read stock return data
            tmp_state = self.stockState + "`%s` in (%s)" % (self.dateField, tmp_range)
            dataStockO = readDB(tmp_state, ConfigQuant)
            # read index return data
            tmp_state = self.indexState + "`%s` in (%s)" % (self.dateField, tmp_range)
            dataIndexO = readDB(tmp_state, ConfigQuant)

            dataStockO = dataStockO.drop_duplicates([self.dateField, self.codeField])
            dataIndexO = dataIndexO.drop_duplicates([self.dateField, self.categoryField])
            # dataStockO = dataStockO.fillna(0)
            # dataIndexO = dataIndexO.fillna(0)

            data_result = self.getStockCategory(dataStockO, dataIndexO, tmp_range, stock_area)
            new_fields = list(map(lambda x: 'EXCESSIVE_%s' % x, self.retSeriesField))
            for field in zip(new_fields, self.retSeriesField):
                data_result[field[0]] = data_result[field[1]] - data_result[field[1]+'_index']

            tot_fields = [self.dateField, self.codeField]
            tot_fields.extend(new_fields)
            data_result = data_result[tot_fields]

            # add timestamp
            print(self.targetTableName)
            print(data_result.shape)
            data_result.loc[:, 'time_stamp'] = datetime.now()

            # dump chunk data into sql
            writeDB(self.targetTableName, data_result, ConfigQuant, self.if_exist)
            self.if_exist = 'append'

    def getStockCategory(self, stock_data, index_data, date_range, stock_area):
        if self.categoryField == 'industry':
            tmp_state = 'select `%s`, `%s`, `%s` from %s where %s in (%s)' % (self.dateField, self.codeField,
                                                                              self.categoryField, self.sourceCategoryTableName,
                                                                              self.dateField, date_range)
            category = readDB(tmp_state, ConfigQuant)
            stock_data = stock_data.merge(category, on=[self.dateField, self.codeField])
            stock_data = stock_data.merge(index_data, on=[self.dateField, self.categoryField],
                                         suffixes=['', '_index'])
        elif self.categoryField == 'area':
            stock_data = stock_data.merge(stock_area, on=self.codeField)
            stock_data = stock_data.merge(index_data, on=[self.dateField, self.categoryField],
                                         suffixes=['','_index'])
        elif self.categoryField == 'market':
            stock_data[self.categoryField] = stock_data[self.codeField].apply(lambda x: 'SH' if x[:2] == '60' else (
                'SMEB' if x[:3] == '002' else (
                    'GEB' if x[:3] == '300' else 'SZ')))
            stock_data = stock_data.merge(index_data, on=[self.dateField, self.categoryField], suffixes=['', '_index'])
        elif self.categoryField == 'HS300':
            stock_data = stock_data.merge(index_data, on=self.dateField, suffixes=['', '_index'])

        return stock_data