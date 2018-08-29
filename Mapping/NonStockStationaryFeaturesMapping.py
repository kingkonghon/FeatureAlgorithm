# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicatorTimeSeries, amountTechnicalIndicators


class NonStockStationaryFeaturesMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.dateField = kwargs.get("dateField")
        self.closeField = kwargs.get("closeField")
        self.amountField = kwargs.get('amountField')
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.alignFlag = kwargs.get('alignFlag')
        self.condition = kwargs.get('condition')
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.dateField, self.lags, self.condition, self.alignFlag)

        # sql statement
        fields = ",".join([self.dateField, self.openField, self.highField, self.lowField, self.closeField, self.volumeField])
        if is_full == 1:  # 全量
            self.state = "SELECT %s FROM %s" % (fields, self.sourceTableName)
            if self.condition != '':
                self.state = self.state + " where " + self.condition # add search condition
            self.if_exist = 'replace'
        elif is_full == 0:  # 增量
            self.last_update_date = last_record_date
            self.state = "SELECT %s FROM %s where `%s` > '%s'" % (
                fields, self.sourceTableName, self.dateField, start_fetch_date)
            if self.condition != '':
                self.state = self.state + " and " + self.condition # add search condition
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
        dataO = dataO.sort_values(self.dateField) # sort by date
        dataO = dataO.fillna(method='ffill')
        # for field in [self.openField, self.highField, self.lowField, self.closeField, self.volumeField]:
        #     dataO.loc[:, field] = dataO[field].astype('float')

        data_result = pd.DataFrame(dataO[self.dateField])

        # MA, MA_DIFF...  (lags >= 2)
        tmp_df = priceTechnicalIndicatorTimeSeries(dataO[self.closeField], self.lags, '')
        data_result = data_result.join(tmp_df)

        # AMT_MA ... AMT_OBV, AMOUNT_RATIO...
        tmp_df = amountTechnicalIndicators(dataO[self.closeField], dataO[self.amountField], self.lags, 'AMT')
        data_result = data_result.join(tmp_df)

        # trim data for increment
        if self.last_update_date != '':
            data_result = data_result.loc[data_result[self.dateField] > self.last_update_date]

        # add timestamp
        data_result['time_stamp'] = datetime.now()

        # dump data into sql
        writeDB(self.targetTableName, data_result, ConfigQuant, self.if_exist)
