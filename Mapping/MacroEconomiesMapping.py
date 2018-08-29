# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from Utils.DB_config import ConfigQuant
from Utils.DBOperation import readDB, writeDB, checkIfIncre
from Utils.Algorithms import priceTechnicalIndicator


class MacroEconomiesMapping:
    def __init__(self, **kwargs):
        self.sourceTableName = kwargs.get("sourceTableName")
        self.dateField = kwargs.get("dateField")
        self.valueField = kwargs.get("valueField")
        self.lags = kwargs.get('lags')
        self.targetTableName = kwargs.get('targetTableName')
        self.state = ''
        self.if_exist = ''
        self.last_update_date = ''

    # prepare sql statement (全量 or 增量)
    def prepareData(self, startDate='2007-01-01'):
        # check if target table exist
        is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, self.sourceTableName,
                                                                   self.targetTableName, self.dateField, self.lags, '')

        # sql statement
        fields = self.dateField + ',' + "`%s`" % self.valueField
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
        dataO = dataO.sort_values(self.dateField)

        data_result = pd.DataFrame(dataO[self.dateField])
        tmp_df = priceTechnicalIndicator(dataO[self.valueField], self.lags, '') # MA, MA_DIFF...
        data_result = data_result.join(tmp_df)

        # trim data for increment
        if self.last_update_date != '':
            data_result = data_result.loc[data_result[self.dateField] > self.last_update_date]

        # add timestamp
        data_result['time_stamp'] = datetime.now()

        # dump data into sql
        writeDB(self.targetTableName, data_result, ConfigQuant, self.if_exist)
        # self.if_exist = 'append'
