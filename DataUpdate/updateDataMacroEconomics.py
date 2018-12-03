import tushare as ts
import pandas as pd
import sys
import os
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Utils.DB_config import ConfigQuant


def getMoneySupply(ann_dt_label):
    raw_data = ts.get_money_supply()
    year_month_label = 'year_month'
    raw_data = raw_data.rename({'month': year_month_label}, axis=1)

    raw_data.loc[:, 'year'] = raw_data[year_month_label].apply(lambda x: x.split('.')[0]).astype('int')
    raw_data.loc[:, 'month'] = raw_data[year_month_label].apply(lambda x: x.split('.')[1]).astype('int')
    raw_data = raw_data.drop(year_month_label, axis=1)

    # ========= change datatype
    raw_data = chgDataType(raw_data, ['year', 'month'])

    # ========= change column names
    raw_data = raw_data.rename({'qm': 'quasi_currency'}, axis=1)

    # ========== announcement date
    tmp_data = raw_data.loc[raw_data['month'] != 12]  # announced month is 1 month later
    tmp_data.loc[:, 'announced_month'] = tmp_data['month'] + 1
    tmp_data.loc[:, 'announced_year'] = tmp_data['year']
    processed_data = tmp_data.copy()

    tmp_data = raw_data.loc[raw_data['month'] == 12]  # December, adding 1 month = January in the next year
    tmp_data.loc[:, 'announced_month'] = 1
    tmp_data.loc[:, 'announced_year'] = tmp_data['year'] + 1
    processed_data = processed_data.append(tmp_data)

    processed_data.loc[:, ann_dt_label] = processed_data.apply(
        lambda x:  datetime.strftime(datetime(int(x['announced_year']), int(x['announced_month']), 15), '%Y-%m-%d'),axis=1)
    processed_data = processed_data.drop(['announced_year', 'announced_month'], axis=1)

    return processed_data

def getGDP(ann_dt_label):
    raw_data =ts.get_gdp_quarter()
    year_quater_label = 'year_quarter'
    raw_data = raw_data.rename({'quarter': year_quater_label}, axis=1)

    raw_data.loc[:, 'year'] = raw_data[year_quater_label].astype('int')
    raw_data.loc[:, 'season'] = ((raw_data[year_quater_label] - raw_data['year']) * 10).apply(lambda x: round(x))
    raw_data = raw_data.drop(year_quater_label, axis=1)

    # ========= change datatype
    # raw_data = chgDataType(raw_data, ['year', 'season'])

    # ========= change column names
    raw_data = raw_data.rename({'pi': 'prime_industry', 'si': 'second_industry'}, axis=1)

    # ========== announcement date
    season_announcement_dict = {
        1: '-04-15',
        2: '-07-15',
        3: '-10-15',
        4: '-01-15',
    }

    processed_data = pd.DataFrame([])
    for tmp_season, tmp_ann_dt in season_announcement_dict.items():
        tmp_data = raw_data.loc[raw_data['season'] == tmp_season]
        tmp_data.loc[:, ann_dt_label] = tmp_data['year'].apply(lambda x: '%d%s' % (x, tmp_ann_dt))
        processed_data = processed_data.append(tmp_data)

    return processed_data

def getPPI(ann_dt_label):
    raw_data = ts.get_ppi()

    year_month_label = 'year_month'
    raw_data = raw_data.rename({'month': year_month_label}, axis=1)

    raw_data.loc[:, 'year'] = raw_data[year_month_label].apply(lambda x: x.split('.')[0]).astype('int')
    raw_data.loc[:, 'month'] = raw_data[year_month_label].apply(lambda x: x.split('.')[1]).astype('int')
    raw_data = raw_data.drop(year_month_label, axis=1)

    # ========= change datatype
    # raw_data = chgDataType(raw_data, ['year', 'month'])

    # ========== announcement date
    tmp_data = raw_data.loc[raw_data['month'] != 12]  # announced month is 1 month later
    tmp_data.loc[:, 'announced_month'] = tmp_data['month'] + 1
    tmp_data.loc[:, 'announced_year'] = tmp_data['year']
    processed_data = tmp_data.copy()

    tmp_data = raw_data.loc[raw_data['month'] == 12]  # December, adding 1 month = January in the next year
    tmp_data.loc[:, 'announced_month'] = 1
    tmp_data.loc[:, 'announced_year'] = tmp_data['year'] + 1
    processed_data = processed_data.append(tmp_data)

    processed_data.loc[:, ann_dt_label] = processed_data.apply(
        lambda x: datetime.strftime(datetime(int(x['announced_year']), int(x['announced_month']), 15), '%Y-%m-%d'), axis=1)
    processed_data = processed_data.drop(['announced_year', 'announced_month'], axis=1)

    return processed_data

def expandDataToDaily(raw_data, trade_calendar, ann_dt_label, start_date='2006-01-01'):
    raw_data = raw_data.sort_values(ann_dt_label, ascending=True)  # sort by announcement date
    raw_data =raw_data.loc[raw_data[ann_dt_label] >= start_date]  # trim data

    raw_data.loc[:, ann_dt_label] = pd.to_datetime(raw_data[ann_dt_label], format='%Y-%m-%d') # convert to datetime
    raw_data = raw_data.set_index(ann_dt_label)

    raw_data = raw_data.resample('D').pad()   # forward fill
    raw_data = raw_data.reset_index()
    raw_data.loc[:, ann_dt_label] = raw_data[ann_dt_label].dt.strftime('%Y-%m-%d')

    # fill out non-tradeday
    raw_data = raw_data.loc[raw_data[ann_dt_label].isin(trade_calendar)]

    # rename column
    raw_data = raw_data.rename({ann_dt_label: 'date'}, axis=1)

    return raw_data

def chgDataType(data, exclude_columns):
    tmp_columns = data.columns.tolist()
    tmp_columns = [x for x in tmp_columns if x not in exclude_columns]

    data = data.replace({'--': np.nan})

    for tmp_col in tmp_columns:
        data.loc[:, tmp_col] = data[tmp_col].astype('float')

    return data


def getEconomicDataFull():
    targetTableName = 'MACROECONOMIC_TUSHARE'
    calendarTableName = 'TRADE_CALENDAR'

    # ========== get trade calendar from db
    sql_conn = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))
    sql_statement = 'select `date` from %s' % calendarTableName
    trade_calendar = pd.read_sql(sql_statement, sql_conn)
    trade_calendar = trade_calendar['date']
    trade_calendar = trade_calendar[trade_calendar >= '2007-01-01']

    # ========== get data from tushare
    ann_dt_label = 'ann_dt'
    economic_data = getMoneySupply(ann_dt_label)
    economic_data = expandDataToDaily(economic_data, trade_calendar, ann_dt_label)

    tmp_data = getGDP(ann_dt_label)
    tmp_data = expandDataToDaily(tmp_data, trade_calendar, ann_dt_label)
    tmp_data = tmp_data.drop('year', axis=1)
    economic_data = economic_data.merge(tmp_data, on='date', how='outer')

    tmp_data = getPPI(ann_dt_label)
    tmp_data = expandDataToDaily(tmp_data, trade_calendar, ann_dt_label)
    tmp_data = tmp_data.drop(['year', 'month'], axis=1)
    economic_data = economic_data.merge(tmp_data, on='date', how='outer')

    # trim data
    economic_data = economic_data.loc[economic_data['date'] >= '2007-01-01']

    # order by date
    economic_data = economic_data.sort_values('date', ascending=True)

    # add timestamp
    economic_data.loc[:, 'time_stamp'] = datetime.now()

    # ========== dump data into db
    sql_conn = create_engine(
        'mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**ConfigQuant))  # reconnect db

    economic_data.to_sql(targetTableName, sql_conn, index=False, if_exists='replace')

def getEconomicDataIncrm():
    pass

if __name__ == '__main__':
    getEconomicDataFull()