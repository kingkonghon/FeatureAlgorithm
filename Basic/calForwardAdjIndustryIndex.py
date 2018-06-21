from sqlalchemy import create_engine
from datetime import datetime
from Utils.DBOperation import getTradeDates, getDataFromSQL, getIncrmDataFromSQL, writeDB, checkIfIncre
from Utils.DB_config import ConfigQuant
from Utils.Algorithms import calWeightedSumIndexQuote

ConfigIndustry = {
    'tableName':'STOCK_INDUSTRY',
    'code': 'code',
    'date': 'date',
    'industry': 'industry',
    'time_stamp': 'time_stamp'
}

ConQuote ={
    'tableName': 'STOCK_FORWARD_ADJ_QUOTE',
    'code': 'code',
    'date': 'date',
    'fields': ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'],
    'time_stamp': 'time_stamp'
}

ConWeights = {
    'tableName': 'STOCK_FUNDAMENTAL_BASIC',
    'code': 'code',
    'date': 'date',
    'weight': 'FREE_MRK_CAP',
    'time_stamp': 'time_stamp'
}

targetTableName = 'INDUSTRY_INDEX_FORWARD_ADJ_QUOTE'
targetDateField = 'date'
targetTimeStampField = 'time_stamp'

def calFullIndustry(db_config, con_industry, con_quote, con_weights, chunk_size, start_date = '2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    # get total trade dates
    trade_dates = getTradeDates(my_engine, con_quote['date'], con_quote['tableName'], start_date)

    # read and process data by trade dates
    quote_fields = list(map(lambda x: '`%s`' % x, con_quote['fields']))
    quote_fields = ','.join(quote_fields)
    write_sql_method = 'replace'
    for i in range(int(trade_dates.size / chunk_size) + 1):
        tmp_trade_dates = trade_dates[i*chunk_size : (i+1)*chunk_size]
        tmp_trade_dates = list(map(lambda x: "'%s'" % x, tmp_trade_dates))
        date_range = ','.join(tmp_trade_dates)

        # get quote data
        basic_data = getDataFromSQL(my_engine, con_quote['date'], con_quote['code'], quote_fields,
                                    con_quote['tableName'], date_range)
        # get weights
        weight_field = '`%s`' % con_weights['weight']
        weights = getDataFromSQL(my_engine, con_weights['date'], con_weights['code'], weight_field,
                                 con_weights['tableName'], date_range)
        weights = weights.rename(columns={con_weights['weight']: 'weight'})

        # get industry
        industry_field = '`%s`' % con_industry['industry']
        industry = getDataFromSQL(my_engine, con_industry['date'], con_weights['code'], industry_field,
                                  con_industry['tableName'], date_range)
        industry = industry.rename(columns={con_industry['industry']: 'industry'})
        # tot_ind = industry['industry'].unique()

        basic_data = basic_data.merge(weights, on=['date', 'code'], how='inner')
        basic_data = basic_data.merge(industry, on=['date', 'code'], how='inner')

        # calculate index quote
        industry_index_quote = calWeightedSumIndexQuote(basic_data, con_quote['fields'], 'date', 'industry', 'weight')

        # add timestamp
        industry_index_quote[targetTimeStampField] = datetime.now()

        # dump data to sql
        writeDB(targetTableName, industry_index_quote, db_config, write_sql_method)
        write_sql_method = 'append'


def calIncrmIndustry(db_config, con_industry, con_quote, con_weights, chunk_size, start_date = '2007-01-01'):
    # create sql engine
    my_engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{db}?charset={charset}'.format(**db_config))

    # incremtental to database
    write_sql_method = 'append'

    # get quote data (trim by timestamp)
    quote_fields = list(map(lambda x: '`%s`' % x, con_quote['fields']))
    quote_fields = ','.join(quote_fields)
    basic_data = getIncrmDataFromSQL(my_engine, con_quote['date'], con_quote['code'], quote_fields,
                                con_quote['tableName'], targetTableName, con_quote['date'], targetDateField)

    # already the latest data
    if basic_data.empty:
        return

    # get weights (trim by timestamp)
    weight_field = '`%s`' % con_weights['weight']
    weights = getIncrmDataFromSQL(my_engine, con_weights['date'], con_weights['code'], weight_field,
                             con_weights['tableName'], targetTableName, con_weights['date'], targetDateField)
    weights = weights.rename(columns={con_weights['weight']: 'weight'})

    # get industry (trim by timestamp)
    ind_field = '`%s`' % con_industry['industry']
    industry = getIncrmDataFromSQL(my_engine, con_industry['date'], con_weights['code'], ind_field,
                                  con_industry['tableName'], targetTableName, con_industry['date'], targetDateField)
    industry = industry.rename(columns={con_industry['industry']: 'industry'})

    basic_data = basic_data.merge(weights, on=['date', 'code'], how='inner')
    basic_data = basic_data.merge(industry, on=['date', 'code'], how='inner')

    # calculate index quote
    industry_index_quote = calWeightedSumIndexQuote(basic_data, con_quote['fields'], 'date', 'industry', 'weight')

    # add timestamp
    industry_index_quote[targetTimeStampField] = datetime.now()

    # dump data to sql
    writeDB(targetTableName, industry_index_quote, db_config, write_sql_method)

def airflowCallable():
    start_date = '2007-01-01'
    chunk_size = 10

    calIncrmIndustry(ConfigQuant, ConfigIndustry, ConQuote, ConWeights, chunk_size, start_date)

if __name__ == '__main__':
    start_date = '2007-01-01'
    chunk_size = 10

    is_full, last_record_date, start_fetch_date = checkIfIncre(ConfigQuant, ConQuote['tableName'],
                                                               targetTableName, ConQuote['date'], [0], '', False)

    # *********** sw index quote is avaliable, try to use them directly instead of sum by weights
    if is_full == 1:
        calFullIndustry(ConfigQuant, ConfigIndustry, ConQuote, ConWeights, chunk_size, start_date)
    elif is_full == 0:
        calIncrmIndustry(ConfigQuant, ConfigIndustry, ConQuote, ConWeights, chunk_size, start_date)
    else:
        pass