import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Mapping.StockTechnicalIndicatorMapping import StockTechnicalIndicatorMapping
from Mapping.ExcessReturnMapping import ExcessReturnMapping
from Mapping.StockOtherFeaturesMapping import StockOtherFeaturesMapping
from Mapping.StockStationaryTechIndicatorMapping import StockStationaryTechnicalIndicatorMapping
from Mapping.StockStationaryFundamentalMapping import StockStationaryFundamentalIndicatorMapping
from Mapping.StockFundamentalTushareMapping import StockFundamentalTushareMapping
import time

ConTech = {
    'sourceTableName': 'STOCK_FORWARD_ADJ_QUOTE',
    'codeField': 'code',
    'dateField': 'date',
    'openField': 'open',
    'highField': 'high',
    'lowField': 'low',
    'closeField': 'close',
    'volumeField': 'volume',
    'turnoverField': 'turnover',
    'lags': [1,2,5,10,20,30,60],
    'targetTableName': 'DERI_STOCK_TECH_INDICATORS',
    'condition': '',
    'chunkSize': 50,
    'isMultiProcess': True,
    'processNum': 4
}

ConExc = [
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_AREA_INDEX_FOR_ADJ',
        'sourceCategoryTableName': 'STOCK_DESCRIPTION',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'area',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,  # not every stock has area (merge)
        'targetTableName': 'DERI_AREA_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_INDUSTRY_INDEX_FOR_ADJ',
        'sourceCategoryTableName': 'STOCK_INDUSTRY',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'industry',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': False,  # not every stock has industry (merge)
        'targetTableName': 'DERI_INDUSTRY_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_MARKET_INDEX',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'market',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_MARKET_EXCESSIVE_RET',
        'chunkSize': 10
    },
    {
        'sourceStockTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceIndexTableName': 'DERI_HS300',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'categoryField': 'HS300',
        'retSeriesField': ['RET'],
        'lags': [1, 2, 5, 10, 20, 30, 60],
        # 'alignFlag': False,
        'targetTableName': 'DERI_HS300_EXCESSIVE_RET',
        'chunkSize': 10
    }
]

ConOthers = [
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_DESCRIPTION',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'area',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': True,   # not every stock has area field, but weekend there is update problem
        'targetTableName': 'DERI_STOCK_RANKING_AREA',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': 'STOCK_INDUSTRY',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'industry',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': True,   # not every stock has industry field
        'targetTableName': 'DERI_STOCK_RANKING_INDUSTRY',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'market',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': True,
        'targetTableName': 'DERI_STOCK_RANKING_MARKET',
        'chunkSize': 10
    },
    {
        'sourceBasicTableName': 'STOCK_FUNDAMENTAL_TTM',
        'sourceTechnicalTableName': 'DERI_STOCK_TECH_INDICATORS',
        'sourceCategoryTableName': '',
        'codeField': 'code',
        'dateField': 'date',
        'basicSeriesField': ['PE_TTM'],
        'techSeriesFieldWithLag': ['RET', 'VRET', 'TURNOVER_ROLLING_SUM', 'AMPLITUDE'],
        'techSeriesFieldWithoutLag': ['VOLUME_RATIO'],
        'reverseSeriesField': ['PE_TTM'],
        'categoryField': 'all',
        'lags': [1, 2, 5, 10, 20, 30, 60],
        'alignFlag': True,
        'targetTableName': 'DERI_STOCK_RANKING_ALL_STOCKS',
        'chunkSize': 10
    },
]

ConTechStationary = {
    'sourceTableName': 'STOCK_FORWARD_ADJ_QUOTE',
    'codeField': 'code',
    'dateField': 'date',
    'openField': 'open',
    'highField': 'high',
    'lowField': 'low',
    'closeField': 'close',
    'volumeField': 'volume',
    'amountField': 'amount',
    'turnoverField': 'turnover',
    'lags': [1,2,5,10,20,30,60,120,250],
    'targetTableName': 'DERI_STOCK_STAT_TECH_INDICATORS',
    'condition': '',
    'chunkSize': 50,
    'isMultiProcess': True,
    'processNum': 4
}

ConFunStationary = {
    'sourceTableName': 'STOCK_FUNDAMENTAL_BASIC',
    'calendarTableName': 'TRADE_CALENDAR',
    'codeField': 'code',
    'dateField': 'date',
    'valueFields': ['TOT_MRK_CAP', 'FREE_MRK_CAP'],
    'lags': [2,5,10,20,30,60,120,250],
    'targetTableName': 'DERI_STOCK_STAT_FUNDAMENTAL_INDICATORS',
    'condition': '',
    'chunkSize': 50,
    'isMultiProcess': True,
    'processNum': 3
}

ConFunTushare = {
    'sourceTableName': 'STOCK_FUNDAMENTAL_TUSHARE',
    'sourceMrkCapTableName': 'STOCK_FUNDAMENTAL_BASIC',
    'sourceStockQuoteTableName': 'STOCK_UNADJUSTED_QUOTE',
    'calendarTableName': 'TRADE_CALENDAR',
    'codeField': 'code',
    'dateField': 'date',
    'yearField': 'fiscal_year',
    'seasonField': 'fiscal_season',
    'releaseDateField': 'report_date',
    'splitDividendField': 'split_and_dividend',
    'valueForYOYFields': ['parent_net_profits', 'eps', 'bvps', 'revenue', 'sales_per_share', 'cfo_per_share', 'acc_rece_turover',
                          'inventory_turnover', 'currentasset_turnover', 'roe_weighted', 'roe_diluted'],
    'valueForTTMFields': ['eps', 'cfo_per_share', 'parent_net_profits', 'sales_per_share', 'dividend'],
    'originalYOYFields': ['revenue_yoy', 'parent_net_profits_yoy', 'consolidated_equity_yoy', 'asset_yoy', 'eps_yoy', 'parent_equity_yoy'],
    'ratioFields': ['current_ratio', 'quick_ratio', 'cash_ratio', 'interest_coverage', 'asset_ratio', 'cfo_to_revenue_ratio',
                    'cfo_to_asset_ratio', 'cfo_liab_ratio', 'parent_net_profit_margin', 'gross_profit_margin'],
    'timeStampField': 'time_stamp',
    'targetTableName': 'STOCK_FUNDAMENTAL_DAILY_TUSHARE',
    'condition': '',
    'chunkSize': 50,
    'isMultiProcess': True,
    'processNum': 2
}


# afternoon tasks
def airflowCallableStockTechnicalInicators():
    features = StockTechnicalIndicatorMapping(**ConTech)
    features.run()

def airflowCallableStockStationaryTechnicalIndicators():
    features = StockStationaryTechnicalIndicatorMapping(**ConTechStationary)
    features.run()

def airflowCallableStockStationaryFundamentalIndicators():
    features = StockStationaryFundamentalIndicatorMapping(**ConFunStationary)
    features.run()

def airflowCallableStockFundamentalTushare():
    features = StockFundamentalTushareMapping(**ConFunTushare)
    features.run()

def airflowCallableStockExcessiveReturnArea():
    con  = ConExc[0]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnIndustry():
    con  = ConExc[1]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnMarket():
    con  = ConExc[2]
    features = ExcessReturnMapping(**con)
    features.run()

def airflowCallableStockExcessiveReturnHS300():
    con  = ConExc[3]
    features = ExcessReturnMapping(**con)
    features.run()

# evening task
def airflowCallableRankingArea():
    con = ConOthers[0]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingIndustry():
    con = ConOthers[1]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingMarket():
    con = ConOthers[2]
    features = StockOtherFeaturesMapping(**con)
    features.run()

def airflowCallableRankingAllStocks():
    con = ConOthers[3]
    features = StockOtherFeaturesMapping(**con)
    features.run()

if __name__ == '__main__':
    start_time = time.clock()

    # features = StockTechnicalIndicatorMapping(**ConTech)
    # features.run()

    # for i in ConExc:
    #     features = ExcessReturnMapping(**i)
    #     features.run()
    #
    #
    # for i in ConOthers:
    #     features = StockOtherFeaturesMapping(**i)
    #     features.run()
    # airflowCallableRankingAllStocks()

    # features = StockStationaryTechnicalIndicatorMapping(**ConTechStationary)
    # features.run()

    features = StockStationaryFundamentalIndicatorMapping(**ConFunStationary)
    features.run()

    # features = StockFundamentalTushareMapping(**ConFunTushare)
    # features.run()

    eclapsed = time.clock() - start_time
    print("Time eclapsed", eclapsed)