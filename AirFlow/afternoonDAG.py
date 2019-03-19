from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from DataUpdate.updateDataStockForAdjQuote import airflowCallable as updateDataStockForAdjQuote
from DataUpdate.updateDataStockDescription import airflowCallable as updateDataStockDescription
from DataUpdate.updateDataStockUnadjustedQuote import airflowCallable as updateDataStockUnadjustedQuote
from DataUpdate.updateDataStockIndex import airflowCallable as updateDataStockIndex
from DataUpdate.updateDataStockFundamental import airflowCallable as updateDataStockFundamental
from DataUpdate.updateDataFundIndex import airflowCallable as updateDataFundIndex
from DataUpdate.updateDataStockFlag import airflowCallable as updateDataStockFlag
from DataUpdate.updateDataTushareDailyQuotes import airflowCallable as updateDataStockTushareQuotes

from Basic.calForwardAdjAreaIndex import airflowCallable as calForwardAdjAreaIndex
from Basic.calForwardAdjIndustryIndex import airflowCallable as calForwardAdjIndustryIndex

from Derivatives.calStockFeatures import airflowCallableStockTechnicalInicators as calStockTechnicalInicators, \
    airflowCallableStockExcessiveReturnArea as calStockExcessiveReturnArea, \
    airflowCallableStockExcessiveReturnIndustry as calStockExcessiveReturnIndustry, \
    airflowCallableStockExcessiveReturnHS300 as calStockExcessiveReturnHS300, \
    airflowCallableStockExcessiveReturnMarket as calStockExcessiveReturnMarket, \
    airflowCallableStockStationaryTechnicalIndicators as calStockStationaryTechnicalIndicators, \
    airflowCallableStockStationaryFundamentalIndicators as calStockStationaryFundamentalIndicators
from Derivatives.calStockMarketFeatures import airflowCallableArea as calAreaRiseRatio, \
    airflowCallableIndustry as calIndustryRiseRatio, \
    airflowCallableMarket as calMarketRiseRatio, \
    airflowCallableAllStock as calAllStockRiseRatio
from Derivatives.calStockBeta import airflowCallable as calStockBeta
from Derivatives.calStockValue import airflowCallable as calStockValue
from Derivatives.calStockDayNum import airflowCallable as calStockDayNum
from Derivatives.calIndexFeatures import airflowCallableHS300Technical as calHS300TechnicalIndicator, \
    airflowCallableStockIndexTechnical as calStockIndexTechnicalIndicators, \
    airflowCallableStockIndexStationaryTechnical as calStockIndexStationaryTechnical, \
    airflowCallableHS300StationaryTechnical as calHS300StationaryTechnical
from Derivatives.calNonStockFeatures import airflowCallableFundIndex as calFundIndexFeatures

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 1, 1),
    'email': ['13602819622@163.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}
# ========= dag =========
# UTC time
dag = DAG('afternoon_dag', catchup=False, schedule_interval='35 18 * * *', default_args=default_args)

# ========= task ==========
# update data
t_stock_for_adj_quote = PythonOperator(
    task_id='data_stock_for_adj_quote',
    python_callable=updateDataStockForAdjQuote,
    dag=dag)

t_stock_unadj_quote = PythonOperator(
    task_id='data_stock_unadj_quote',
    python_callable=updateDataStockUnadjustedQuote,
    dag=dag)

t_stock_description = PythonOperator(
    task_id='data_stock_description',
    python_callable=updateDataStockDescription,
    dag=dag)

t_stock_ts_quotes = PythonOperator(
    task_id='data_stock_ts_quotes',
    python_callable=updateDataStockTushareQuotes,
    dag=dag)

t_stock_index = PythonOperator(
    task_id='data_stock_index',
    python_callable=updateDataStockIndex,
    dag=dag)

# =============== boundary ============= (update source data first, and later update final data from different sources)
data_boundary_data_source = DummyOperator(task_id='data_boundary_afternoon_data_source', dag=dag)
data_boundary_data_source.set_upstream([t_stock_ts_quotes])

t_fund_index = PythonOperator(
    task_id='data_fund_index',
    python_callable=updateDataFundIndex,
    dag=dag)

t_stock_fundamental = PythonOperator(
    task_id='data_stock_fundamental',
    python_callable=updateDataStockFundamental,
    dag=dag)

data_boundary_data_source.set_downstream([t_fund_index, t_stock_fundamental])

# =============== boundary ============= (dervative on stock quote and public index, composite customized index)
data_boundary = DummyOperator(task_id='data_boundary_afternoon', dag=dag)
data_boundary.set_upstream([t_stock_for_adj_quote, t_stock_unadj_quote, t_stock_description, t_stock_index, t_stock_fundamental, t_fund_index])

# calculate features
t_stock_technical = PythonOperator(
    task_id='feature_stock_technical',
    python_callable=calStockTechnicalInicators,
    dag=dag)

t_stock_stat_technical = PythonOperator(
    task_id='feature_stock_stat_technical',
    python_callable=calStockStationaryTechnicalIndicators,
    dag=dag)

t_stock_stat_fundamental = PythonOperator(
    task_id='feature_stock_stat_fundamental',
    python_callable=calStockStationaryFundamentalIndicators,
    dag=dag)

t_stock_beta = PythonOperator(
    task_id='feature_stock_beta',
    python_callable=calStockBeta,
    dag=dag)

t_stock_value = PythonOperator(
    task_id='feature_stock_value',
    python_callable=calStockValue,
    dag=dag)

t_stock_daynum = PythonOperator(
    task_id='feature_stock_daynum',
    python_callable=calStockDayNum,
    dag=dag)

t_HS300_technical = PythonOperator(
    task_id='feature_HS300_technical',
    python_callable=calHS300TechnicalIndicator,
    dag=dag)

t_HS300_stat_technical = PythonOperator(
    task_id='feature_HS300_stat_technical',
    python_callable=calHS300StationaryTechnical,
    dag=dag)

t_area_index_for_adj_quote = PythonOperator(
    task_id='data_area_index_for_adj_quote',
    python_callable=calForwardAdjAreaIndex,
    dag=dag)

t_industry_index_for_adj_quote = PythonOperator(
    task_id='data_industry_index_for_adj_quote',
    python_callable=calForwardAdjIndustryIndex,
    dag=dag)

t_fund_index_technical = PythonOperator(
    task_id='feature_fund_index',
    python_callable=calFundIndexFeatures,
    dag=dag)

t_stock_flag = PythonOperator(
    task_id='data_stock_flag',
    python_callable=updateDataStockFlag,
    dag=dag)

data_boundary.set_downstream([t_stock_technical, t_stock_stat_technical, t_stock_stat_fundamental,
                              t_stock_beta, t_stock_value, t_stock_daynum, t_HS300_technical, t_HS300_stat_technical,
                              t_area_index_for_adj_quote, t_industry_index_for_adj_quote, t_fund_index_technical, t_stock_flag])

# ================= boundry 2 ===============  (derivative on customized index quote)
data_boundary2 = DummyOperator(task_id='data_boundary_afternoon_2', dag=dag)
data_boundary2.set_upstream([t_area_index_for_adj_quote, t_industry_index_for_adj_quote])

t_stock_index_technical = PythonOperator(
    task_id='feature_stock_index_technical',
    python_callable=calStockIndexTechnicalIndicators,
    dag=dag)

t_stock_index_stat_technical = PythonOperator(
    task_id='feature_stock_index_stat_technical',
    python_callable=calStockIndexStationaryTechnical,
    dag=dag)

data_boundary2.set_downstream([t_stock_index_technical, t_stock_index_stat_technical])

# ============== boundry 3 ================== excessive return (stock derivatives over index derivatives)
data_boundary3 = DummyOperator(task_id='data_boundary_afternoon_3', dag=dag)
data_boundary3.set_upstream([t_stock_technical, t_stock_index_technical, t_HS300_technical])

t_exc_ret_area = PythonOperator(
    task_id='feature_stock_excessive_return_area',
    python_callable=calStockExcessiveReturnArea,
    dag=dag)

t_exc_ret_industry = PythonOperator(
    task_id='feature_stock_excessive_return_industry',
    python_callable=calStockExcessiveReturnIndustry,
    dag=dag)

t_exc_ret_hs300 = PythonOperator(
    task_id='feature_stock_excessive_return_HS300',
    python_callable=calStockExcessiveReturnHS300,
    dag=dag)

t_exc_ret_market = PythonOperator(
    task_id='feature_stock_excessive_return_market',
    python_callable=calStockExcessiveReturnMarket,
    dag=dag)

t_rise_ratio_area = PythonOperator(
    task_id='feature_stock_rise_ratio_area',
    python_callable=calAreaRiseRatio,
    dag=dag)

t_rise_ratio_industry = PythonOperator(
    task_id='feature_stock_rise_ratio_industry',
    python_callable=calIndustryRiseRatio,
    dag=dag)

t_rise_ratio_market = PythonOperator(
    task_id='feature_stock_rise_ratio_market',
    python_callable=calMarketRiseRatio,
    dag=dag)

t_rise_ratio_all_stock = PythonOperator(
    task_id='feature_stock_rise_ratio_all',
    python_callable=calAllStockRiseRatio,
    dag=dag)

data_boundary3.set_downstream([t_exc_ret_area, t_exc_ret_industry, t_exc_ret_market, t_exc_ret_hs300,
                               t_rise_ratio_area, t_rise_ratio_industry, t_rise_ratio_market, t_rise_ratio_all_stock])