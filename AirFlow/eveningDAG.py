from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from DataUpdate.updateDataStockTTM import airflowCallable as updateDataStockTTM
from DataUpdate.updateDataPreciousMetal import airflowCallable as updateDataPreciousMetal

from Derivatives.calNonStockFeatures import airflowCallablePreciousMetal as calPreciousMetalFeatures
from Derivatives.calStockFeatures import airflowCallableRankingArea as calStockFeatureRankingArea, \
    airflowCallableRankingIndustry as calStockFeatureRankingIndustry, \
    airflowCallableRankingMarket as calStockFeatureRankingMarket, \
    airflowCallableRankingAllStocks as calStockFeatureRankingAllStocks


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 1, 1),
    'email': ['jianghan@nuoyuan.com.cn'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

# UTC time
dag = DAG('evening_dag', catchup=False, schedule_interval='0 13 * * *', default_args=default_args)

# t1, t2 and t3 are examples of tasks created by instantiating operators
t_stock_ttm = PythonOperator(
    task_id='data_stock_ttm',
    python_callable=updateDataStockTTM,
    dag=dag)

t_precious_metal = PythonOperator(
    task_id='data_precious_metal',
    python_callable=updateDataPreciousMetal,
    dag=dag)

# boundary data === algorithm
data_boundary = DummyOperator(task_id='data_boundary_evening', dag=dag)
data_boundary.set_upstream([t_stock_ttm, t_precious_metal])

t_presious_metal_technical = PythonOperator(
    task_id='feature_precious_metal',
    python_callable=calPreciousMetalFeatures,
    dag=dag)

t_stock_feature_ranking_area = PythonOperator(
    task_id='feature_stock_feature_ranking_in_area',
    python_callable=calStockFeatureRankingArea,
    dag=dag)

t_stock_feature_ranking_industry = PythonOperator(
    task_id='feature_stock_feature_ranking_in_industry',
    python_callable=calStockFeatureRankingIndustry,
    dag=dag)

t_stock_feature_ranking_market = PythonOperator(
    task_id='feature_stock_feature_ranking_in_market',
    python_callable=calStockFeatureRankingMarket,
    dag=dag)

t_stock_feature_ranking_all_stocks = PythonOperator(
    task_id='feature_stock_feature_ranking_in_all_stocks',
    python_callable=calStockFeatureRankingAllStocks,
    dag=dag)

data_boundary.set_downstream([t_presious_metal_technical, t_stock_feature_ranking_area, t_stock_feature_ranking_industry,
                              t_stock_feature_ranking_market, t_stock_feature_ranking_all_stocks])
# boundry 2, derivative data == algorithm on derivative data