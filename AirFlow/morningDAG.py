from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from DataUpdate.updateDataUSIndex import airflowCallable as updateDataUSIndex
from DataUpdate.updateDataStockIndustry import airflowCallable as updateDataStockIndustry

from DataUpdate.updateTradeDateList import airflowCallable as updateTradeDateList
from DataUpdate.updateDataCrudeOil import airflowCallable as updateDataCrudeOil


from Derivatives.calMacroEconomiesFeatures import airflowCaCallableUSStockIndex as calUSStockIndexFeatures
from Derivatives.calMacroEconomiesFeatures import airflowCallableOil as calCrudeOilFeatures

from DataUpdate.updateDataHS300Weights import airflowCallable as updateHS300Weights


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

# ======= dag =======
# UTC time
dag = DAG('morning_dag', catchup=False, schedule_interval='40 23 * * *', default_args=default_args)

# ========= task ==========
# update trade calendar
t_tradedate = PythonOperator(
    task_id='data_tradedate',
    python_callable=updateTradeDateList,
    dag=dag)

# ========= boundry 1 =========== (fetch data)
data_boundary = DummyOperator(task_id='data_boundary_morning', dag=dag)
data_boundary.set_upstream([t_tradedate])

# update data
t_us_index_quote = PythonOperator(
    task_id='data_us_index_quote',
    python_callable=updateDataUSIndex,
    dag=dag)


t_stock_industry = PythonOperator(
    task_id='data_stock_industry',
    python_callable=updateDataStockIndustry,
    dag=dag)

t_crude_oil = PythonOperator(
    task_id='data_crude_oil',
    python_callable=updateDataCrudeOil,
    dag=dag)

t_hs300_weights = PythonOperator(
    task_id='data_hs300_weights',
    python_callable=updateHS300Weights,
    dag=dag)

data_boundary.set_downstream([t_us_index_quote, t_stock_industry, t_crude_oil, t_hs300_weights])

#  ======== boundary 2 =========== algorithm
data_boundary2 = DummyOperator(task_id='data_boundary_morning_2', dag=dag)
data_boundary2.set_upstream([t_us_index_quote, t_stock_industry, t_crude_oil])

# calculate features
t_US_stock_index_technical = PythonOperator(
    task_id='feature_US_stock_index',
    python_callable=calUSStockIndexFeatures,
    dag=dag)

t_oil_technical = PythonOperator(
    task_id='feature_curde_oil',
    python_callable=calCrudeOilFeatures,
    dag=dag)

data_boundary2.set_downstream([t_US_stock_index_technical, t_oil_technical])