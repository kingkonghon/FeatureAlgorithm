from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# from DataUpdate.updateTradeDateList import airflowCallable as updateTradeDateList
from DataUpdate.updateDataSHIBOR import airflowCallable as updateDataSHIBOR
from Derivatives.calMacroEconomiesFeatures import airflowCallableShibor as calShiborFeatures

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

# ========== dag =========
# UTC time
dag = DAG('midday_dag', catchup=False, schedule_interval='30 4 * * *', default_args=default_args)

# ========== task ==========
# update data tasks
t_shibor = PythonOperator(
    task_id='data_shibor',
    python_callable=updateDataSHIBOR,
    dag=dag)

# t_tradedate = PythonOperator(
#     task_id='data_tradedate',
#     python_callable=updateTradeDateList,
#     dag=dag)

# boundary
data_boundary = DummyOperator(task_id='data_boundary_midday', dag=dag)
data_boundary.set_upstream([t_shibor])

# calculate features
t_feature_shibor = PythonOperator(
    task_id='feature_shibor',
    python_callable=calShiborFeatures,
    dag=dag)

data_boundary.set_downstream(t_feature_shibor)