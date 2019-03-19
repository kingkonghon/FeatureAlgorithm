from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from CheckData.checkDataUpdateStatusMorning import airflowCallable as checkDataUpdateStatusMorning


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
dag = DAG('check_morning_dag', catchup=False, schedule_interval='50 7 * * *', default_args=default_args)

# ========= task ==========
# update data
t_check_morning = PythonOperator(
    task_id='check_data_update_afternoon',
    python_callable=checkDataUpdateStatusMorning,
    dag=dag)
