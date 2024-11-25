import os
import sys
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator

# Установка переменной окружения и добавление пути
path = "/home/albe_vip"
os.environ["PROJECT_PATH"] = path
sys.path.insert(0, path)

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 11, 14),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
    'execution_timeout': dt.timedelta(minutes=60),
}

with DAG(
    dag_id='session_prediction',
    default_args=default_args,
    schedule_interval=None,  # Убедитесь, что это значение соответствует вашим потребностям
    catchup=False
) as dag:

    def modelbe_wrapper():
        from modules.modelbe import modelbe
        modelbe()

    def prediction_wrapper():
        from modules.prediction import prediction
        prediction()

    # Задача для обучения модели
    modelbe_task = PythonOperator(
        task_id='modelbe_task',
        python_callable=modelbe_wrapper,
    )

    # Задача для выполнения предсказаний
    prediction_task = PythonOperator(
        task_id='prediction_task',
        python_callable=prediction_wrapper,
    )

    # Установка зависимости между задачами
    modelbe_task >> prediction_task


