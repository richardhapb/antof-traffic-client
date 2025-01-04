import requests

import datetime
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

APP_URL = "http://host.docker.internal:8050"
local_tz = pendulum.timezone("America/Santiago")


def update_data():
    response = requests.get(APP_URL + "/update_data", timeout=10)
    print(response.text)


def fetch_data():
    response = requests.get(APP_URL + "/fetch_data", timeout=10)
    print(response.text)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["richard.hapb@icloud.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": datetime.timedelta(minutes=5),
}

with DAG(
    dag_id="update_data",
    description="Retrieve info from Waze API and put in app",
    default_args=default_args,
    start_date=pendulum.datetime(2024, 1, 1, tz=local_tz),
    schedule="*/5 * * * *",
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=2),
) as dag:

    task_fetch = PythonOperator(
        task_id="fetch_data_from_api", python_callable=fetch_data
    )

    task_put = PythonOperator(
        task_id="update_app_data",
        python_callable=update_data,
    )

    task_fetch >> task_put


def train_model():
    response = requests.get(APP_URL + "/train", timeout=20)
    print(response.text)


def update_model():
    response = requests.get(APP_URL + "/update_model", timeout=10)
    print(response.text)


with DAG(
    dag_id="update_model",
    description="Update ML Model and update in the app",
    default_args=default_args,
    start_date=pendulum.datetime(2024, 1, 1, tz=local_tz),
    schedule="0 0 1,15 * *",
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=10),
) as dag:

    task_train = PythonOperator(task_id="train_model", python_callable=train_model)

    task_update_model = PythonOperator(
        task_id="update_model_in_app",
        python_callable=update_model,
    )

    task_train >> task_update_model
