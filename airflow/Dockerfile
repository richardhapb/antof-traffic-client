FROM apache/airflow:slim-2.10.3-python3.9 AS airflow

USER root

RUN apt-get update

USER airflow

ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow/"
COPY requirements.txt .
RUN pip install --user -r requirements.txt

COPY . .

