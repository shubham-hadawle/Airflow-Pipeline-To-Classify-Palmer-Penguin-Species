FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /tmp/penguin-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/penguin-requirements.txt

COPY --chown=airflow:root penguin_lab /opt/airflow/penguin_lab
COPY --chown=airflow:root dataset /opt/airflow/dataset
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root tests /opt/airflow/tests
ENV PYTHONPATH=/opt/airflow
