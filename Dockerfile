# Dockerfile for Apache Airflow Standalone
# Enriched with Telco Churn Prediction python dependencies & Kafka-python-ng client

FROM apache/airflow:2.7.2

USER root

# Install system dependencies if any are needed (none for our pure python codebase, but good to have)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy local dependencies manifest
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install kafka-python-ng client compatible with python 3.11/3.12/3.13
RUN pip install --no-cache-dir kafka-python-ng
