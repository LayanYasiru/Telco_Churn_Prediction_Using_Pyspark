"""
src/orchestration/dags/retrain_dag.py
──────────────────────────────────────
Airflow DAG for retraining the Telco Churn Prediction model.
Scheduled hourly with concurrency controls to prevent SQLite locks.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "telco_churn_retrain_dag",
    default_args=default_args,
    description="Continuous retraining pipeline for Telco Churn prediction",
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # Critical to avoid SQLite write locks
    tags=["mlops", "telco_churn"],
) as dag:

    # Task 1: Run Data Pipeline (Ingestion, Imputation, Scaling, Splitting)
    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command="python -m src.orchestration.data_pipeline --config config.yaml",
        cwd="/opt/airflow",
    )

    # Task 2: Run Training Pipeline (Model Training, Evaluation, Champion Election)
    run_training_pipeline = BashOperator(
        task_id="run_training_pipeline",
        bash_command="python -m src.orchestration.training_pipeline --config config.yaml",
        cwd="/opt/airflow",
    )

    # Set dependencies
    run_data_pipeline >> run_training_pipeline
