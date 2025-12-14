from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import pickle

# Define paths (Using container paths)
BASE_PATH = "/opt/airflow/dags"
DATA_PATH = os.path.join(BASE_PATH, "data.csv")
MODEL_PATH = os.path.join(BASE_PATH, "model.pkl")


def load_data_func():
    """Generates dummy data"""
    print("Generating dummy data...")
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
        'target': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")


def train_model_func():
    """Trains model"""
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Data file not found!")

    df = pd.read_csv(DATA_PATH)
    X = df[['feature1', 'feature2']]
    y = df['target']

    print("Training model...")
    model = LogisticRegression()
    model.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


def log_results_func():
    """Logs success"""
    if os.path.exists(MODEL_PATH):
        print(f"SUCCESS: Model found at {MODEL_PATH}")
    else:
        raise FileNotFoundError("Model was not saved!")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('mlops_train_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    t1 = PythonOperator(task_id='load_data', python_callable=load_data_func)
    t2 = PythonOperator(task_id='train_model', python_callable=train_model_func)
    t3 = PythonOperator(task_id='log_results', python_callable=log_results_func)

    t1 >> t2 >> t3