import logging
import os
import dill
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.cloud import bigquery

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    filename='predict.log',
                    filemode="w")

path = os.environ.get("PROJECT_PATH", "/home/albe_vip")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{path}/credentials/deep-ethos-436810-c7-32b7bed8002f.json"
model_dir = f"{path}/model_directory"


# Загрузка модели, энкодера и скалера
def load_model():
    model_files = sorted([file for file in os.listdir(model_dir) if file.endswith('.pkl') and 'model' in file])
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    latest_model_file = model_files[-1]
    latest_model_path = os.path.join(model_dir, latest_model_file)
    logging.info(f"Loading model from {latest_model_path}")
    with open(latest_model_path, 'rb') as file:
        model = dill.load(file)
    logging.info(f"Model loaded: {type(model)} - {latest_model_path}")

    return model


model = load_model()

encoder_path = os.path.join(model_dir, "encoder.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

encoder = joblib.load(encoder_path)
logging.info(f"Encoder loaded: {type(encoder)}")
scaler = joblib.load(scaler_path)
logging.info(f"Scaler loaded: {type(scaler)}")


# Функция для обработки данных
def process_data(query, encoder, scaler):
    client = bigquery.Client()
    data = client.query(query).to_dataframe()
    session_client_data = data[['session_id', 'client_id']].copy()
    if 'conversion_rate' in data.columns:
        data = data.drop(['session_id', 'client_id', 'conversion_rate'], axis=1, errors='ignore')
    else:
        data = data.drop(['session_id', 'client_id'], axis=1)

    if 'visit_number' in data.columns:
        data['visit_number'] = data['visit_number'].astype('int64')

    string_columns = data.select_dtypes(include=['object']).columns
    encoded_strings = encoder.transform(data[string_columns])
    data_numeric = data.drop(string_columns, axis=1)

    data_encoded = hstack([data_numeric, encoded_strings])
    data_encoded = csr_matrix(data_encoded)

    # Применение скалера
    data_encoded = scaler.transform(data_encoded)

    return data_encoded, session_client_data


# Функция предсказания
def predict():
    logging.info(f"Обработка файла: {file_path}")

    # SQL запрос для загрузки данных из BigQuery
    query = "SELECT * FROM `deep-ethos-436810-c7.my_dataset.modified`"

    X, session_client_data = process_data(query, encoder, scaler)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    # Добавляем колонки с вероятностями и классификацией
    client = bigquery.Client()
    data = client.query(query).to_dataframe()
    data = data.drop('conversion_rate', axis=1, errors='ignore')

    # Условия классификации
    conditions = [
        (probabilities >= 0.70) & (predictions == 1),  # True Positive (TP)
        (probabilities >= 0.20) & (probabilities < 0.70) & (predictions == 0),  # False Positive (FP)
        (probabilities < 0.20) & (predictions == 0),  # True Negative (TN)
        (probabilities >= 0.20) & (probabilities < 0.70) & (predictions == 1)  # False Negative (FN)
    ]
    choices = ['TP', 'FP', 'TN', 'FN']

    data['predicted_class'] = np.select(conditions, choices, default='Unknown')

    # Восстанавливаем session_id и client_id
    data['session_id'] = session_client_data['session_id']
    data['client_id'] = session_client_data['client_id']

    # Сохранение результатов обратно в BigQuery
    table_id = "deep-ethos-436810-c7.my_dataset.session-predictions"
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    job = client.load_table_from_dataframe(data, table_id, job_config=job_config)
    job.result()  # Ожидание завершения загрузки

    logging.info("Предсказания сохранены в BigQuery")


if __name__ == "__main__":
    predict()

