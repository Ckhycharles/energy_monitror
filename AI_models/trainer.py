
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from energy.models import ElectricityReading
import joblib
import os
import django
import sys
import pymysql
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'energy_monitor.settings')
django.setup()


def train_and_save_models():
    DB_NAME = 'energy'
    DB_USER = 'root'
    DB_PASSWORD = 'Charles01.' 
    DB_HOST = '127.0.0.1'
    DB_PORT = '3306'

    try:
        engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        df = pd.read_sql('SELECT id, timestamp, device_id, voltage, current, room FROM energy_electricityreading ORDER BY timestamp;', engine)
        engine.dispose()
        print(f"Successfully loaded {len(df)} records ")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("No data found in ElectricityReading")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month

    df = pd.get_dummies(df, columns=['device_id', 'room'], drop_first=True)

    feature_columns = ['hour', 'day_of_week', 'day_of_year', 'month'] + \
                      [col for col in df.columns if col.startswith('device_id_') or col.startswith('room_')]


    X = df[feature_columns]
    y_voltage = df['voltage']
    y_current = df['current']
    print("Training Voltage ..")
    voltage_model = RandomForestRegressor(n_estimators=100, random_state=42)
    voltage_model.fit(X, y_voltage)
    voltage_predictions = voltage_model.predict(X)
    print(f"Voltage R2 Score: {r2_score(y_voltage, voltage_predictions):.2f}")
    print(f"Voltage MSE: {mean_squared_error(y_voltage, voltage_predictions):.2f}")

    print("Training Current ..")
    current_model = RandomForestRegressor(n_estimators=100, random_state=42)
    current_model.fit(X, y_current)
    current_predictions = current_model.predict(X)
    print(f"Current R2 Score: {r2_score(y_current, current_predictions):.2f}")
    print(f"Current MSE: {mean_squared_error(y_current, current_predictions):.2f}")

    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(voltage_model, os.path.join(model_dir, 'voltage_model.joblib'))
    joblib.dump(current_model, os.path.join(model_dir, 'current_model.joblib'))
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.joblib'))

    print(f"Models saved to: {model_dir}")
    print("training complete")


def predict_voltage_current(input_dict):
    import numpy as np
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    voltage_model = joblib.load(os.path.join(model_dir, 'voltage_model.joblib'))
    current_model = joblib.load(os.path.join(model_dir, 'current_model.joblib'))
    feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.joblib'))
    X_input = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in X_input.columns:
            X_input[col] = 0 
    X_input = X_input[feature_columns]

    voltage_pred = voltage_model.predict(X_input)[0]
    current_pred = current_model.predict(X_input)[0]
    return voltage_pred, current_pred

if __name__ == '__main__':
    train_and_save_models()
