import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import django
from energy.models import ElectricityReading


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'energy_monitor.settings')
django.setup()
def train_and_save_models():
    print("Starting model training process...")
    DB_NAME = 'energy'
    DB_USER = 'root'
    DB_PASSWORD = 'Charles01.' 
    DB_HOST = '127.0.0.1'
    DB_PORT = '3306'
    try:
        engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        df = pd.read_sql('SELECT id, timestamp, device_id, voltage, current, room FROM energy_electricityreading ORDER BY timestamp;', engine)
        engine.dispose() 
        print(f"Successfully loaded {len(df)} records from the database.")
    except Exception as e:
        print(f"Error connecting to database or loading data: {e}")
        print("Please ensure your database is running and credentials in Ai_models/trainer.py are correct.")
        return

    if df.empty:
        print("No data found .")
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
    print("Training Voltage Prediction Model...")
    voltage_model = RandomForestRegressor(n_estimators=100, random_state=42)
    voltage_model.fit(X, y_voltage)
    voltage_predictions = voltage_model.predict(X)
    print(f"Voltage Model R2 Score: {r2_score(y_voltage, voltage_predictions):.2f}")
    print(f"Voltage Model MSE: {mean_squared_error(y_voltage, voltage_predictions):.2f}")
    print("Training Current Prediction Model...")
    current_model = RandomForestRegressor(n_estimators=100, random_state=42)

    current_model.fit(X, y_current)
    current_predictions = current_model.predict(X)
    print(f"Current Model R2 Score: {r2_score(y_current, current_predictions):.2f}")
    print(f"Current Model MSE: {mean_squared_error(y_current, current_predictions):.2f}")
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True) 

    joblib.dump(voltage_model, os.path.join(model_dir, 'voltage_model.joblib'))
    joblib.dump(current_model, os.path.join(model_dir, 'current_model.joblib'))
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.joblib'))
    print(f"Models and feature columns saved to: {model_dir}")
    print("Model training complete.")

if __name__ == '__main__':
    train_and_save_models()

