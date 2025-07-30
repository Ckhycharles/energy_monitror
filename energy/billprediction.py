# Ai_models/trainer.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # More robust, consider using this for better accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib # For saving/loading models
import os
import django

# Configure Django settings to allow database access from this script
# This assumes 'energy_monitor' is your project's main settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'energy_monitor.settings')
django.setup()

# Import your model after Django setup
from energy.models import ElectricityReading

def train_and_save_models():
    print("Starting model training process...")

    # Database connection details (from your Django settings.py)
    # Ensure these match your actual database configuration
    DB_NAME = 'energy'
    DB_USER = 'root'
    DB_PASSWORD = 'Charles01.' # REPLACE WITH YOUR ACTUAL MYSQL ROOT PASSWORD
    DB_HOST = '127.0.0.1'
    DB_PORT = '3306'

    # Create a database engine
    # Use 'mysql+mysqlconnector://' if you installed mysql-connector-python
    # Use 'mysql+pymysql://' if you installed PyMySQL
    try:
        engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        # Load data directly into a Pandas DataFrame
        df = pd.read_sql('SELECT id, timestamp, device_id, voltage, current, room FROM energy_electricityreading ORDER BY timestamp;', engine)
        engine.dispose() # Close the connection
        print(f"Successfully loaded {len(df)} records from the database.")
    except Exception as e:
        print(f"Error connecting to database or loading data: {e}")
        print("Please ensure your database is running and credentials in Ai_models/trainer.py are correct.")
        return

    if df.empty:
        print("No data found in ElectricityReading table. Cannot train models.")
        return

    # --- Feature Engineering ---
    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    # You can add more features like:
    # df['minute'] = df['timestamp'].dt.minute
    # df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # df['time_of_day_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    # df['time_of_day_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Convert categorical features to numerical using one-hot encoding
    # This is important if device_id or room have predictive power
    df = pd.get_dummies(df, columns=['device_id', 'room'], drop_first=True)

    # Define features (X) and targets (y)
    # Use features that are available at prediction time (e.g., time components)
    # and potentially other stable features like device_id/room if they are fixed.
    # For future prediction, we can't use 'voltage' to predict 'current' or vice-versa,
    # unless we are predicting the *next* value in a time series.
    # For simplicity, we'll predict based on time features.

    # Identify all engineered features
    feature_columns = ['hour', 'day_of_week', 'day_of_year', 'month'] + \
                      [col for col in df.columns if col.startswith('device_id_') or col.startswith('room_')]

    # Ensure all feature columns exist, fill missing (e.g., if a dummy column isn't generated for all data)
    # This is crucial for consistent feature sets between training and prediction
    # We'll need to store the list of feature columns for prediction later.
    
    X = df[feature_columns]
    y_voltage = df['voltage']
    y_current = df['current']

    # --- Train Voltage Model ---
    print("Training Voltage Prediction Model...")
    # Using RandomForestRegressor for potentially better performance than LinearRegression
    voltage_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # For a simple start, you could use: voltage_model = LinearRegression()
    
    voltage_model.fit(X, y_voltage)
    voltage_predictions = voltage_model.predict(X)
    print(f"Voltage Model R2 Score: {r2_score(y_voltage, voltage_predictions):.2f}")
    print(f"Voltage Model MSE: {mean_squared_error(y_voltage, voltage_predictions):.2f}")

    # --- Train Current Model ---
    print("Training Current Prediction Model...")
    current_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # For a simple start, you could use: current_model = LinearRegression()

    current_model.fit(X, y_current)
    current_predictions = current_model.predict(X)
    print(f"Current Model R2 Score: {r2_score(y_current, current_predictions):.2f}")
    print(f"Current Model MSE: {mean_squared_error(y_current, current_predictions):.2f}")

    # --- Save Models ---
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(model_dir, exist_ok=True) # Create directory if it doesn't exist

    joblib.dump(voltage_model, os.path.join(model_dir, 'voltage_model.joblib'))
    joblib.dump(current_model, os.path.join(model_dir, 'current_model.joblib'))
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.joblib')) # Save feature names

    print(f"Models and feature columns saved to: {model_dir}")
    print("Model training complete.")

if __name__ == '__main__':
    train_and_save_models()

