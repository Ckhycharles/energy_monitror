# energy/views.py

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.http import JsonResponse, HttpResponse
from .models import ElectricityReading
import json
import datetime
import csv
import joblib # For loading models
import pandas as pd # For data manipulation
import os # For path manipulation

# --- Authentication Views ---
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')

        if password == password2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email already exists. Please login.')
                return redirect('login')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'Username is already taken.')
                return redirect('register')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                messages.success(request, 'Account created successfully! You can now log in.')
                return redirect('login')
        else:
            messages.info(request, 'Passwords do not match.')
            return redirect('register')
    else:
        return render(request, 'register.html')

def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        if not email or not password:
            messages.error(request, 'Please enter both email and password.')
            return render(request, 'login.html')

        user = None
        try:
            user_obj = User.objects.filter(email=email).first()
            if user_obj:
                user = authenticate(request, username=user_obj.username, password=password)
        except Exception as e:
            print(f"An unexpected error occurred during user lookup: {e}")
            messages.error(request, 'An internal error occurred. Please try again.')
            return render(request, 'login.html')

        if user is not None:
            if user.is_active:
                auth_login(request, user)
                messages.success(request, f"Welcome, {user.username}!")
                return redirect('dashboard')
            else:
                messages.info(request, 'Account is inactive. Please contact support.')
                return render(request, 'login.html')
        else:
            messages.error(request, 'Invalid email or password.')
            return render(request, 'login.html')
    else:
        return render(request, "login.html")

def logout(request):
    auth_logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('login')

# --- Dashboard & Data Views ---
def index(request):
    current_time = datetime.datetime.now()
    latest_reading = ElectricityReading.objects.order_by('-timestamp').first()
    readings_for_chart = ElectricityReading.objects.all().order_by('timestamp')[:50]

    # --- Future Usage Prediction Logic ---
    predicted_voltage = None
    predicted_current = None
    predicted_timestamp = None
    prediction_error = None

    try:
        hours_ahead = 1 # Default prediction for 1 hour ahead
        future_timestamp = datetime.datetime.now() + datetime.timedelta(hours=hours_ahead)

        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Ai_models', 'trained_models')
        
        voltage_model_path = os.path.join(model_dir, 'voltage_model.joblib')
        current_model_path = os.path.join(model_dir, 'current_model.joblib')
        feature_columns_path = os.path.join(model_dir, 'feature_columns.joblib')

        if not (os.path.exists(voltage_model_path) and os.path.exists(current_model_path) and os.path.exists(feature_columns_path)):
            prediction_error = 'Machine learning models not found. Please train them first by running Ai_models/trainer.py'
        else:
            voltage_model = joblib.load(voltage_model_path)
            current_model = joblib.load(current_model_path)
            feature_columns = joblib.load(feature_columns_path)

            future_data = pd.DataFrame(0, index=[0], columns=feature_columns)
            
            future_data['hour'] = future_timestamp.hour
            future_data['day_of_week'] = future_timestamp.weekday()
            future_data['day_of_year'] = future_timestamp.timetuple().tm_yday
            future_data['month'] = future_timestamp.month

            # Ensure the order of columns matches the training data
            future_data = future_data[feature_columns]

            predicted_voltage = round(voltage_model.predict(future_data)[0], 2)
            predicted_current = round(current_model.predict(future_data)[0], 2)
            predicted_timestamp = future_timestamp.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        print(f"Error during prediction in index view: {e}")
        import traceback
        traceback.print_exc()
        prediction_error = f'Error calculating prediction: {str(e)}'
    # --- End Future Usage Prediction Logic ---

    context = {
        'current_time': current_time,
        'latest_reading': latest_reading,
        'readings_for_chart_json': json.dumps([
            {'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'voltage': r.voltage, 'current': r.current}
            for r in readings_for_chart
        ]),
        'predicted_voltage': predicted_voltage,
        'predicted_current': predicted_current,
        'predicted_timestamp': predicted_timestamp,
        'prediction_error': prediction_error,
    }
    return render(request, 'dashboard.html', context)

def predict_bill(request):
    from Ai_models.predictor import predict_monthly_bill
    readings = ElectricityReading.objects.all()
    prediction = predict_monthly_bill(readings)
    return JsonResponse({'predicted_bill': prediction})

def receive_latest_data(request):
    readings = ElectricityReading.objects.order_by('-timestamp')[:20][::-1]
    data = [{
        'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'voltage': r.voltage,
        'current': r.current,
    } for r in readings]
    return JsonResponse({'readings': data})

def export_readings_to_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="electricity_readings.csv"'

    writer = csv.writer(response)
    
    writer.writerow(['ID', 'Timestamp', 'Device ID', 'Voltage', 'Current', 'Room'])

    readings = ElectricityReading.objects.all().order_by('timestamp')
    for reading in readings:
        writer.writerow([
            reading.id,
            reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            reading.device_id,
            reading.voltage,
            reading.current,
            reading.room,
        ])
    return response

