import numpy as np
from models.lstm_model import train_lstm_model, predict_water_consumption_for_next_month
from services.weather_service import fetch_weather_for_tomorrow
import json
from flask import jsonify

def predict_day():
    # Dữ liệu thời tiết từ file JSON
    weather_data = [
        { "day": "08-11-2024", "humidity": 72, "temp": 28, "rainfall": 15, "vol": 185 },
        { "day": "09-11-2024", "humidity": 74, "temp": 28, "rainfall": 49, "vol": 149 },
        { "day": "10-11-2024", "humidity": 77, "temp": 27, "rainfall": 60, "vol": 137 },
        { "day": "11-11-2024", "humidity": 76, "temp": 27, "rainfall": 53, "vol": 144 },
        { "day": "12-11-2024", "humidity": 79, "temp": 26, "rainfall": 66, "vol": 129 },
        { "day": "13-11-2024", "humidity": 89, "temp": 26, "rainfall": 102, "vol": 85 },
        { "day": "14-11-2024", "humidity": 95, "temp": 25, "rainfall": 121, "vol": 80 },
        { "day": "15-11-2024", "humidity": 82, "temp": 24, "rainfall": 85, "vol": 119 },
        { "day": "16-11-2024", "humidity": 72, "temp": 22, "rainfall": 35, "vol": 171 },
        { "day": "17-11-2024", "humidity": 71, "temp": 22, "rainfall": 7, "vol": 200 },
        { "day": "18-11-2024", "humidity": 70, "temp": 24, "rainfall": 2, "vol": 204 },
        { "day": "19-11-2024", "humidity": 71, "temp": 26, "rainfall": 8, "vol": 195 },
    ]

   
    
    data_weather= fetch_weather_for_tomorrow()
     # Dữ liệu thời tiết ngày mai
   # Check if the data was successfully fetched
    if data_weather:
        # Dữ liệu thời tiết ngày mai
        weather_tomorrow = {
            "day":data_weather[3],
            "humidity": data_weather[1],  # Humidity
            "temp": data_weather[0],  # Temperature
            "rainfall": data_weather[2]  # Rainfall
        }
        print(weather_tomorrow)  # You can print this to verify the output
    else:
        print("Unable to fetch weather data.")

    # Chuẩn bị dữ liệu cho mô hình
    features = []
    targets = []

    for entry in weather_data:
        features.append([entry["humidity"], entry["temp"], entry["rainfall"]])
        targets.append(entry["vol"])

    # Huấn luyện mô hình
    model, scaler = train_lstm_model(features, targets)

    # Chuẩn bị dữ liệu thời tiết ngày mai để dự đoán
    tomorrow_features = [
        weather_tomorrow["humidity"],
        weather_tomorrow["temp"],
        weather_tomorrow["rainfall"]
    ]

    # Dự đoán lượng nước cần dùng
    predicted_water = predict_water_consumption_for_next_month(
        model, scaler, tomorrow_features
    )
    result = {
        "predicted_water_consumption": float(predicted_water),
        "day": weather_tomorrow["day"]
    }

    # In kết quả ra console
    print("Predicted Result:", result)

    # Trả về kết quả
    return result
    
   
