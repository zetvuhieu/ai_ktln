from flask import Flask, jsonify, request
from models.lstm_model import train_lstm_model, predict_water_consumption_for_next_month
import datetime
import numpy as np
import os
import json

app = Flask(__name__)

@app.route('/predict_water_consumption', methods=['GET'])
def predict_water():
   # Đọc dữ liệu thời tiết từ file JSON
    try:
        with open('data/data.json', 'r') as file:
            weather_data = json.load(file)
    except FileNotFoundError:
        return jsonify({"error": "File dữ liệu không tìm thấy."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Dữ liệu trong file không hợp lệ."}), 400

    # Chuẩn bị dữ liệu cho mô hình
    features = []
    targets = []

    for entry in weather_data:
        features.append([entry['humidity'], entry['temp'], entry['rainfall']])
        targets.append(entry['vol'])

    # Huấn luyện mô hình
    model, scaler = train_lstm_model(features, targets)

    # Xác định tháng tiếp theo
    current_month = datetime.datetime.now().month
    next_month = current_month + 1 if current_month < 12 else 1
    next_month_str = str(next_month).zfill(2)

    # Lấy dữ liệu thời tiết của tháng tiếp theo từ tập dữ liệu
    next_month_weather_info = next(
        (item for item in weather_data if item['month'] == next_month_str), None)

    if not next_month_weather_info:
        return jsonify({"error": "Không tìm thấy dữ liệu thời tiết cho tháng tiếp theo."}), 400

    # Chuẩn bị dữ liệu dự đoán
    next_month_features = [
        next_month_weather_info['humidity'],
        next_month_weather_info['temp'],
        next_month_weather_info['rainfall']
    ]
    
    predicted_water = predict_water_consumption_for_next_month(
        model, scaler, next_month_features)

    # Trả về kết quả
    return jsonify({
        "predicted_water_consumption": float(predicted_water),
        "next_month": next_month_str
    })

if __name__ == '__main__':
    # Lấy port từ môi trường, nếu không có thì sử dụng cổng 5000 mặc định
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

