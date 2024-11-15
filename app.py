from flask import Flask, jsonify, request
from models.lstm_model import train_lstm_model, predict_water_consumption_for_next_month
import datetime
import numpy as np

app = Flask(__name__)

@app.route('/predict_water_consumption', methods=['GET'])
def predict_water():
    weather_data = [
        {'month': '01', 'humidity': 71, 'temp': 22, 'rainfall': 7, 'vol': 200},  
        {'month': '02', 'humidity': 70, 'temp': 24, 'rainfall': 2, 'vol': 204}, 
        {'month': '03', 'humidity': 71, 'temp': 26, 'rainfall': 8, 'vol': 195}, 
        {'month': '04', 'humidity': 72, 'temp': 28, 'rainfall': 15, 'vol': 185}, 
        {'month': '05', 'humidity': 74, 'temp': 28, 'rainfall': 49, 'vol': 149}, 
        {'month': '06', 'humidity': 77, 'temp': 27, 'rainfall': 60, 'vol': 137}, 
        {'month': '07', 'humidity': 76, 'temp': 27, 'rainfall': 53, 'vol': 144}, 
        {'month': '08', 'humidity': 79, 'temp': 26, 'rainfall': 66, 'vol': 129}, 
        {'month': '09', 'humidity': 89, 'temp': 26, 'rainfall': 102, 'vol': 85}, 
        {'month': '10', 'humidity': 95, 'temp': 25, 'rainfall': 121, 'vol': 80}, 
        {'month': '11', 'humidity': 82, 'temp': 24, 'rainfall': 85, 'vol': 119}, 
        {'month': '12', 'humidity': 72, 'temp': 22, 'rainfall': 35, 'vol': 171}, 
        {'month': '01', 'humidity': 61, 'temp': 22, 'rainfall': 7, 'vol': 210},  
        {'month': '02', 'humidity': 75, 'temp': 24, 'rainfall': 2, 'vol': 199}, 
        {'month': '03', 'humidity': 71, 'temp': 28, 'rainfall': 12, 'vol': 189}, 
        {'month': '04', 'humidity': 72, 'temp': 28, 'rainfall': 15, 'vol': 185}, 
        {'month': '05', 'humidity': 74, 'temp': 28, 'rainfall': 49, 'vol': 149}, 
        {'month': '06', 'humidity': 77, 'temp': 27, 'rainfall': 60, 'vol': 137}, 
        {'month': '07', 'humidity': 76, 'temp': 27, 'rainfall': 53, 'vol': 144}, 
        {'month': '08', 'humidity': 79, 'temp': 26, 'rainfall': 66, 'vol': 129}, 
        {'month': '09', 'humidity': 89, 'temp': 26, 'rainfall': 102, 'vol': 85}, 
        {'month': '10', 'humidity': 95, 'temp': 25, 'rainfall': 121, 'vol': 80},
        {'month': '01', 'humidity': 71, 'temp': 22, 'rainfall': 7, 'vol': 200},  
        {'month': '02', 'humidity': 70, 'temp': 24, 'rainfall': 2, 'vol': 204}, 
        {'month': '03', 'humidity': 71, 'temp': 26, 'rainfall': 8, 'vol': 195}, 
        {'month': '04', 'humidity': 72, 'temp': 28, 'rainfall': 15, 'vol': 185}, 
        {'month': '05', 'humidity': 74, 'temp': 28, 'rainfall': 49, 'vol': 149}, 
        {'month': '06', 'humidity': 77, 'temp': 27, 'rainfall': 60, 'vol': 137}, 
        {'month': '07', 'humidity': 76, 'temp': 27, 'rainfall': 53, 'vol': 144}, 
        {'month': '08', 'humidity': 79, 'temp': 26, 'rainfall': 66, 'vol': 129}, 
        {'month': '09', 'humidity': 89, 'temp': 26, 'rainfall': 102, 'vol': 85}, 
        {'month': '10', 'humidity': 95, 'temp': 25, 'rainfall': 121, 'vol': 80}, 
        {'month': '11', 'humidity': 82, 'temp': 24, 'rainfall': 85, 'vol': 119}, 
        {'month': '12', 'humidity': 72, 'temp': 22, 'rainfall': 35, 'vol': 171}, 
        {'month': '11', 'humidity': 82, 'temp': 24, 'rainfall': 85, 'vol': 119}, 
        {'month': '12', 'humidity': 72, 'temp': 22, 'rainfall': 40, 'vol': 166},
    ]

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

