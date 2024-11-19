from flask import Flask, jsonify
from services.predict_month import predict_water
from services.predict_day import predict_day
import os

app = Flask(__name__)

@app.route('/predict_water_consumption', methods=['GET'])
def predict_water_consumption():
    return predict_water()

@app.route('/predict_day', methods=['GET'])
def predict_tomorrow():
    try:
        prediction = predict_day()
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Lấy port từ môi trường, nếu không có thì sử dụng cổng 5000 mặc định
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
