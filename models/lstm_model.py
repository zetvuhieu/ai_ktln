import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Hàm huấn luyện mô hình LSTM
def train_lstm_model(features, targets):
    # Chuyển đổi dữ liệu đầu vào thành mảng numpy và reshape cho LSTM
    features = np.array(features)
    targets = np.array(targets)
    features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

    # Chuẩn hóa dữ liệu đầu ra (targets)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_targets = scaler.fit_transform(targets.reshape(-1, 1))

    # Khởi tạo mô hình LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, features.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Huấn luyện mô hình với 100 epochs
    model.fit(features, scaled_targets, epochs=5, verbose=1)

    return model, scaler

# Hàm dự đoán lượng nước cần tưới cho tháng tiếp theo
def predict_water_consumption_for_next_month(model, scaler, next_month_features):
    # Định dạng dữ liệu đầu vào cho mô hình
    next_month_features = np.array(next_month_features).reshape((1, 1, len(next_month_features)))
    
    # Dự đoán giá trị và chuyển đổi giá trị dự đoán về đơn vị gốc
    scaled_prediction = model.predict(next_month_features)
    predicted_value = scaler.inverse_transform(scaled_prediction)[0][0]
    
    return predicted_value
