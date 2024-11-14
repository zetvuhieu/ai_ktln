import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm_model(features, targets):
    # Chuyển đổi dữ liệu đầu vào thành mảng numpy và reshape để phù hợp với LSTM
    features = np.array(features)
    targets = np.array(targets)
    
    # Reshape dữ liệu đầu vào thành (số lượng mẫu, 1, số lượng đặc trưng)
    features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    targets = scaler.fit_transform(targets.reshape(-1, 1))

    # Khởi tạo mô hình LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, features.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Huấn luyện mô hình
    model.fit(features, targets, epochs=200, verbose=0)
    
    return model, scaler

def predict_water_consumption_for_next_month(model, scaler, next_month_features):
    # Chuyển đổi đầu vào thành numpy array và reshape để phù hợp với mô hình LSTM
    next_month_features = np.array(next_month_features).reshape((1, 1, len(next_month_features)))
    
    # Dự đoán và đảo chuẩn hóa
    predicted_scaled = model.predict(next_month_features)
    predicted = scaler.inverse_transform(predicted_scaled)
    
    return predicted[0, 0]

