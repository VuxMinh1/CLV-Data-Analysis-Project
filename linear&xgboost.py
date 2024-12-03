import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Đọc dữ liệu từ tệp Excel
file_path = "J:/CLV/cleaned_online_retail_II.xlsx"
data = pd.read_excel(file_path)

# Kiểm tra tên các cột trong dữ liệu để xác định tên cột chính xác
print(data.columns)

# Xử lý dữ liệu thiếu
data.dropna(subset=['Quantity', 'Price'], inplace=True)

# Tính toán TotalSpent từ Quantity và Price
data['TotalSpent'] = data['Quantity'] * data['Price']

# Chuyển đổi cột 'InvoiceDate' thành kiểu datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Tính toán Recency: số ngày kể từ giao dịch cuối cùng
data['Recency'] = (data['InvoiceDate'].max() - data['InvoiceDate']).dt.days

# Nhóm dữ liệu theo 'Customer ID' và tính toán các chỉ số RFM
rfm = data.groupby('Customer ID').agg({
    'Recency': 'min',  # Khoảng thời gian gần đây nhất khách hàng mua
    'Invoice': 'count',  # Số lần giao dịch (thay InvoiceNo bằng Invoice)
    'TotalSpent': 'sum'  # Tổng chi tiêu của khách hàng
}).reset_index()

# Tính Monetary là tổng chi tiêu của khách hàng
rfm['Monetary'] = rfm['TotalSpent']

# Tính toán CLV cho mỗi khách hàng
rfm['CLV'] = rfm['Monetary'] * rfm['Invoice'] / (rfm['Recency'] + 1)  # Công thức CLV đơn giản

# Xem trước dữ liệu RFM và CLV
print(rfm.head())

# Tách dữ liệu thành features (X) và target (y)
X = rfm[['Recency', 'Invoice', 'Monetary']]  # Các chỉ số RFM
y = rfm['CLV']  # CLV là target

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu: Chuẩn hóa dữ liệu (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hồi quy tuyến tính ---
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Đánh giá mô hình hồi quy tuyến tính
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# --- Tối ưu hóa XGBoost với GridSearchCV ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

# Thiết lập các tham số để tối ưu hóa
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Sử dụng mô hình tốt nhất
best_xgb_model = grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test_scaled)

# Đánh giá mô hình XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# In kết quả đánh giá
print(f'Hồi quy tuyến tính - RMSE: {rmse_lr}, R2: {r2_lr}, MAE: {mae_lr}')
print(f'XGBoost - RMSE: {rmse_xgb}, R2: {r2_xgb}, MAE: {mae_xgb}')

# --- Trực quan hóa kết quả ---
# Biểu đồ phân phối
plt.figure(figsize=(12, 6))
sns.histplot(y_test, color='blue', kde=True, label='Giá trị thực tế CLV', bins=30)
sns.histplot(y_pred_lr, color='orange', kde=True, label='Dự đoán CLV - Hồi quy tuyến tính', bins=30)
sns.histplot(y_pred_xgb, color='green', kde=True, label='Dự đoán CLV - XGBoost', bins=30)
plt.title('So sánh Phân phối: Giá trị thực tế và Dự đoán CLV')
plt.xlabel('Giá trị CLV')
plt.ylabel('Tần suất')
plt.legend()
plt.show()

# Biểu đồ so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(12, 6))

# Scatter plot giữa giá trị thực tế và giá trị dự đoán từ hồi quy tuyến tính
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.5)
plt.title('So sánh CLV - Hồi quy tuyến tính')
plt.xlabel('Giá trị thực tế CLV')
plt.ylabel('Giá trị dự đoán CLV (Hồi quy tuyến tính)')
plt.plot([0, max(y_test)], [0, max(y_test)], color='black', linestyle='--', label='Dự đoán hoàn hảo')
plt.legend()

# Scatter plot giữa giá trị thực tế và giá trị dự đoán từ XGBoost
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, color='green', alpha=0.5)
plt.title('So sánh CLV - XGBoost')
plt.xlabel('Giá trị thực tế CLV')
plt.ylabel('Giá trị dự đoán CLV (XGBoost)')
plt.plot([0, max(y_test)], [0, max(y_test)], color='black', linestyle='--', label='Dự đoán hoàn hảo')
plt.legend()

plt.tight_layout()
plt.show()

# --- Biểu đồ sai số ---
# Biểu đồ sai số giữa giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(12, 6))

# Sai số cho hồi quy tuyến tính
plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_lr, color='blue', kde=True, bins=30)
plt.title('Sai số - Hồi quy tuyến tính')
plt.xlabel('Sai số (Giá trị thực tế - Dự đoán)')

# Sai số cho XGBoost
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_xgb, color='green', kde=True, bins=30)
plt.title('Sai số - XGBoost')
plt.xlabel('Sai số (Giá trị thực tế - Dự đoán)')

plt.tight_layout()
plt.show()
