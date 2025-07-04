import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
import joblib


data = pd.read_csv("car_data.csv")
data.head()

data.describe()

data.info()

"""Xử lý dữ liệu"""

# Xóa các dòng null
data.dropna(inplace=True)

data.info()

data.describe()

data['year'].unique()

data['selling_price'].min()

data['selling_price'].max()

data['selling_price'].unique()

data['km_driven'].min()

data['km_driven'].max()

data['km_driven'].unique()

data["fuel"].unique()

data['seller_type'].unique()

data['transmission'].unique()

data['owner'].unique()

data['engine'].unique()

data['engine'] = data['engine'].str.replace(' CC', '').astype(int)

data['engine'].unique()

data['seats'].max()

data['seats'].min()

fuel_encoder = LabelEncoder()
data['fuel'] = fuel_encoder.fit_transform(data['fuel'])

a = data['fuel']
a = dict(zip(fuel_encoder.classes_, fuel_encoder.transform(fuel_encoder.classes_)))
print(a)

seller_type_encoder = LabelEncoder()
data['seller_type'] = seller_type_encoder.fit_transform(data['seller_type'])

b = data['seller_type']
b = dict(zip(seller_type_encoder.classes_, seller_type_encoder.transform(seller_type_encoder.classes_)))
print(b)

transmission_encoder = LabelEncoder()
data['transmission'] = transmission_encoder.fit_transform(data['transmission'])

c = data['transmission']
c = dict(zip(transmission_encoder.classes_, transmission_encoder.transform(transmission_encoder.classes_)))
print(c)

owner_encoder = LabelEncoder()
data['owner'] = owner_encoder.fit_transform(data['owner'])

d = data['owner']
d = dict(zip(owner_encoder.classes_, owner_encoder.transform(owner_encoder.classes_)))
print(d)

data.describe()

X = data[["year",'km_driven','fuel','seller_type','transmission','owner','engine','seats']]

y = data['selling_price']

"""Chia dữ liệu thành tập huấn luyện"""

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

"""Xây dựng mô hình"""

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'R2 Score: {r2}')

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor()
}
model_scores = {}
# Đánh giá các mô hình
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    print(f"{name}:")
    print(f"  R2 Score: {r2}")
    print('-' * 30)
    model_scores[name] = {'R2': r2}

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

models['X_test'] = X_test
models['y_test'] = y_test
models['model_scores'] = model_scores
# joblib.dump(models, "model.pkl")