from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.metrics import r2_score

# Load model.pkl
models_data = joblib.load('model.pkl')

# Tách dữ liệu huấn luyện & kiểm tra ra khỏi models
X_test = models_data.pop("X_test", None)
y_test = models_data.pop("y_test", None)
model_scores = models_data.pop("model_scores", {})

# Mặc định chọn mô hình đầu tiên
selected_model = list(models_data.keys())[0]

app = Flask(__name__)

@app.route('/')
def home():
    # Lọc danh sách model (loại bỏ X_test, y_test)
    model_list = list(models_data.keys())
    return render_template('index.html', models=model_list, selected_model=selected_model)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global selected_model

        # Lấy mô hình được chọn
        model_name = request.form.get('model_name')
        if model_name and model_name in models_data:
            selected_model = model_name

        model = models_data[selected_model]

        # Nhận dữ liệu đầu vào từ form
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = int(request.form['fuel'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])
        engine = int(request.form['engine'])
        seats = int(request.form['seats'])

        # Chuẩn bị input
        input_features = np.array([[year, km_driven, fuel, seller_type, transmission, owner, engine, seats]])
        
        # Dự đoán giá xe
        predicted_price = model.predict(input_features)[0]

        # Tính R² Score nếu có tập kiểm tra
        r2 = "N/A"
        if X_test is not None and y_test is not None:
            y_pred_test = model.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)

        return render_template(
            'index.html', 
            prediction_text=f'{predicted_price:,.2f}', 
            r2_score=f'{r2:.4f}' if isinstance(r2, float) else "N/A",
            models=models_data.keys(), 
            selected_model=model_name
        )
    except Exception as e:
        return render_template(
            'index.html', 
            prediction_text=f'Lỗi: {str(e)}', 
            r2_score="N/A",
            models=models_data.keys(), 
            selected_model=selected_model
        )

if __name__ == '__main__':
    app.run(debug=True)
