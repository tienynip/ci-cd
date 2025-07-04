DỰ ĐOÁN GIÁ XE
- Ứng dụng web sử dụng **Flask** để dự đoán giá xe dựa trên thông tin đầu vào như năm sản xuất, số km đã đi, loại nhiên liệu, hộp số, v.v.  
- Mô hình Machine Learning được lưu trữ trong 'model.pkl' và có thể chọn nhiều thuật toán khác nhau để so sánh hiệu suất.

---

CHỨC NĂNG CHÍNH
- Dự đoán giá xe dựa trên thông tin nhập từ người dùng  
- Hỗ trợ nhiều mô hình Machine Learning khác nhau  
- Hiển thị **R² Score** để đánh giá độ chính xác của mô hình  
- Giao diện web đơn giản, dễ sử dụng  

---

CÁCH SỬ DỤNG
- Nếu lần đầu tiên chạy hãy xóa mô hình 'model.pkl'
- Sau đó chạy lại file car_prediction.py (lưu ý nhớ xóa ghi chú dòng này để tạo mô hình: joblib.dump(models, "model.pkl"))
- Sau đó chạy tiếp file main.py
- Sau đó copy http://127.0.0.1:5000 và lên trình duyệt dán link này trong trình duyệt

---

CÁC FEATURES ĐÃ THAY ĐỔI TỪ CHỮ THÀNH SỐ
Loại nhiên liệu: {"CNG": 0, "Diesel": 1, "LPG": 2, "Petrol": 3}
Loại người bán: {"Dealer": 0, "Individual": 1, "Trustmark Dealer": 2}
Hộp số: {"Automatic": 0, "Manual": 1}
Chủ sở hữu: {"First Owner": 0, "Fourth & Above Owner": 1, "Second Owner": 2, "Test Drive Car": 3, "Third Owner": 4}
