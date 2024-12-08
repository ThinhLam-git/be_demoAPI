from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


app = Flask(__name__)
CORS(app)  # Cho phép tất cả các nguồn gốc (origin)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Đảm bảo thư mục 'uploads' tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/run_decision_tree', methods=['POST'])
def run_decision_tree():
    # Kiểm tra file được upload 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        return ({"error": "Invalid file format, please input a .csv file"}), 400
    
    # Lưu file tạm thời 
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Đọc dữ liệu từ file csv
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
    
    # Kiểm tra dữ liệu đầu vào
    if data.shape[1] < 2:
        return jsonify({"error": "Dataset must have at least 2 columns"}), 400
    
    # Tách dữ liệu thành đầu vào (features) và đầu ra (label)
    X = data.iloc[:, :-1].values  # Tất cả các cột trừ cột cuối
    y = data.iloc[:, -1].values  # Cột cuối cùng là label

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tạo mô hình Decision Tree và huấn luyện
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Dự đoán và tính độ chính xác
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Xuất cấu trúc cây quyết định dưới dạng văn bản
    tree_structure = export_text(clf, feature_names=data.columns[:-1].tolist())

    # Trả về kết quả
    result = {
        "accuracy": accuracy,
        "tree_structure": tree_structure
    }
    return jsonify(result)

@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    # Kiểm tra file được upload 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        return ({"error": "Invalid file format, please input a .csv file"}), 400
    
    # Lưu file tạm thời 
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Đọc dữ liệu từ file csv
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
    
    # Kiểm tra dữ liệu đầu vào 
    if data.shape[1] < 2:
        return jsonify({"error": "Dataset must have at least 2 columns"}), 400
    
    # Example: Lấy 2 cột đầu tiên để gom cụm
    X = data.iloc[:, :2]

    # Áp dụng thuật toán K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    
    # Trả về kết quả
    clusters = kmeans.labels_.tolist()
    cluster_centers = kmeans.cluster_centers_.tolist()
    result = {
        "clusters": clusters,
        "cluster_centers": cluster_centers,
        "n_clusters": len(cluster_centers)
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Cổng từ môi trường
    app.run(host='0.0.0.0', port=port)  # Flask sẽ lắng nghe trên cổng này
