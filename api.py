from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math

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

class NaiveBayesClassifier:
    def __init__(self, smoothing=False):
        self.classes = None
        self.class_probs = {}
        self.feature_probs = {}
        self.smoothing = smoothing

    def fit(self, X, y):
        # Xác định các lớp
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Tính xác suất của mỗi lớp
        self.class_probs = {}
        for cls in self.classes:
            self.class_probs[cls] = np.sum(y == cls) / n_samples

        # Tính xác suất của các đặc trưng cho mỗi lớp
        self.feature_probs = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            
            # Khởi tạo từng đặc trưng
            cls_feature_probs = []
            for feature_idx in range(n_features):
                feature_values = X_cls[:, feature_idx]
                unique_values = np.unique(feature_values)
                
                # Xác suất của từng giá trị đặc trưng
                value_probs = {}
                for val in unique_values:
                    if self.smoothing:  # Làm trơn Laplace
                        count = np.sum((feature_values == val)) + 1
                        total = len(feature_values) + len(unique_values)
                    else:  # Không làm trơn
                        count = np.sum((feature_values == val))
                        total = len(feature_values)
                    
                    value_probs[val] = count / total
                
                cls_feature_probs.append(value_probs)
            
            self.feature_probs[cls] = cls_feature_probs

    def predict(self, X):
        predictions = []
        for sample in X:
            # Tính xác suất cho từng lớp
            class_scores = {}
            for cls in self.classes:
                # Bắt đầu với log xác suất của lớp  
                score = math.log(self.class_probs[cls])
                
                # Cộng log xác suất của từng đặc trưng
                for feature_idx, feature_val in enumerate(sample):
                    feature_prob = self.feature_probs[cls][feature_idx].get(feature_val, 1e-10)
                    score += math.log(feature_prob)
                
                class_scores[cls] = score
            
            # Chọn lớp có điểm số cao nhất
            predictions.append(max(class_scores, key=class_scores.get))
        
        return np.array(predictions)

def calculate_confusion_matrix(y_true, y_pred, classes):
    """Tính ma trận nhầm lẫn"""
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        true_idx = class_to_index[true]
        pred_idx = class_to_index[pred]
        confusion_matrix[true_idx, pred_idx] += 1
    
    return confusion_matrix.tolist()

def calculate_precision_recall_f1(confusion_matrix, classes):
    """Tính precision, recall, và F1-score cho từng lớp"""
    metrics = {}
    for i, cls in enumerate(classes):
        # True Positive
        tp = confusion_matrix[i][i]
        
        # False Positive (tổng cột trừ TP)
        fp = sum(confusion_matrix[j][i] for j in range(len(classes)) if j != i)
        
        # False Negative (tổng hàng trừ TP)
        fn = sum(confusion_matrix[i][j] for j in range(len(classes)) if j != i)
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return metrics

@app.route('/naive_bayes_no_smoothing', methods=['POST'])
def naive_bayes_no_smoothing():
    # Kiểm tra file được upload 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400
    
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

    # Chia dữ liệu thành tập huấn luyện và kiểm tra (70% huấn luyện, 30% kiểm tra)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_index = int(0.7 * len(X))
    
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    # Thuật toán Naive Bayes không làm trơn Laplace
    nb_classifier = NaiveBayesClassifier(smoothing=False)
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    
    # Tính toán các độ đo
    classes = np.unique(y)
    confusion_matrix = calculate_confusion_matrix(y_test, y_pred, classes)
    metrics = calculate_precision_recall_f1(confusion_matrix, classes)
    
    # Tính độ chính xác tổng thể
    accuracy = np.mean(y_test == y_pred)

    # Trả về kết quả
    result = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "class_metrics": metrics,
        "classes": classes.tolist()
    }
    return jsonify(result)

@app.route('/naive_bayes_with_smoothing', methods=['POST'])
def naive_bayes_with_smoothing():
    # Kiểm tra file được upload 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400
    
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

    # Chia dữ liệu thành tập huấn luyện và kiểm tra (70% huấn luyện, 30% kiểm tra)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_index = int(0.7 * len(X))
    
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    # Thuật toán Naive Bayes có làm trơn Laplace
    nb_classifier = NaiveBayesClassifier(smoothing=True)
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    
    # Tính toán các độ đo
    classes = np.unique(y)
    confusion_matrix = calculate_confusion_matrix(y_test, y_pred, classes)
    metrics = calculate_precision_recall_f1(confusion_matrix, classes)
    
    # Tính độ chính xác tổng thể
    accuracy = np.mean(y_test == y_pred)

    # Trả về kết quả
    result = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "class_metrics": metrics,
        "classes": classes.tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render tự động thiết lập cổng
    app.run(debug=True, host='0.0.0.0', port=port)  # Flask sẽ lắng nghe trên cổng do Render cung cấp
