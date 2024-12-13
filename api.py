from flask import Flask, jsonify, request,send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from numpy import log2 as log
from sklearn.preprocessing import LabelEncoder
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

eps = np.finfo(float).eps

# ----------------------------- DECISION TREE-----------------------------

#1. Tìm entropy cho cột cuối cùng (cột mục tiêu)
def find_entropy(df):
    Class = df.keys()[-1] #danh sách các tên cột
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -(fraction * np.log2(fraction))
    return entropy

#2.Tính entropy cho từng cột
def find_entropy_attribute (df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()

    entropy2= 0
    for variable in variables:
        entropy = 0
        for target_variable  in target_variables:
            num = len(df[attribute][df[attribute]== variable][df[Class]== target_variable]) #vd cột weather có Sunny,Rain -> đếm số dòng dữ liệu của Weather có Sunny mà Yes và No
            den = len(df[attribute][df[attribute]== variable]) #đếm số dòng của thuộc tính (tổng dòng dữ liệu của data)
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps) #Tính entropy của từng specific values trong cột Weather (ví dụ)
        fraction2 = den / len(df) #den = tổng số dòng weather là sunny  / len = số dòng dữ liệu của dataset
        entropy2 += -fraction2*entropy #Tổng entropy của cột Weather = cộng dồn sau mỗi giá trị cụ thể của cột Weather

    return (abs(round(entropy2,4)))


#3. Tìm thuộc tính info thấp nhất
def find_winner (df):
    Entropy_att = []
    Inf = []

    for key in df.keys()[:-1]:
        Inf.append(find_entropy(df)-find_entropy_attribute(df,key)) #Tính information từng attribute rồi thêm vào mảng Inf
    
    return df.keys()[:-1][np.argmax(Inf)] #Tìm thuộc tính có info thấp nhất (gain mới lấy cao nhất)

def get_subtable (df,node, value):
    return df[df[node]== value].reset_index(drop=True) #lọc ra các giá trị của cột được chọn từ hàm find_winner

#4. Xây dựng ây quyết định
def build_tree (df, tree = None):
    Class = df.keys()[-1]

    #Tìm node
    node = find_winner(df)

    #Lấy ra những giá trị của cột node
    att_value = np.unique(df[node])

    #Tạo dic rỗng để tạo cây
    if tree is None:
        tree = {}
        tree[node] = {}

    #Tạo vòng lặp -> gọi đệ quy
    for value in att_value:

        subtable = get_subtable(df,node,value)
        clValue, counts = np.unique(subtable[Class], return_counts = True)

        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = build_tree(subtable)

    return tree

#5. Vẽ cây
import pydot
import uuid
def generate_unique_node():
    """ Generate a unique node label"""
    return str(uuid.uuid1())

def create_node (graph, label, shape = 'oval'):
    node = pydot.Node(generate_unique_node(), label = label, shape = shape)
    graph.add_node(node)
    return node

def create_edge(graph, node_parent, node_child, label):
    link = pydot.Edge(node_parent,node_child, label = label)
    graph.add_edge(link)
    return link

def walk_tree (graph, dictionary, prev_node = None):
    """Recursive construction of a decision tree stored as a dictionary"""

    for parent, child in dictionary.items():
        #root
        if not prev_node:
            root = create_node(graph,parent)
            walk_tree(graph,child,root)
            continue

        #node
        if isinstance(child, dict):
            for p,c  in child.items():
                n = create_node(graph,p)
                create_edge(graph, prev_node, n, str(parent))
                walk_tree(graph,c,n)

        #leaf
        else:
            leaf = create_node(graph, str(child), shape='box')
            create_edge(graph, prev_node, leaf, str(parent))


def plot_tree (dictionary, filename):
    graph = pydot.Dot(graph_type = 'graph')
    walk_tree(graph, dictionary)
    graph.write_png(filename)

    return filename

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    """Phục vụ tệp hình ảnh từ thư mục uploads"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
# -----------------------------------DECISION TREE ----------------------------------------------
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
    
    tree = build_tree(data) 
 
    # Vẽ cây quyết định và lấy tên file hình ảnh
    filename = os.path.join(app.config['UPLOAD_FOLDER'], "DecisionTree3.png")
    image_file = plot_tree(tree, filename=filename)

    result = {
         "image_file": image_file,
    }
    return jsonify(result)

# -----------------------------------------Gom cụm----------------------------------------------------------------------
# Khởi tạo lớp KMeans
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def initialize_random_clusters(self, X):
        # Chọn ngẫu nhiên các chỉ số điểm dữ liệu để làm centroid ban đầu
        random_indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        
        # Các cụm ban đầu, mỗi cụm sẽ chứa các điểm dữ liệu tương ứng
        initial_clusters = {i: [] for i in range(self.n_clusters)}
        
        # Gán các điểm dữ liệu vào các cụm ngẫu nhiên
        for idx in range(X.shape[0]):
            cluster_id = np.random.choice(self.n_clusters)  # Chọn ngẫu nhiên một cụm
            initial_clusters[cluster_id].append(idx)  # Gán điểm dữ liệu vào cụm
        

        # Tính toán centroid ban đầu từ các điểm trong mỗi cụm
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        for cluster_id, indices in initial_clusters.items():
            if indices:  # Kiểm tra xem cụm có điểm dữ liệu không
                cluster_points = X[indices]
                self.centroids[cluster_id] = np.mean(cluster_points, axis=0)
        
        #  Làm tròn centroids
        self.centroids = np.round(self.centroids, decimals=4)
        
        # Gán nhãn ban đầu cho tất cả các điểm dữ liệu
        self.labels = np.zeros(X.shape[0], dtype=int)
        for cluster_id, indices in initial_clusters.items():
            for idx in indices:
                self.labels[idx] = cluster_id
        
        return self.labels, self.centroids

    def fit(self, X):
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            old_labels = self.labels.copy()

            # Tính toán khoảng cách từ mỗi điểm đến các centroid
            distances = np.zeros((X.shape[0], self.n_clusters))  # Ma trận khoảng cách
            for i in range(self.n_clusters):
                # Tính khoảng cách Euclidean giữa điểm và centroid i
                distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
            
            # Gán nhãn cho các điểm dữ liệu theo centroid gần nhất
            self.labels = np.argmin(distances, axis=1)
            
            # Cập nhật lại centroid cho mỗi cụm
            for k in range(self.n_clusters):
                # Lấy các điểm thuộc cụm k
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = np.mean(cluster_points, axis=0)
            
            # Làm tròn toàn bộ centroids
            self.centroids = np.round(self.centroids, decimals=2)
            
            # Kiểm tra nếu centroids không thay đổi
            if np.all(old_centroids == self.centroids):
                break
        
        return self.centroids, self.labels

@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    # Kiểm tra file được upload
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400

    # Lưu file tạm thời
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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

    # Mã hóa dữ liệu phân loại
    le = LabelEncoder()
    df_encoded = data.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Kiểm tra số cột sau mã hóa
    if df_encoded.shape[1] < 3:
        return jsonify({"error": "Dataset must have at least 3 columns after encoding"}), 400

    # Lấy 3 thuộc tính đầu tiên để chạy K-Means
    data_for_clustering = df_encoded.iloc[:, :3].to_numpy()

    # Khởi tạo KMeans
    kmeans = KMeans(n_clusters=3)
    try:
        initial_labels, initial_centroids = kmeans.initialize_random_clusters(data_for_clustering)
        centroids, labels = kmeans.fit(data_for_clustering)
    except Exception as e:
        return jsonify({"error": f"Clustering failed: {str(e)}"}), 500

  # Nhóm các điểm theo cụm
    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for idx, cluster_id in enumerate(labels):
        clusters[cluster_id].append(idx)

    # Chuẩn bị kết quả trả về dạng JSON
    result = {
        "clusters": clusters,
        "centroids": centroids.tolist(),
        "initial_centroids": initial_centroids.tolist(),
        "initial_labels": initial_labels.tolist(),
        "final_labels": labels.tolist(),
    }
    return jsonify(result)

# -----------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------
# Function to calculate support
def calculate_support(transactions, itemsets):
    support_count = {}
    for itemset in itemsets:
        count = 0
        for transaction in transactions:
            if all(item in transaction for item in itemset):
                count += 1
        support_count[tuple(itemset)] = count
    return support_count

# Function to generate combinations manually
def generate_combinations(items, length):
    combinations = []
    n = len(items)
    def combine(current, start):
        if len(current) == length:
            combinations.append(current)
            return
        for i in range(start, n):
            combine(current + [items[i]], i + 1)
    combine([], 0)
    return combinations

# Generate frequent itemsets
def apriori(transactions, min_support):
    single_items = list(set(item for transaction in transactions for item in transaction))
    current_itemsets = [[item] for item in single_items]
    frequent_itemsets = []

    while current_itemsets:
        # Calculate support
        support = calculate_support(transactions, current_itemsets)
        # Filter based on minimum support
        current_itemsets = [
            list(itemset) for itemset, count in support.items()
            if count / len(transactions) >= min_support
        ]
        frequent_itemsets.extend(current_itemsets)
        # Generate next level combinations
        current_itemsets = generate_combinations(
            sorted(set(item for subset in current_itemsets for item in subset)),
            len(current_itemsets[0]) + 1
        ) if current_itemsets else []
    return frequent_itemsets

# Generate rules manually
def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            subsets = generate_combinations(itemset, i)
            for subset in subsets:
                remainder = [item for item in itemset if item not in subset]
                if remainder:
                    support_itemset = calculate_support(transactions, [itemset])[tuple(itemset)]
                    support_subset = calculate_support(transactions, [subset])[tuple(subset)]
                    confidence = support_itemset / support_subset
                    if confidence >= min_confidence:
                        rules.append({
                            "antecedent": subset,
                            "consequent": remainder,
                            "confidence": confidence
                        })
    return rules

@app.route('/association_rules', methods=['POST'])
def association_rules():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Đọc dữ liệu từ tệp CSV
        data = pd.read_csv(file_path, header=None)
        transactions = data.applymap(str).values.tolist()
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    # Lấy tham số từ người dùng
    min_support = float(request.form.get('min_support', 0.5))
    min_confidence = float(request.form.get('min_confidence', 0.7))

    # Tạo các tập mục phổ biến và luật kết hợp
    frequent_itemsets = apriori(transactions, min_support)
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)

    # Trả về kết quả
    return jsonify({
        "frequent_itemsets": frequent_itemsets,
        "rules": [
            {"antecedent": antecedent, "consequent": consequent, "confidence": confidence}
            for antecedent, consequent, confidence in rules
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render tự động thiết lập cổng
    app.run(debug=True, host='0.0.0.0', port=port)  # Flask sẽ lắng nghe trên cổng do Render cung cấp