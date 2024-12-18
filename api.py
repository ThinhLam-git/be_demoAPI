import base64
import csv
from io import BytesIO
from flask import Flask, jsonify, request,send_from_directory
from flask_cors import CORS
from matplotlib import pyplot as plt
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
import logging

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
    filename = os.path.join(app.config['UPLOAD_FOLDER'], "DecisionTree.png")
    image_file = plot_tree(tree, filename=filename)
    
    result = {
         "image_file": f"https://be-webdm.onrender.com/uploads/{os.path.basename(image_file)}"
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
        iteration_result = []

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

            iteration_result.append({
            "iteration": iteration + 1,
            "labels": self.labels.tolist(),  # Chuyển labels thành list
            "centroids": self.centroids.tolist() 
            })
            
            
            # Kiểm tra nếu centroids không thay đổi
            if np.all(old_centroids == self.centroids):
                break
        
        return iteration_result,self.centroids, self.labels

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
        iteration_result,centroids, labels = kmeans.fit(data_for_clustering)
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
        "final": iteration_result
    }
    return jsonify(result)

# -------------------------------------------NAIVE BAYES----------------------------------------------
class NaiveBayesClassifier:
    def __init__(self, smoothing=False):
        self.classes = None
        self.class_probs = {}
        self.feature_probs = {}
        self.smoothing = smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        self.class_probs = {cls: np.sum(y == cls) / n_samples for cls in self.classes}
        self.feature_probs = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.feature_probs[cls] = []
            for feature_idx in range(n_features):
                feature_values = X_cls[:, feature_idx]
                unique_values = np.unique(feature_values)
                value_probs = {
                    val: (np.sum(feature_values == val) + (1 if self.smoothing else 0)) /
                         (len(feature_values) + (len(unique_values) if self.smoothing else 0))
                    for val in unique_values
                }
                self.feature_probs[cls].append(value_probs)

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {
                cls: math.log(self.class_probs[cls]) + sum(
                    math.log(self.feature_probs[cls][feature_idx].get(feature_val, 1e-10))
                    for feature_idx, feature_val in enumerate(sample)
                )
                for cls in self.classes
            }
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

def calculate_confusion_matrix(y_true, y_pred, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[class_to_index[true], class_to_index[pred]] += 1
    return confusion_matrix

def calculate_precision_recall_f1(confusion_matrix, classes):
    metrics = {}
    for i, cls in enumerate(classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics[cls] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
    return metrics

def plot_confusion_matrix(confusion_matrix, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/naive_bayes_no_smoothing', methods=['POST'])
def naive_bayes():
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

    # Parse input data
    X = np.array(data['X'])
    y = np.array(data['y'])
    X_test = np.array(data['X_test'])
    smoothing = data.get('smoothing', False)

    # Initialize and train the classifier
    classifier = NaiveBayesClassifier(smoothing=smoothing)
    classifier.fit(X, y)

    # Predict the test set
    y_pred = classifier.predict(X_test)

    # Calculate confusion matrix and metrics
    classes = classifier.classes
    confusion_matrix = calculate_confusion_matrix(y, y_pred, classes)
    metrics = calculate_precision_recall_f1(confusion_matrix, classes)

    # Generate confusion matrix plot as base64 string
    confusion_matrix_img = plot_confusion_matrix(confusion_matrix, classes)

    return jsonify({
        'y_pred': y_pred.tolist(),
        'confusion_matrix': confusion_matrix.tolist(),
        'metrics': metrics,
        'confusion_matrix_plot': confusion_matrix_img
    })

@app.route('/naive_bayes_with_smoothing', methods=['POST'])
def naive_bayes_laplace():
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
    
    # Parse input data
    X = np.array(data['X'])
    y = np.array(data['y'])
    X_test = np.array(data['X_test'])

    # Initialize and train the classifier with Laplace smoothing
    classifier = NaiveBayesClassifier(smoothing=True)
    classifier.fit(X, y)

    # Predict the test set
    y_pred = classifier.predict(X_test)

    # Calculate confusion matrix and metrics
    classes = classifier.classes
    confusion_matrix = calculate_confusion_matrix(y, y_pred, classes)
    metrics = calculate_precision_recall_f1(confusion_matrix, classes)

    # Generate confusion matrix plot as base64 string
    confusion_matrix_img = plot_confusion_matrix(confusion_matrix, classes)

    return jsonify({
        'y_pred': y_pred.tolist(),
        'confusion_matrix': confusion_matrix.tolist(),
        'metrics': metrics,
        'confusion_matrix_plot': confusion_matrix_img
    })

#------------------------------tap pho bien -----------------------
# Function to calculate support
def calculate_support(transactions, itemsets):
    support_count = {}
    for itemset in itemsets:
        count = 0
        for transaction in transactions:
            # Chuyển itemset và transaction sang set để sử dụng issubset
            if set(itemset).issubset(set(transaction)):
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

def generate_rules(frequent_itemsets, transactions, min_confidence):
    """
    Generate association rules from frequent itemsets.
    """
    logging.info("Starting rule generation")
    
    # Tạo bảng tra cứu hỗ trợ (support_table)
    support_table = {}
    for item in frequent_itemsets:
        # Lấy đúng `itemset` từ dict
        itemset = item["itemset"]
        support_table[frozenset(itemset)] = item["support"]

    rules = []
    for item in frequent_itemsets:
        itemset = item["itemset"]  # Lấy itemset từ dict
        if len(itemset) < 2:
            continue  # Bỏ qua các tập chỉ có 1 phần tử

        itemset_frozen = frozenset(itemset)
        subsets = generate_subsets(itemset)  # Sinh tất cả tập con không rỗng
        for antecedent in subsets:
            antecedent_frozen = frozenset(antecedent)
            consequent = itemset_frozen - antecedent_frozen
            if consequent:
                # Tra cứu hỗ trợ
                support_itemset = support_table[itemset_frozen]
                support_antecedent = support_table[antecedent_frozen]

                # Tính độ tin cậy
                confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0
                if confidence >= min_confidence:
                    # Thêm luật vào danh sách
                    rules.append({
                        "antecedent": set(antecedent),
                        "consequent": set(consequent),
                        "confidence": confidence,
                    })
    logging.info("Rule generation finished")
    return rules

logging.basicConfig(level=logging.DEBUG,  # Log everything from DEBUG level and above
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # Log to file
                        logging.StreamHandler()         # Log to console
                    ])

# Hàm sinh tập phổ biến tối đại
def find_maximal_itemsets(frequent_itemsets):
    """
    Tìm tập phổ biến tối đại từ danh sách các tập phổ biến.

    Parameters:
        frequent_itemsets (list of dict): Danh sách các tập phổ biến với cấu trúc
            [{"itemset": [items], "support": support_value}, ...]

    Returns:
        list of dict: Danh sách các tập phổ biến tối đại với cấu trúc tương tự frequent_itemsets.
    """
    # Chuyển danh sách tập phổ biến thành danh sách tập hợp để dễ xử lý
    itemsets = [set(item["itemset"]) for item in frequent_itemsets]
    maximal_itemsets = []

    for i, itemset in enumerate(itemsets):
        # Kiểm tra nếu không có tập phổ biến nào chứa itemset hiện tại
        is_maximal = all(not itemset < other for j, other in enumerate(itemsets) if i != j)
        if is_maximal:
            maximal_itemsets.append(frequent_itemsets[i])

    return maximal_itemsets

def apriori(transactions, min_support):
    logging.info("Starting Apriori algorithm")
    single_items = list(set(item for transaction in transactions for item in transaction))
    current_itemsets = [[item] for item in single_items]
    frequent_itemsets = []
    step = 1

    while current_itemsets:
        logging.info("Step %d: Current itemsets = %s", step, current_itemsets)
        support = calculate_support(transactions, current_itemsets)
        logging.info("Support counts: %s", support)

        # Filter itemsets based on min_support and store support
        filtered_itemsets = [
            {"itemset": list(itemset), "support": count / len(transactions)}
            for itemset, count in support.items()
            if count / len(transactions) >= min_support
        ]

        frequent_itemsets.extend(filtered_itemsets)
        logging.info("Frequent itemsets after filtering: %s", filtered_itemsets)

        # Generate next level combinations
        current_itemsets = generate_combinations(
            sorted(set(item for subset in current_itemsets for item in subset)),
            len(current_itemsets[0]) + 1
        ) if current_itemsets else []

        step += 1

    logging.info("Apriori algorithm finished")
    # Tìm tập phổ biến tối đại từ frequent itemsets
    maximal_itemsets = find_maximal_itemsets(frequent_itemsets)
    logging.info("Maximal itemsets: %s", maximal_itemsets)
    return frequent_itemsets, maximal_itemsets


# Sinh tất cả tập hợp con của một tập hợp
def generate_subsets(itemset):
    items = list(itemset)
    subsets = []
    for i in range(1, 1 << len(items)): # Từ 1 đến 2^n - 1
        subset = {items[j] for j in range(len(items)) if (i & (1 << j))}
        subsets.append(subset)
    return subsets
    
@app.route('/association_rules', methods=['POST'])
def association_rules():
    logging.info("Received request at /association_rules")

    if 'file' not in request.files:
        logging.warning("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        logging.warning("Invalid file format: %s", file.filename)
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        logging.info("File saved successfully at %s", file_path)
    except Exception as e:
        logging.error("Failed to save file: %s", str(e))
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        # Read data from CSV file
        data = pd.read_csv(file_path)
        transactions = data.applymap(str).fillna('').values.tolist()
# Loại bỏ cột đầu tiên bằng cách lấy tất cả cột từ cột thứ 2 trở đi
        transactions = [[item for item in transaction[1:] if item] for transaction in transactions]
        logging.info("File processed successfully")
    except Exception as e:
        logging.error("Failed to read or process file: %s", str(e))
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    # Get parameters from user input
    try:
        min_support = float(request.form.get('min_support', 0.5))
        min_confidence = float(request.form.get('min_confidence', 0.7))
        logging.info("Parameters received: min_support=%f, min_confidence=%f", min_support, min_confidence)
    except ValueError:
        logging.error("Invalid parameters for min_support or min_confidence")
        return jsonify({"error": "Invalid parameters for min_support or min_confidence"}), 400
        
    try:
        # Sinh tập phổ biến tối đại và luật kết hợp
        frequent_itemsets, maximal_itemsets = apriori(transactions, min_support)
        rules = generate_rules(frequent_itemsets, transactions, min_confidence)
        logging.info("Frequent itemsets and rules generated successfully")
    except Exception as e:
        logging.error("Error during Apriori algorithm execution: %s", str(e))
        return jsonify({"error": f"Error generating association rules: {str(e)}"}), 500
    result = {
    "frequent_itemsets": [
        {"itemset": list(item["itemset"]), "support": item["support"]}
        for item in frequent_itemsets
    ],
    "maximal_itemsets": [
        {"itemset": list(item["itemset"]), "support": item["support"]}
        for item in maximal_itemsets
    ],
    "rules": [
        {
            "antecedent": list(rule["antecedent"]),
            "consequent": list(rule["consequent"]),
            "confidence": rule["confidence"]
        }
        for rule in rules
    ]
}

    logging.info("Results generated successfully")
    
    # Return results
    return jsonify(result)

#-----------------------------ĐỘ TƯƠNG QUAN--------------------------------
    
def read_csv(file_path):
        """
        Đọc dữ liệu từ tệp CSV và chuyển thành các danh sách số liệu.
        Giả sử tệp có hai cột dữ liệu, không tính dòng tiêu đề.
        """
        x = []
        y = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua dòng tiêu đề, nếu có
            for row in reader:
                if len(row) >= 2:  # Đảm bảo có ít nhất 2 cột
                    x.append(float(row[0].strip()))
                    y.append(float(row[1].strip()))
        return x, y

def mean(values):
    """Tính giá trị trung bình."""
    return sum(values) / len(values)

def pearson_correlation(x, y):
    """
    Tính hệ số tương quan Pearson giữa hai danh sách số liệu x và y.
    Điều kiện: x và y phải có cùng độ dài.
    """
    if len(x) != len(y):
        raise ValueError("Hai danh sách x và y phải có cùng độ dài.")

    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    # Tính các thành phần của công thức
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    denominator = (denominator_x * denominator_y) ** 0.5

    if denominator == 0:
        return 0  # Trường hợp đặc biệt: nếu biến x hoặc y không thay đổi.

    return numerator / denominator

def interpret_correlation(r):
    """
    Đưa ra kết luận dựa trên hệ số tương quan Pearson.
    """
    if r == 1:
        return "Hai biến có mối quan hệ tuyến tính hoàn hảo và cùng chiều."
    elif r == -1:
        return "Hai biến có mối quan hệ tuyến tính hoàn hảo nhưng ngược chiều."
    elif 0.7 <= r < 1:
        return "Hai biến có mối quan hệ tuyến tính chặt chẽ và cùng chiều."
    elif -1 < r <= -0.7:
        return "Hai biến có mối quan hệ tuyến tính chặt chẽ nhưng ngược chiều."
    elif 0.3 <= r < 0.7:
        return "Hai biến có mối quan hệ tuyến tính trung bình và cùng chiều."
    elif -0.7 < r <= -0.3:
        return "Hai biến có mối quan hệ tuyến tính trung bình nhưng ngược chiều."
    elif -0.3 < r < 0.3:
        return "Hai biến có rất ít hoặc không có mối quan hệ tuyến tính."
    else:
        return "Mối quan hệ giữa hai biến không rõ ràng."

@app.route('/pearson_correlation', methods=['POST'])
def calculate_correlation():
    logging.info("Received request at /pearson_correlation")

    if 'file' not in request.files:
        logging.warning("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        logging.warning("Invalid file format: %s", file.filename)
        return jsonify({"error": "Invalid file format, please input a .csv file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        logging.info("File saved successfully at %s", file_path)
    except Exception as e:
        logging.error("Failed to save file: %s", str(e))
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        # Đọc dữ liệu từ CSV file
        x, y = read_csv(file_path)
        logging.info("File processed successfully")
    except Exception as e:
        logging.error("Failed to read or process file: %s", str(e))
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    try:
        # Tính toán hệ số tương quan Pearson
        r = pearson_correlation(x, y)
        conclusion = interpret_correlation(r)
        logging.info("Pearson correlation calculated successfully")
    except Exception as e:
        logging.error("Error during correlation calculation: %s", str(e))
        return jsonify({"error": f"Error calculating correlation: {str(e)}"}), 500

    result = {
        "pearson_correlation": r,
        "interpretation": conclusion
    }
    logging.info("Results generated successfully")

    return jsonify(result)
#----------------------------- Tập thô (reduct)-----------------------------
# Sinh luật với độ chính xác 100%
def generate_rules_rough_set(data, reduct):
    """Sinh các luật dựa trên reduct."""
    rules = []
    decision_col = data.columns[-1]

    for _, group in data.groupby(list(reduct)):
        decisions = group[decision_col].unique()
        if len(decisions) == 1:  # Chỉ tạo luật nếu quyết định duy nhất
            condition = " AND ".join(
                f"{col}={group.iloc[0][col]}" for col in reduct
            )
            rule = f"IF {condition} THEN {decision_col}={decisions[0]}"
            rules.append(rule)
    return rules


def indiscernibility_relation(data, attributes):
    """Calculate indiscernibility relation for given attributes."""
    groups = data.groupby(attributes).groups
    return [list(indices) for indices in groups.values()]

def lower_approximation(data, target_set, ind_relation):
    """Calculate lower approximation of a target set."""
    lower = []
    for subset in ind_relation:
        if set(subset).issubset(target_set):
            lower.extend(subset)
    return lower

def upper_approximation(data, target_set, ind_relation):
    """Calculate upper approximation of a target set."""
    upper = []
    for subset in ind_relation:
        if set(subset) & set(target_set):
            upper.extend(subset)
    return upper

def rough_accuracy(lower, upper):
    """Calculate the accuracy of a rough set."""
    return len(lower) / len(upper) if len(upper) > 0 else 0

def discernibility_matrix(data):
    """Tạo ma trận phân biệt."""
    n = len(data)
    attributes = list(data.columns[1:-1])  # Loại bỏ cột quyết định
    matrix = [[set() for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            diff_attributes = {
                attr
                for attr in attributes
                if data.iloc[i][attr] != data.iloc[j][attr]
            }
            if data.iloc[i][-1] != data.iloc[j][-1]:  # Chỉ xét cặp khác giá trị quyết định
                matrix[i][j] = diff_attributes
    return matrix

def get_discernibility_conditions(matrix):
    """Trích xuất tất cả các điều kiện phân biệt từ ma trận."""
    conditions = set()
    for row in matrix:
        for cell in row:
            if cell:  # Chỉ lưu các tập khác rỗng
                conditions.add(frozenset(cell))
    return list(conditions)

def reducts_from_conditions(attributes, conditions):
    """Tính reduct nhỏ nhất từ các điều kiện phân biệt."""
    min_reduct_size = len(attributes)  # Kích thước reduct nhỏ nhất
    reducts = []

    def satisfies_condition(subset):
        subset_set = set(subset)
        return all(
            any(term.issubset(subset_set) for term in conditions)
            for term in conditions
        )

    for size in range(1, len(attributes) + 1):
        for subset in generate_all_combinations(attributes):
            if len(subset) == size and satisfies_condition(subset):
                if size < min_reduct_size:
                    reducts = [set(subset)]
                    min_reduct_size = size
                elif size == min_reduct_size:
                    reducts.append(set(subset))
        if reducts:  # Kết thúc ngay khi tìm thấy reduct nhỏ nhất
            break

    return reducts

def generate_all_combinations(attributes):
    """Sinh tất cả các tổ hợp thuộc tính."""
    combinations = []

    def combine(current, start):
        if current:
            combinations.append(current)
        for i in range(start, len(attributes)):
            combine(current + [attributes[i]], i + 1)

    combine([], 0)
    return combinations


@app.route('/rough-set', methods=['POST'])
def rough_set():
    try:
        # Kiểm tra xem file có được tải lên không
        if 'file' not in request.files:
            return jsonify({'error': "No file uploaded."}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': "Invalid file type. Please upload a CSV file."}), 400

        # Đọc file CSV thành DataFrame
        dataset = pd.read_csv(file)
        attributes = list(dataset.columns[1:-1])  # Bỏ qua cột đầu tiên và cột quyết định
        target_class_combinations = generate_all_combinations(attributes)

        # Tính toán tập thô
        rough_set_results = {}

        for comb in target_class_combinations:
            ind_relation = indiscernibility_relation(dataset, comb)
            lower = {}
            upper = {}
            accuracy = {}

            # Lấy các lớp quyết định
            decision_classes = dataset[dataset.columns[-1]].unique()

            for cls in decision_classes:
                # Chuyển cls thành chuỗi để tránh lỗi so sánh
                target_set = dataset.index[
                    dataset[dataset.columns[-1]] == str(cls)  # cls phải phù hợp với giá trị trong cột quyết định
                ].tolist()

                # Tính toán lower và upper approximation
                lower[str(cls)] = lower_approximation(dataset, target_set, ind_relation)
                upper[str(cls)] = upper_approximation(dataset, target_set, ind_relation)

                # Tính accuracy
                accuracy[str(cls)] = rough_accuracy(lower[str(cls)], upper[str(cls)])

            rough_set_results[str(comb)] = {
                "lower_approximation": lower,
                "upper_approximation": upper,
                "accuracy": accuracy,
            }

        # Tính reducts và sinh luật
        matrix = discernibility_matrix(dataset)
        conditions = get_discernibility_conditions(matrix)
        reducts = reducts_from_conditions(attributes, conditions)
        all_rules = {}
        for reduct in reducts:
            rules = generate_rules_rough_set(dataset, reduct)
            all_rules[f"Reduct {list(reduct)}"] = rules

        # Trả về kết quả
        return jsonify({
            'rough_set': rough_set_results,
            'reducts_and_rules': {
                'reducts': [list(reduct) for reduct in reducts],
                'rules': all_rules
            }
        })

    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


    
@app.errorhandler(404)
def page_not_found(e):
    logging.warning("404 error: %s", str(e))
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))   #Render tự động thiết lập cổng
    app.run(debug=True, host='0.0.0.0', port=port)  
