"""
配置参数模块
"""

# --- 路径配置 ---
INPUT_DIR = './ships/'
OUTPUT_FILE = './processed_container_data_hetero(new_with_kmeans).pkl'
OUTPUT_STATS_FILE = './pipeline_statistics.json'
VISUALIZATION_OUTPUT_DIR = './visualizations/'

# --- 特征配置 ---
CONTINUOUS_FEATURES = ['Unit Weight (kg)']
CATEGORICAL_FEATURES = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
OTHER_FEATURES = ['order', 'Unit Nbr', 'Time Completed']
FEATURES_FOR_GRAPH = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# --- 聚类算法配置 ---
# 可选值: 'spectral' (谱聚类) 或 'kmeans' (K-means)
CLUSTERING_METHOD = 'kmeans'

# --- 谱聚类超参数 ---
ALPHA = 0.9  # 位置相似度权重
BETA = 0.1   # 重量相似度权重

# --- K-means 超参数 ---
KMEANS_MAX_ITER = 300  # K-means 最大迭代次数
KMEANS_N_INIT = 10     # K-means 初始化次数

# --- 图构建超参数 ---
KNN_K = 3  # KNN 邻居数
