"""
配置参数模块
"""

# --- 路径配置 ---
# 训练集
TRAIN_INPUT_DIR ='./ships_train/'
TRAIN_OUTPUT_FILE = './processed_train.pkl'
TRAIN_STATS_FILE = './train_statistics.json'

# 测试集
TEST_INPUT_DIR = './ships_test/'
TEST_OUTPUT_FILE = './processed_test.pkl'
TEST_STATS_FILE = './test_statistics.json'

# 可视化
VISUALIZATION_OUTPUT_DIR = './visualizations/'

# --- 特征配置 ---
CONTINUOUS_FEATURES = ['Unit Weight (kg)']
CATEGORICAL_FEATURES = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
OTHER_FEATURES = ['order', 'Unit Nbr', 'Time Completed']

# 预处理后生成的特征（包含组合特征）
# position_id: yard + bay + col 的组合，标识垂直堆叠位置
PROCESSED_FEATURES = CATEGORICAL_FEATURES + ['position_id']

# 用于图构建的特征
# 包含：连续特征 + 类别特征 + 组合位置ID
FEATURES_FOR_GRAPH = CONTINUOUS_FEATURES + PROCESSED_FEATURES

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
