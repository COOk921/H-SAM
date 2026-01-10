import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.utils import shuffle
from itertools import permutations
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import copy

import pdb
from tqdm import tqdm

# ==============================================================================
# 步骤 1: 数据准备 - 生成训练用的Pair
# ==============================================================================

def create_pairwise_data(df: pd.DataFrame, feature_cols: list, window_size: int):
    """
    为单个DataFrame生成所有窗口内的pair数据和标签.

    Args:
        df (pd.DataFrame): 输入的DataFrame, 原始顺序即为Ground Truth.
        feature_cols (list): 用于计算的特征列名列表.
        window_size (int): 滑动窗口的大小 (D).

    Returns:
        tuple: 包含两个元素的元组:
               - X_pairs (np.ndarray): N x (2 * num_features) 的数组, 每行为一个pair的拼接特征.
               - y_labels (np.ndarray): N x 1 的数组, 每行是一个pair的标签 (0或1).
    """
    # 记录Ground Truth顺序
    df['ground_truth_order'] = np.arange(len(df))

    # 打乱DataFrame
    df_shuffled = shuffle(df, random_state=42).reset_index(drop=True)

    # 提取特征和顺序信息
    features = df_shuffled[feature_cols].values
    orders = df_shuffled['ground_truth_order'].values

    X_pairs = []
    y_labels = []
    pair_original_indices = []

    # 滑动窗口生成pairs
    for i in range(len(df_shuffled) - window_size + 1):
        window_features = features[i : i + window_size]
        window_orders = orders[i : i + window_size]

        # 第一个node作为锚点
        anchor_feature = window_features[0]
        anchor_order = window_orders[0]

        # 与窗口内其他node生成pair
        for j in range(1, window_size):
            other_feature = window_features[j]
            other_order = window_orders[j]

            # 特征拼接
            pair_feature = np.concatenate([anchor_feature, other_feature])
            X_pairs.append(pair_feature)

            # 生成标签
            label = 1 if anchor_order < other_order else 0
            y_labels.append(label)

            pair_original_indices.append([anchor_order, other_order])

    if not X_pairs:
        # 如果DataFrame的行数小于窗口大小，则返回空数组
        num_features = len(feature_cols)
        return np.array([]).reshape(0, 2 * num_features), np.array([]),np.array([])

    return np.array(X_pairs), np.array(y_labels),pair_original_indices


# ==============================================================================
# 步骤 2: 建立模型
# ==============================================================================

class PairwiseRankingModel(nn.Module):
    """
    一个简单的MLP模型，用于预测pair的正确顺序概率.
    输入是一个拼接了两个node特征的向量.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim (int): 输入特征的维度 (等于 2 * num_node_features).
            hidden_dim (int): 隐藏层的维度.
        """
        super(PairwiseRankingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.network(x)


# ==============================================================================
# 步骤 3: 训练模型
# ==============================================================================

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, learning_rate: float = 0.001,device: torch.device = torch.device('cpu')):
    """
    训练单个模型.

    Args:
        model (nn.Module): 待训练的模型实例.
        X_train (np.ndarray): 训练数据特征.
        y_train (np.ndarray): 训练数据标签.
        epochs (int): 训练轮数.
        learning_rate (float): 学习率.
    """
    if X_train.shape[0] == 0:
        print("    警告: 没有可供训练的pair数据，跳过训练。")
        return
   
    # 转换为PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"    开始训练... 共 {epochs} 个 epochs.")
    # 训练循环
    model.train()
    for epoch in tqdm(range(epochs), desc="    Training progress"):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        preds = (outputs > 0.5).float() 
        correct = (preds == y_tensor).sum().item()  
        acc = correct / y_tensor.size(0)  

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, ACC: {acc:.4f}")
    


# ==============================================================================
# 步骤 4 & 5: 预测、生成邻接矩阵并创建PyG图
# ==============================================================================
def build_graph_from_pairs(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols_for_graph: list,
    X_pairs: np.ndarray,
    pair_original_indices: np.ndarray,
    threshold: float,
    device: torch.device = torch.device('cpu')
):
    """
    使用训练好的模型对生成过的pair进行预测，构建带权有向图.

    Args:
        model (nn.Module): 训练好的模型.
        df (pd.DataFrame): 原始DataFrame，用于提取节点特征.
        feature_cols_for_graph (list): 用于图节点特征的列名.
        X_pairs (np.ndarray): 在第一步中生成的pair特征数据.
        pair_original_indices (np.ndarray): 每个pair对应的原始节点索引 <源, 目标>.
        threshold (float): 置信度阈值P.

    Returns:
        torch_geometric.data.Data: 创建好的PyG图对象.
    """
    
    if X_pairs.shape[0] == 0:
        print("    警告: 没有pair数据，创建一个空图。")
        x = torch.tensor(df[feature_cols_for_graph].values, dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_pairs, dtype=torch.float32).to(device)
        # 批量预测所有pair的置信度
        confidences = model(X_tensor).squeeze().cpu().numpy()

    # 根据阈值筛选边
    edge_mask = confidences > threshold
    
    # 获取通过筛选的边的索引和特征
    filtered_indices = np.array(pair_original_indices)[edge_mask]
    filtered_confidences = confidences[edge_mask]
    
    # --- 创建PyG图 ---
    # 1. 节点特征 (x)
    x = torch.tensor(df[feature_cols_for_graph].values, dtype=torch.float32)

    if filtered_indices.shape[0] > 0:
        # 2. 边索引 (edge_index) - [2, num_edges]
        edge_index = torch.tensor(filtered_indices.T, dtype=torch.long)
        # 3. 边特征 (edge_attr) - [num_edges, num_edge_features]
        edge_attr = torch.tensor(filtered_confidences, dtype=torch.float32).view(-1, 1)
    else:
        # 如果没有边通过阈值，则创建空边
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    
    # 4. 创建Data对象
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"    图已创建: {graph}")
    print(f"    根据阈值 {threshold}，共添加了 {edge_index.shape[1]} 条边.")
   
    return graph


def read_data(file_path: str,continuous_features:list,categorical_features:list,other_features:list) -> dict:
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)
   
    data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}
    
    processed_data_local = {}
    for key, df in data.items():
        processed_df = pd.DataFrame(index=df.index)

        # 复制其他特征
        for col in other_features:
            processed_df[col] = df[col]

        # 转化连续特征
        local_scaler = StandardScaler()
        scaled_continuous = local_scaler.fit_transform(df[continuous_features])
        for i, col_name in enumerate(continuous_features):
            processed_df[col_name] = scaled_continuous[:, i]
        
        local_vocab_mappings = {}
        for col in categorical_features:
            unique_categories = df[col].unique()
            local_vocab_mappings[col] = {category: i + 1 for i, category in enumerate(unique_categories)}
            local_vocab_mappings[col]['[UNK]'] = 0 # 同样为未知类别保留0
            
            processed_df[col] = df[col].map(local_vocab_mappings[col]).fillna(0).astype(int)
    
        processed_data_local[key] = processed_df

    return processed_data_local


# ==============================================================================
# 步骤 6: 主流程
# ==============================================================================
def build_graphs_for_dataset(
    data_dict: dict,
    global_model: nn.Module, # 接收训练好的模型
    feature_cols_for_model: list,
    feature_cols_for_graph: list,
    window_size: int,
    threshold: float,
    device: torch.device
) -> dict:
    """
    使用一个预训练的全局模型，为数据字典中的每个DataFrame生成图。
    """
    final_results = {}
    
    #确保模型在评估模式
    global_model.to(device)
    global_model.eval()
    
    idx= 0
    for key, df in data_dict.items():
        print(f"\n===== 开始处理 DataFrame: '{key}-{idx}/{len(data_dict)}' =====")
        idx+=1
        
        original_df = copy.deepcopy(df)

        # 步骤 1: 创建推理数据 (使用新函数)
        print("步骤 1: 创建Inference pair数据...")
        X_infer_pairs, pair_indices = create_inference_pairs(original_df, feature_cols_for_model, window_size)
        
        print(f"    为 '{key}' 生成了 {len(X_infer_pairs)} 个待预测pair.")

        # 步骤 2 & 3: 生成邻接矩阵并创建图 (使用现有函数)
        print("步骤 2 & 3: 生成邻接矩阵并创建PyG图...")
        
        # [cite: 12] build_graph_from_pairs 函数可以复用
        # 它只需要模型和pair数据，不关心是否是训练数据
        pyg_graph = build_graph_from_pairs(
            model=global_model, # [cite: 13] 使用全局模型
            df=original_df,
            feature_cols_for_graph=feature_cols_for_graph,
            X_pairs=X_infer_pairs, # [cite: 14] 使用推理pair
            pair_original_indices=pair_indices, # [cite: 14] 使用推理pair的索引
            threshold=threshold,
            device=device
        )

        # 步骤 4: 存储结果
        final_results[key] = {
            'data': original_df, 
            'graph': pyg_graph 
        }
        print(f"===== DataFrame '{key}' 处理完成 =====")

    return final_results


# new 用于训练 
def run_training_pipeline(data_dict: dict, feature_cols_for_model: list, window_size: int, epochs: int, learning_rate: float, hidden_dim: int, device: torch.device):
    """
    聚合所有DataFrame的pair数据，训练一个全局模型。
    """
    print("===== 开始全局模型训练 =====")
    all_X_train = []
    all_y_train = []
    
    print("步骤 1: 聚合所有DataFrame的pair数据...")
    for key, df in data_dict.items():
        processing_df = copy.deepcopy(df)
        
        #  使用您现有的函数为每个df生成训练pair
        X_train, y_train, _ = create_pairwise_data(processing_df, feature_cols_for_model, window_size)
        
        if X_train.shape[0] > 0:
            all_X_train.append(X_train)
            all_y_train.append(y_train)

    # 检查是否有数据
    if not all_X_train:
        print("警告: 没有任何可供训练的pair数据。")
        return None
        
    # 合并为一个大的ndarray
    global_X_train = np.concatenate(all_X_train, axis=0)
    global_y_train = np.concatenate(all_y_train, axis=0)

    
    print(f"    聚合完成。总共 {global_X_train.shape[0]} 个训练pair。")

    # 2. 初始化全局模型
    print("步骤 2: 初始化全局模型...")
    input_dim = 2 * len(feature_cols_for_model)
    global_model = PairwiseRankingModel(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"    模型已创建，输入维度: {input_dim}")

    # 3. 训练全局模型 [cite: 9]
    print("步骤 3: 训练全局模型...")
    train_model(global_model, global_X_train, global_y_train, epochs=epochs, learning_rate=learning_rate, device=device)
    
    print("===== 全局模型训练完成 =====")
    return global_model

# new 用于创建推理pair 
def create_inference_pairs(df: pd.DataFrame, feature_cols: list, window_size: int):
    """
    为单个DataFrame生成所有窗口内的pair数据 (用于推理).
    这个版本不打乱数据，也不生成标签。

    Args:
        df (pd.DataFrame): 输入的DataFrame.
        feature_cols (list): 用于计算的特征列名列表.
        window_size (int): 滑动窗口的大小 (D).

    Returns:
        tuple: 
               - X_pairs (np.ndarray): N x (2 * num_features) 的数组.
               - pair_original_indices (np.ndarray): N x 2 的数组，[源, 目标] 索引.
    """
    
    #  (修改点: 删除了 'ground_truth_order' 和 'shuffle')
    features = df[feature_cols].values
    # 原始索引
    original_indices = np.arange(len(df)) 

    X_pairs = []
    pair_original_indices = []

    # [cite: 3] 滑动窗口生成pairs (在原始顺序上)
    for i in range(len(df) - window_size + 1):
        window_features = features[i : i + window_size]
        window_orders = original_indices[i : i + window_size] # 这里的 "order" 就是索引

        # 第一个node作为锚点
        anchor_feature = window_features[0]
        anchor_order = window_orders[0] # 即索引 i

        # [cite: 4] 与窗口内其他node生成pair
        for j in range(1, window_size):
            other_feature = window_features[j]
            other_order = window_orders[j] # 即索引 i+j

            # 特征拼接
            pair_feature = np.concatenate([anchor_feature, other_feature])
            X_pairs.append(pair_feature)

            #  (修改点: 不再生成标签 y_labels)
            
            # 我们记录 (锚点, 其他点) 的原始索引
            pair_original_indices.append([anchor_order, other_order]) 

    if not X_pairs:
        num_features = len(feature_cols)
        return np.array([]).reshape(0, 2 * num_features), np.array([])

    return np.array(X_pairs), np.array(pair_original_indices)

# ==============================================================================
# 示例：如何使用
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 设定超参数 ---
    # (这部分保持不变) 
    continuous_features = ['Unit Weight (kg)']
    categorical_features = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer', ]
    other_features = ['order', 'Unit Nbr','Time Completed']

    FEATURES_FOR_MODEL = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
    FEATURES_FOR_GRAPH = ['Unit Weight (kg)','Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer']
    
    
    D_WINDOW_SIZE = 4
    P_THRESHOLD = 0.6
    EPOCHS = 200
    LEARNING_RATE = 0.005
    HIDDEN_DIM = 512
    
    READ_PATH = "./data/container_data_cluster.pkl"
    WRITE_PATH = "./data/processed_container_data_cluster.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. 读取数据 ---
    training_data = read_data(READ_PATH, continuous_features, categorical_features,other_features)

    # --- 3. 运行训练流程 ---
#     print("--- 阶段 1: 训练全局模型 ---")
    trained_global_model = run_training_pipeline(
        data_dict=training_data,
        feature_cols_for_model=FEATURES_FOR_MODEL,
        window_size=D_WINDOW_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_dim=HIDDEN_DIM,
        device=device
    )

#     # (可选) 在这里保存模型
    torch.save(trained_global_model.state_dict(), "global_classify_model.pth")
    
    
    # (可选) 如果是测试，在这里加载模型
    # trained_global_model.load_state_dict(torch.load("global_ranking_model.pth"))

    # --- 4. 运行推理流程 (使用已训练的模型) ---
    print("\n--- 阶段 2: 使用全局模型构建图 ---")
    
    # 备注: 
    # 在真实场景中，您会在这里加载 *测试* 数据 (test_data)
    # final_output = build_graphs_for_dataset(test_data, ...)
    
    # 这里我们暂时使用 训练 数据来构建图 
    final_output = build_graphs_for_dataset(
        data_dict=training_data, 
        global_model=trained_global_model,
        feature_cols_for_model=FEATURES_FOR_MODEL,
        feature_cols_for_graph=FEATURES_FOR_GRAPH,
        window_size=D_WINDOW_SIZE,
        threshold=P_THRESHOLD,
        device=device
    )
    
    # --- 5. 保存结果 ---
    with open(WRITE_PATH, 'wb') as f:
        pickle.dump(final_output, f)
        
    print(f"处理完成，结果已保存到 {WRITE_PATH}")

