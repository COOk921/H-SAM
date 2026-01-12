import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import permutations

import pdb

# 引入配置 (确保路径正确，需要运行目录在项目根目录)
try:
    from data.cluster.config import ALPHA, BETA
except ImportError:
    # Fallback default values if config not found
    print("Warning: Could not import config, using default ALPHA=0.9, BETA=0.1")
    ALPHA = 0.9
    BETA = 0.1

def get_location_features(df):
    """提取位置特征"""
    location_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer']
    # 确保是 DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame().T
        
    location_data = df[location_cols].copy()
    for col in location_cols:
        le = LabelEncoder()
        location_data[col] = le.fit_transform(location_data[col].astype(str))
    return location_data.values.astype(np.float32)

def get_weight_features(df):
    """提取重量特征"""
    return df['Unit Weight (kg)'].values.reshape(-1, 1)

def compute_median_sigma(features, sample_size=1000):
    """计算高斯核带宽 sigma (基于中位数启发式)"""
    n = len(features)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        features = features[indices]
    
    dist_sq = cdist(features, features, metric='sqeuclidean')
    # 取上三角部分
    upper_tri = dist_sq[np.triu_indices_from(dist_sq, k=1)]
    if len(upper_tri) == 0:
        return 1.0
    median_dist = np.median(np.sqrt(upper_tri))
    return median_dist if median_dist > 1e-8 else 1.0

def compute_global_sigmas(df_list):
    """基于所有数据计算全局统一的 sigma_w 和 sigma_l"""
    all_df = pd.concat(df_list, ignore_index=True)
    
    # Weight Sigma
    w_feats = get_weight_features(all_df)
    scaler_w = StandardScaler()
    w_feats = scaler_w.fit_transform(w_feats)
    sigma_w = compute_median_sigma(w_feats)
    
    # Location Sigma
    l_feats = get_location_features(all_df)
    scaler_l = StandardScaler()
    l_feats = scaler_l.fit_transform(l_feats)
    sigma_l = compute_median_sigma(l_feats)
    
    return sigma_w, sigma_l, scaler_w, scaler_l

def calculate_inter_group_affinity(df_i, df_j, sigma_w, sigma_l, scaler_w, scaler_l):
    """计算两个组之间的亲和力 A_Si,Sj"""
    # 提取特征并标准化
    w_i = scaler_w.transform(get_weight_features(df_i))
    w_j = scaler_w.transform(get_weight_features(df_j))
    
    l_i = scaler_l.transform(get_location_features(df_i))
    l_j = scaler_l.transform(get_location_features(df_j))
    
    # 计算两组间的距离矩阵 (N_i x N_j)
    dist_sq_w = cdist(w_i, w_j, metric='sqeuclidean')
    dist_sq_l = cdist(l_i, l_j, metric='sqeuclidean')
    
    # 计算相似度矩阵
    sim_w = np.exp(-dist_sq_w / (2 * sigma_w ** 2))
    sim_l = np.exp(-dist_sq_l / (2 * sigma_l ** 2))
    
    # 融合相似度
    sim_total = ALPHA * sim_l + BETA * sim_w
    
    # 计算平均亲和力: sum(sim) / (|Si|*|Sj|)
    affinity = np.mean(sim_total)
    
    return affinity

def solve_tsp_path_brute_force(dist_matrix):
    """
    遍历所有可能的排列以找到绝对最优解。
    由于 K 很小，全排列搜索是可行的 (K <= 10)。
    """
    K = dist_matrix.shape[0]
    if K <= 1:
        return [0]
    
    best_path = None
    min_total_dist = float('inf')
    
    # 遍历所有排列
    for path in permutations(range(K)):
        current_dist = 0
        valid_path = True
        for i in range(K - 1):
            current_dist += dist_matrix[path[i], path[i+1]]
            if current_dist >= min_total_dist:
                valid_path = False
                break
        
        if valid_path and current_dist < min_total_dist:
            min_total_dist = current_dist
            best_path = list(path)
            
    return best_path


def solve_tsp_path_nn(dist_matrix):
    """
    使用 Nearest Neighbor 算法寻找最短 Hamiltonian Path。
    由于 K 较小，尝试以每个节点为起点的 NN，取最优。
    
    Args:
        dist_matrix: K x K 距离矩阵
    Returns:
        best_path: 节点索引列表
    """
    K = dist_matrix.shape[0]
    if K <= 1:
        return [0]
    
    best_path = []
    min_total_dist = float('inf')
    
    # 尝试将每个节点作为起点
    for start_node in range(K):
        current_path = [start_node]
        current_node = start_node
        visited = {start_node}
        total_dist = 0
        
        while len(visited) < K:
            # 找最近的未访问邻居
            nearest_dist = float('inf')
            nearest_node = -1
            
            for next_node in range(K):
                if next_node not in visited:
                    d = dist_matrix[current_node, next_node]
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_node = next_node
            
            if nearest_node != -1:
                visited.add(nearest_node)
                current_path.append(nearest_node)
                total_dist += nearest_dist
                current_node = nearest_node
            else:
                break
        
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_path = current_path
            
    return best_path

def sort_sub_blocks(df_list):
    """
    核心排序逻辑:
    1. 计算全局 Sigma
    2. 构建组间距离矩阵 (基于 Affinity)
    3. 求解 TSP Path 确定顺序
    """
    if not df_list:
        return []
    if len(df_list) == 1:
        return df_list
        
    K = len(df_list)
    print(f"Sorting {K} sub-blocks using Affinity-based TSP...")

    # 1. 预计算全局标准化器和 Sigma
    sigma_w, sigma_l, scaler_w, scaler_l = compute_global_sigmas(df_list)
    
    # 2. 构建距离矩阵
    dist_matrix = np.zeros((K, K))
    epsilon = 1e-6
    
    for i in range(K):
        for j in range(i + 1, K):
            affinity = calculate_inter_group_affinity(
                df_list[i], df_list[j], 
                sigma_w, sigma_l, scaler_w, scaler_l
            )
            # 转化为距离: D = 1 / (A + eps)
            distance = 1.0 / (affinity + epsilon)
            
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance  # 对称
    
    # 3. 求解 TSP Path
    # TODO: 这里可以使用 LKH-3 替换 solve_tsp_path_nn 以获得更优解
    sorted_indices = solve_tsp_path_nn(dist_matrix)
    
    #sorted_indices = solve_tsp_path_brute_force(dist_matrix)
    
    print(f"Computed Order: {sorted_indices}")
    
    return [df_list[i] for i in sorted_indices]


def sort_sub_blocks_T(df_list):
    """
    1. 子块排序逻辑：根据每个子块中 'order' 列的首个数值进行排序。
    """
    # 提取每个子块的第一行 order 值作为排序依据
    first_order_vals = [df['order'].values[0] for df in df_list]
    # 获取排序后的索引顺序
    sort_indices = np.argsort(first_order_vals)

    print(f"GT Order: {sort_indices}")
    return [df_list[i] for i in sort_indices]


def preprocess_and_rerank_pred(sorted_dfs):
    """
    2. 全局预测值重排：
       - 过滤特征全为 0 的无效行。
       - 对 Pred 进行重新排名与偏移，确保移除无效行后 Pred 依然连续。
    """
    merged_df = pd.DataFrame()
    num_offset = 0
    random_samples = pd.DataFrame()

    for df in sorted_dfs:
        # 过滤无效行：除了最后一列 (pred) 以外，其他特征若全为 0 则删除
        mask = ~(df.iloc[:, :-1] == 0).all(axis=1)
        df_valid = df[mask].copy()

        if df_valid.empty:
            continue

        # 对当前块的 pred 进行局部重排 (0 到 n-1) 并加上全局偏移量
        df_valid['pred'] = df_valid['pred'].rank(method='dense').astype(int) - 1 + num_offset
        
        # 更新偏移量为当前已合并的总行数
        num_offset += len(df_valid)

        # 随机采样
        sample = df_valid.sample(n=1) if len(df_valid) > 0 else pd.DataFrame()
        if not sample.empty:
            random_samples = pd.concat([random_samples, sample], ignore_index=True)
        
        # 合并到总表
        merged_df = pd.concat([merged_df, df_valid], ignore_index=True)

    return merged_df, random_samples

def evaluate_performance(merged_df, group_key):
    """
    3. 性能评估：计算 Kendall's Tau 相关系数。
    """
    if merged_df.empty:
        return 0.0
    
    kendall_corr, _ = kendalltau(merged_df['order'].values, merged_df['pred'].values)
    print(f"Group {group_key} - Single Kendall: {kendall_corr:.4f}")
    return kendall_corr

def main():
    result_folder = './result/'
    output_folder = './result/merged/'
    os.makedirs(output_folder, exist_ok=True)

    # 1. 扫描文件并分组
    csv_files = [f for f in os.listdir(result_folder) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in result folder.")
        return

    file_groups = {}
    for file in csv_files:
        key = file.split("',")[0] + "'"
        file_groups.setdefault(key, []).append(file)

    avg_kendall_corr = 0
    all_cluster_row = {}

    # 2. 遍历每个组进行处理
    for group_key, files in file_groups.items():
        print(f"\nProcessing Group: {group_key}")
        # 读取该组所有文件
        df_list = [pd.read_csv(os.path.join(result_folder, f)) for f in files]

        # A. 子块排序 (Affinity TSP)
        #sorted_dfs = sort_sub_blocks(df_list)
        sorted_dfs = sort_sub_blocks_T(df_list)
        

        # B. 过滤与 Pred 全局重排
        merged_df, random_rows = preprocess_and_rerank_pred(sorted_dfs)

        # C. 统计与存储
        all_cluster_row[group_key[1:]] = random_rows
        
        # D. 评估
        corr = evaluate_performance(merged_df, group_key)
        avg_kendall_corr += corr

        # E. 保存结果
        merged_file_name = group_key.strip("'") + '\').csv'
        merged_df.to_csv(os.path.join(output_folder, merged_file_name), index=False)

    # 3. 后处理与最终输出
    if file_groups:
        #process_merged_data(all_cluster_row)
        print(f"\nOverall Average Kendall: {avg_kendall_corr / len(file_groups):.4f}")
    
    # pdb.set_trace()

if __name__ == "__main__":
    main()