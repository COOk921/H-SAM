# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# import pdb
# import os
# from sklearn.preprocessing import MinMaxScaler

# # 1. 定义输入和输出文件夹
# ships_dir = './ships/'
# output_dir = './ships_processed/'  # 新的保存文件夹

# # 2. 确保输出文件夹存在
# os.makedirs(output_dir, exist_ok=True)

# csv_files = [f for f in os.listdir(ships_dir) if f.endswith('.csv')]
# total_max = 0

# for csv_file in csv_files:
#     file_path = os.path.join(ships_dir, csv_file)
#     # 加载数据
#     data = pd.read_csv(file_path)
    
#     # 提取需要的特征
#     selected_data = data[['from_yard', 'from_bay', 'from_col', 'from_layer', 'Unit POD','Unit Weight (kg)']]

#     numeric_cols = ['Unit Weight (kg)']
#     categorical_cols = ['from_yard', 'from_bay', 'from_col', 'from_layer', 'Unit POD']

#     # 对数值型特征进行归一化处理
#     scaler = MinMaxScaler()
#     scaled_numeric = scaler.fit_transform(selected_data[numeric_cols])
#     scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
    
#     # 对非数值型特征进行独热编码
#     encoder = OneHotEncoder(sparse_output=False)
#     encoded_categorical = encoder.fit_transform(selected_data[categorical_cols])
#     encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

#     # 合并归一化后的数值型特征和编码后的非数值型特征
#     processed_data = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)

#     # 计算簇的数量
#     n_clusters = max(1, int(len(data) / 50))
#     # 使用 KMeans 进行聚类
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     data['cluster'] = kmeans.fit_predict(processed_data)

#     # 统计信息
#     cluster_counts = data['cluster'].value_counts()
#     max_cluster_samples = cluster_counts.max()
#     total_max = max(max_cluster_samples, total_max)
#     print(f"文件 {csv_file}, 簇的数量 {n_clusters}, 簇内最大样本数量 {max_cluster_samples}")

#     # 插入 order 列
#     if 'order' not in data.columns:
#         data.insert(0, 'order', range(len(data)))

#     # --- 关键修改：保存到新路径 ---
#     save_path = os.path.join(output_dir, csv_file)
#     data.to_csv(save_path, index=False)
#     print(f"处理完成，已保存至: {save_path}")

# print(f"\n全部处理完毕，历史最大簇样本数为: {total_max}")



# import json

# file_path = './test_statistics.json'

# try:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     total_clusters = sum(item['num_clusters'] for item in data.get('clustering_stats', []))
    
#     print(f"文件: {file_path}")
#     print(f"总 num_clusters 数: {total_clusters}")
#     print(f"处理的文件数: {len(data.get('clustering_stats', []))}")

# except Exception as e:
#     print(f"发生错误: {e}")




import os
import pandas as pd
import numpy as np
from core.metrics import evaluate_correlation_metrics

def main():
    ships_dir = './data/ships_test'
    if not os.path.exists(ships_dir):
        print(f"Error: Directory {ships_dir} does not exist.")
        return

    csv_files = [f for f in os.listdir(ships_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {ships_dir}.")
        return

    results = []
    for filename in sorted(csv_files):
        file_path = os.path.join(ships_dir, filename)
        group_key = os.path.splitext(filename)[0]
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"Skipping {filename}: DataFrame is empty.")
                continue
            
            # Preprocess columns that might be strings to avoid np.unique axis error in metrics.py
            # The columns used for grouping in compute_rehandle_rate are:
            # from_yard, from_bay, from_col, Unit POD
            for col in ['from_yard', 'from_bay', 'from_col', 'Unit POD']:
                if col in df.columns:
                    df[col] = pd.factorize(df[col])[0]
                
            # Generate 'pred' column as a random permutation
            # Use 'order' as a template for unique values if available, otherwise just 0 to n-1
            n = len(df)
            pred_values = np.random.permutation(n)
            df['pred'] = pred_values
            
            # Call evaluate_correlation_metrics
            # Returns: (kendall_corr, rho, rehandle_rate, rehandle_rate_GT)
            metrics = evaluate_correlation_metrics(df, group_key)
            results.append((group_key, *metrics))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Optional: Summary of all files
    if results:
        summary_df = pd.DataFrame(results, columns=['Ship', 'Kendall', 'Spearman', 'Rehandle_Rate_Pred', 'Rehandle_Rate_GT'])
        print("\n=== Summary Evaluation Results ===")
        print(summary_df.describe())
        
if __name__ == "__main__":
    main()

