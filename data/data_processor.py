import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import single_batch_graph_data
import pdb

root_dir = "./data/container_data2.pkl"
output_filename = "./data/processed_container_data2.pkl"

continuous_features = ['Unit Weight (kg)']
categorical_features = ['Unit POD', 'from_yard', 'from_bay', 'from_col', 'from_layer', ]


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pd.read_pickle(f)
    return data

data = read_pkl(root_dir)
data = {tuple(key) if isinstance(key, np.ndarray) else key: value for key, value in data.items()}


print("\n--- Starting LOCAL processing for each DataFrame ---")
processed_data_local = {}
# 用于存储每个DataFrame独立的词汇表，以供检查
local_vocabs_for_inspection = {}

pdb.set_trace()
"""原始建图方法"""
for key, df in data.items():
    #print(f"\nProcessing data locally for key: {key}")
    processed_df = pd.DataFrame(index=df.index)
    if 'Unit Nbr' in df.columns:
        processed_df['Unit Nbr'] = df['Unit Nbr']
    if 'Time Completed' in df.columns:
        processed_df['Time Completed'] = df['Time Completed']
    
    # --- 2a: 在当前DataFrame上学习并转换连续特征 ---
    # !!! Scaler在每次循环中都重新创建和学习 !!!
    local_scaler = StandardScaler()
    scaled_continuous = local_scaler.fit_transform(df[continuous_features])
    for i, col_name in enumerate(continuous_features):
        processed_df[col_name] = scaled_continuous[:, i]

    # --- 2b: 在当前DataFrame上构建局部词汇表并转换类别特征 ---
    # !!! 词汇表在每次循环中都重新构建 !!!
    local_vocab_mappings = {}
    for col in categorical_features:
        unique_categories = df[col].unique()
        # 构建只包含当前DataFrame类别的局部词汇表
        local_vocab_mappings[col] = {category: i + 1 for i, category in enumerate(unique_categories)}
        local_vocab_mappings[col]['[UNK]'] = 0 # 同样为未知类别保留0
        
        # 应用局部词汇表
        processed_df[col] = df[col].map(local_vocab_mappings[col]).fillna(0).astype(int)
    
    # 存储处理结果和该次处理的词汇表
    processed_data_local[key] = processed_df
    local_vocabs_for_inspection[key] = local_vocab_mappings
   
    batch_graph = single_batch_graph_data(processed_df.iloc[:, -6:].to_numpy())

    processed_data_local[key] = {
        'data':processed_df,
        'graph':batch_graph
    }



print("\n--- Step 3: Verifying the locally processed data ---")
key1 = list(data.keys())[0]


with open(output_filename, 'wb') as f:
    pickle.dump(processed_data_local, f)

print(f"\n--- Step 4: Saving Data ---")
print(f"Processed data has been successfully saved to '{output_filename}'")

# (可选) 验证一下是否能成功读回保存的文件
print("\nVerifying the saved file...")
with open(output_filename, 'rb') as f:
    reloaded_data = pickle.load(f)

# print("File reloaded successfully.")
# print(f"Data for key '{key1}' in reloaded file:")
# print(reloaded_data[key1].head())
# print(reloaded_data[key1].shape)
