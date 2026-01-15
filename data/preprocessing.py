"""
数据预处理模块
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, OTHER_FEATURES


def preprocess_group(df, continuous_features=None, categorical_features=None, other_features=None):
    """
    对数据组进行预处理：数值型标准化，类别型编码映射
    
    Args:
        df: 原始 DataFrame
        continuous_features: 连续型特征列表
        categorical_features: 类别型特征列表
        other_features: 其他需要保留的特征列表
    
    Returns:
        processed_df: 预处理后的 DataFrame
    """
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if other_features is None:
        other_features = OTHER_FEATURES
        
    processed_df = pd.DataFrame(index=df.index)
    
    # 保留其他特征
    for col in other_features:
        if col in df.columns:
            processed_df[col] = df[col]
            
    # 数值型标准化
    local_scaler = StandardScaler()
    scaled_continuous = local_scaler.fit_transform(df[continuous_features])
    for i, col_name in enumerate(continuous_features):
        processed_df[col_name] = scaled_continuous[:, i]
        
    # 类别型特征处理
    # 方案：组合位置ID + 保留层级数值
    # 目标：让模型学习到"在同一个垂直位置(yard+bay+col)下，优先放在下层(layer-1)"
    
    # 1. 创建垂直位置ID（yard + bay + col 的组合）
    # 这个ID标识了一个唯一的"垂直堆叠位置"
    position_hierarchy = ['from_yard', 'from_bay', 'from_col', 'from_layer']
    has_position_features = all(feat in categorical_features for feat in position_hierarchy)
    
    if has_position_features:
        # 创建组合位置ID：yard_bay_col
        # 例如: "1A_47_10" 表示堆场1A的第47贝位第10列
        df['position_id'] = (df['from_yard'].astype(str) + '_' + 
                            df['from_bay'].astype(str) + '_' + 
                            df['from_col'].astype(str))
        
        # 将组合位置ID映射为整数
        unique_positions = df['position_id'].unique()
        position_mapping = {pos: i + 1 for i, pos in enumerate(unique_positions)}
        position_mapping['[UNK]'] = 0
        processed_df['position_id'] = df['position_id'].map(position_mapping).fillna(0).astype(int)
    
    # 2. 处理各个特征
    string_categorical_features = ['Unit POD', 'from_yard']
    numeric_position_features = ['from_bay', 'from_col', 'from_layer']
    
    for col in categorical_features:
        if col in string_categorical_features:
            # 字符型特征：进行类别映射
            unique_categories = df[col].unique()
            mapping = {cat: i + 1 for i, cat in enumerate(unique_categories)}
            mapping['[UNK]'] = 0
            processed_df[col] = df[col].map(mapping).fillna(0).astype(int)
        elif col in numeric_position_features:
            # 数值型位置特征：保留原始值
            # 特别重要：from_layer 保留原值，让模型能学习 layer-1 的关系
            processed_df[col] = df[col].fillna(0).astype(int)
        else:
            # 其他特征：根据类型处理
            if df[col].dtype == 'object':
                unique_categories = df[col].unique()
                mapping = {cat: i + 1 for i, cat in enumerate(unique_categories)}
                mapping['[UNK]'] = 0
                processed_df[col] = df[col].map(mapping).fillna(0).astype(int)
            else:
                processed_df[col] = df[col].fillna(0).astype(int)
        
    return processed_df
