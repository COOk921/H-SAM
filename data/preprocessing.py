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
        
    # 类别型映射
    for col in categorical_features:
        unique_categories = df[col].unique()
        mapping = {cat: i + 1 for i, cat in enumerate(unique_categories)}
        mapping['[UNK]'] = 0
        processed_df[col] = df[col].map(mapping).fillna(0).astype(int)
        
    return processed_df
