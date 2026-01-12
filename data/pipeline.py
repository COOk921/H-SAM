"""
主流水线模块：整合聚类、预处理和图构建
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from config import (
    INPUT_DIR, OUTPUT_FILE, OUTPUT_STATS_FILE,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, OTHER_FEATURES, FEATURES_FOR_GRAPH,
    CLUSTERING_METHOD
)
from clustering import get_cluster_labels, get_method_name
from preprocessing import preprocess_group
from graph_builder import build_hetero_graph


def run_pipeline():
    """
    运行完整的数据处理流水线：
    1. 读取 ships/ 目录下的 CSV
    2. 使用谱聚类添加 cluster 标签
    3. 按航次和 cluster 进行嵌套切分
    4. 对每个组进行预处理和异构图构建
    5. 保存结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    # 获取聚类方法名称
    method_name = get_method_name(CLUSTERING_METHOD)
    print(f"\n当前聚类算法: {method_name}\n")
    
    # 初始化统计信息
    statistics = {
        'clustering_method': CLUSTERING_METHOD,
        'clustering_method_name': method_name,
        'clustering_stats': [],
        'graph_stats': [],
        'summary': {}
    }
    
    # --- 步骤 1: 读取和聚类 ---
    all_dfs = []
    print(f"步骤 1/3: 正在读取和聚类 {len(csv_files)} 个文件...")
    for f in tqdm(csv_files):
        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path)
        
        # 添加 order 列
        if 'order' not in df.columns:
            df['order'] = range(len(df))
        
        # 聚类
        df['cluster'] = get_cluster_labels(df)
        
        # 记录统计信息
        cluster_counts = df['cluster'].value_counts()
        cluster_stats = {
            'csv_filename': f,
            'total_samples': int(len(df)),
            'num_clusters': int(len(cluster_counts)),
            'samples_per_cluster_mean': float(cluster_counts.mean()),
            'samples_per_cluster_median': float(cluster_counts.median()),
            'samples_per_cluster_min': int(cluster_counts.min()),
            'samples_per_cluster_max': int(cluster_counts.max()),
            'samples_per_cluster_std': float(cluster_counts.std())
        }
        statistics['clustering_stats'].append(cluster_stats)
        
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- 步骤 2: 数据切分 ---
    print("步骤 2/3: 正在进行数据切分...")
    grouped = combined_df.groupby(['Unit O/B Actual Visit', 'cluster'])
    #grouped = combined_df.groupby(['cluster'])
    # --- 步骤 3: 预处理和图构建 ---
    final_results = {}
    print(f"步骤 3/3: 正在预处理每个组并构建异构图 (总计 {len(grouped)} 组)...")
    for key, group_df in tqdm(grouped):
        group_key = tuple(key) if isinstance(key, (list, np.ndarray)) else key
        
        # 预处理
        preprocessed_df = preprocess_group(
            group_df, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, OTHER_FEATURES
        )
        
        # 构建图
        graph = build_hetero_graph(preprocessed_df, FEATURES_FOR_GRAPH, device)
        
        # 记录统计信息
        graph_stats = {
            'group_key': str(group_key),
            'voyage': str(key[0]) if isinstance(key, (tuple, list)) else str(key),
            'cluster_id': int(key[1]) if isinstance(key, (tuple, list)) and len(key) > 1 else None,
            'num_nodes': int(graph['container'].x.shape[0]),
            'num_edges_blocks': int(graph['container', 'blocks', 'container'].edge_index.shape[1]),
            'num_edges_spatial': int(graph['container', 'spatial', 'container'].edge_index.shape[1]),
            'num_edges_similar': int(graph['container', 'similar', 'container'].edge_index.shape[1]),
            'total_edges': int(
                graph['container', 'blocks', 'container'].edge_index.shape[1] +
                graph['container', 'spatial', 'container'].edge_index.shape[1] +
                graph['container', 'similar', 'container'].edge_index.shape[1]
            ),
            'num_features': int(graph['container'].x.shape[1])
        }
        statistics['graph_stats'].append(graph_stats)
        
        final_results[group_key] = {
            'data': preprocessed_df,
            'graph': graph
        }
    
    # --- 汇总统计 ---
    statistics['summary'] = {
        'total_csv_files': int(len(csv_files)),
        'total_graphs': int(len(final_results)),
        'total_samples': int(len(combined_df)),
        'avg_clusters_per_csv': float(np.mean([s['num_clusters'] for s in statistics['clustering_stats']])),
        'avg_nodes_per_graph': float(np.mean([s['num_nodes'] for s in statistics['graph_stats']])),
        'avg_edges_per_graph': float(np.mean([s['total_edges'] for s in statistics['graph_stats']])),
        'total_nodes_all_graphs': int(sum([s['num_nodes'] for s in statistics['graph_stats']])),
        'total_edges_all_graphs': int(sum([s['total_edges'] for s in statistics['graph_stats']]))
    }
    
    # --- 保存结果 ---
    print(f"正在保存最终结果到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"正在保存统计信息到 {OUTPUT_STATS_FILE}...")
    with open(OUTPUT_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    # --- 输出摘要 ---
    print("完成！")
    print(f"\n统计摘要:")
    print(f"  - 聚类算法: {method_name}")
    print(f"  - 处理的CSV文件数: {statistics['summary']['total_csv_files']}")
    print(f"  - 生成的异构图数: {statistics['summary']['total_graphs']}")
    print(f"  - 总样本数: {statistics['summary']['total_samples']}")
    print(f"  - 平均每个CSV的聚类数: {statistics['summary']['avg_clusters_per_csv']:.2f}")
    print(f"  - 平均每个图的节点数: {statistics['summary']['avg_nodes_per_graph']:.2f}")
    print(f"  - 平均每个图的边数: {statistics['summary']['avg_edges_per_graph']:.2f}")


if __name__ == "__main__":
    run_pipeline()
