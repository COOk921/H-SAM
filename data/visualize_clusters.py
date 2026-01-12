"""
聚类可视化模块：对指定的 CSV 文件进行聚类并降维可视化
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import INPUT_DIR, VISUALIZATION_OUTPUT_DIR, CLUSTERING_METHOD
from clustering import get_cluster_labels, get_clustering_features, get_method_name


def visualize_clusters(csv_path, output_path=None, perplexity=30, random_state=42, method=None):
    """
    对指定 CSV 文件进行聚类，使用 t-SNE 降维到 2D 并可视化
    
    Args:
        csv_path: CSV 文件路径
        output_path: 输出图片路径，None 时自动生成
        perplexity: t-SNE 困惑度参数
        random_state: 随机种子
        method: 聚类方法 ('spectral' 或 'kmeans')，None 时使用配置
    
    Returns:
        output_path: 保存的图片路径
    """
    # 确定使用的聚类方法
    if method is None:
        method = CLUSTERING_METHOD
    method_display_name = get_method_name(method)
    
    # 读取数据
    print(f"正在读取: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 添加 order 列（如果不存在）
    if 'order' not in df.columns:
        df['order'] = range(len(df))
    
    # 进行聚类
    print(f"正在使用 {method_display_name} 进行聚类...")
    cluster_labels = get_cluster_labels(df, method=method)
    df['cluster'] = cluster_labels
    n_clusters = len(np.unique(cluster_labels))
    print(f"聚类完成，共 {n_clusters} 个簇")
    
    # 获取特征矩阵
    features = get_clustering_features(df)
    
    # t-SNE 降维
    print("正在进行 t-SNE 降维...")
    # 调整 perplexity 以适应小数据集
    actual_perplexity = min(perplexity, len(df) - 1)
    if actual_perplexity < 5:
        actual_perplexity = 5
    
    tsne = TSNE(
        n_components=2, 
        perplexity=actual_perplexity, 
        random_state=random_state,
        max_iter=1000
    )
    features_2d = tsne.fit_transform(features)
    
    # 可视化
    print("正在生成可视化...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用更好的配色方案
    cmap = plt.cm.get_cmap('tab20' if n_clusters <= 20 else 'viridis')
    
    scatter = ax.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=cluster_labels, 
        cmap=cmap,
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )
    
    # 添加图例
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=12)
    
    # 标题和标签
    csv_name = os.path.basename(csv_path).replace('.csv', '')
    ax.set_title(f'{method_display_name} Visualization\n{csv_name} ({len(df)} samples, {n_clusters} clusters)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加聚类统计信息
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    stats_text = f"Cluster Distribution:\n"
    for i, (cluster_id, count) in enumerate(cluster_counts.items()):
        if i < 10:  # 只显示前 10 个
            stats_text += f"  Cluster {cluster_id}: {count} samples\n"
    if len(cluster_counts) > 10:
        stats_text += f"  ... ({len(cluster_counts) - 10} more clusters)"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 确定输出路径
    if output_path is None:
        os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, 
            f"{csv_name}_{method}_clusters.png"
        )
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化已保存到: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='对指定 CSV 文件进行聚类可视化'
    )
    parser.add_argument(
        'csv_file', 
        type=str, 
        help='CSV 文件路径或文件名 (如果在 ships/ 目录下可只写文件名)'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default=None,
        help='输出图片路径 (默认自动生成)'
    )
    parser.add_argument(
        '-p', '--perplexity', 
        type=int, 
        default=30,
        help='t-SNE 困惑度参数 (默认: 30)'
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['spectral', 'kmeans'],
        default=None,
        help='聚类方法: spectral (谱聚类) 或 kmeans (默认使用配置文件设置)'
    )
    
    args = parser.parse_args()
    
    # 处理 CSV 路径
    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        # 尝试在 ships/ 目录下查找
        alt_path = os.path.join(INPUT_DIR, csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            print(f"错误: 文件不存在 - {csv_path}")
            return
    
    visualize_clusters(csv_path, args.output, args.perplexity, method=args.method)


if __name__ == "__main__":
    main()
