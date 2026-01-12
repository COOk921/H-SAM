# 集装箱数据聚类与异构图构建流水线

本模块用于将原始集装箱 CSV 数据转换为适合图神经网络 (GNN) 使用的异构图数据格式。

## 📁 文件结构

```
cluster/
├── config.py              # 配置参数（路径、特征列、聚类算法、超参数）
├── clustering.py          # 聚类模块（谱聚类 + K-means）
├── preprocessing.py       # 预处理模块（特征标准化、类别编码）
├── graph_builder.py       # 图构建模块（异构图边构建）
├── pipeline.py            # 主流水线（完整处理流程）
├── visualize_clusters.py  # 聚类可视化工具
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 配置聚类算法

在 `config.py` 中选择聚类算法：

```python
# 可选值: 'spectral' (谱聚类) 或 'kmeans' (K-means)
CLUSTERING_METHOD = 'spectral'
```

### 2. 运行完整流水线

处理 `ships/` 目录下的所有 CSV 文件，生成异构图数据：

```bash
cd /Users/liuyang/Documents/example/CORL/data/cluster
python pipeline.py
```

**输出文件：**
- `processed_container_data_hetero.pkl` - 处理后的异构图数据
- `pipeline_statistics.json` - 聚类和图构建统计信息（包含使用的聚类算法）

### 3. 聚类可视化

对指定的 CSV 文件进行聚类并生成 2D 可视化图：

```bash
# 使用配置文件中的默认算法
python visualize_clusters.py AILK_009.csv

# 显式指定使用谱聚类
python visualize_clusters.py AILK_009.csv -m spectral

# 显式指定使用 K-means
python visualize_clusters.py AILK_009.csv -m kmeans

# 指定输出路径和 t-SNE 参数
python visualize_clusters.py AILK_009.csv -o ./my_viz.png -p 50
```

**命令行参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `csv_file` | CSV 文件路径或文件名 | (必填) |
| `-o, --output` | 输出图片路径 | 自动生成 |
| `-p, --perplexity` | t-SNE 困惑度参数 | 30 |
| `-m, --method` | 聚类方法 (spectral/kmeans) | 使用配置文件 |

**输出文件：**
- `visualizations/<csv_name>_<method>_clusters.png` - 聚类可视化图片

## 📊 聚类算法对比

### 谱聚类 (Spectral Clustering)

使用自定义亲和矩阵进行聚类，亲和度计算公式：

```
A_ij = sim_p(i, j) × [α × sim_l(i, j) + β × sim_w(i, j)]
```

| 相似度 | 含义 | 计算方式 |
|--------|------|----------|
| `sim_p` | 航次相似度 | 相同航次为 1，否则为 0 |
| `sim_w` | 重量相似度 | 高斯核（带宽由中位数启发式确定） |
| `sim_l` | 位置相似度 | 高斯核（带宽由中位数启发式确定） |

**优点**：能够捕捉复杂的非线性关系，适合发现任意形状的簇  
**缺点**：计算复杂度较高 O(n³)，不适合超大规模数据

### K-means

基于距离的经典聚类算法，直接在特征空间中进行聚类。

**特征**：重量（标准化） + 位置编码（标准化）

**优点**：计算速度快 O(nk)，适合大规模数据  
**缺点**：假设簇为凸形，可能无法捕捉复杂结构

### 算法选择建议

| 场景 | 推荐算法 |
|------|----------|
| 数据量小 (<5000)，需要高精度 | `spectral` |
| 数据量大，需要快速处理 | `kmeans` |
| 集装箱位置关系复杂 | `spectral` |
| 快速原型验证 | `kmeans` |

## ⚙️ 配置说明

编辑 `config.py` 修改配置：

```python
# --- 聚类算法配置 ---
CLUSTERING_METHOD = 'spectral'  # 'spectral' 或 'kmeans'

# --- 谱聚类超参数 ---
ALPHA = 0.5  # 位置相似度权重
BETA = 0.5   # 重量相似度权重

# --- K-means 超参数 ---
KMEANS_MAX_ITER = 300  # 最大迭代次数
KMEANS_N_INIT = 10     # 初始化次数

# --- 图构建超参数 ---
KNN_K = 3  # KNN 邻居数
```

## 🔧 异构图边类型

| 边类型 | 含义 | 构建规则 |
|--------|------|----------|
| `blocks` | 堆叠关系 | 同一堆垛中，上层指向下层 |
| `spatial` | 空间关系 | 同层不同排的集装箱之间 |
| `similar` | 相似关系 | 基于 KNN 的特征相似性 |

## 📈 输出数据格式

### processed_container_data_hetero.pkl

```python
{
    (voyage, cluster_id): {
        'data': pd.DataFrame,       # 预处理后的特征数据
        'graph': HeteroData         # PyG 异构图对象
    },
    ...
}
```

### pipeline_statistics.json

```json
{
    "clustering_method": "spectral",
    "clustering_method_name": "谱聚类 (Spectral Clustering)",
    "clustering_stats": [...],
    "graph_stats": [...],
    "summary": {
        "total_csv_files": 10,
        "total_graphs": 150,
        "avg_nodes_per_graph": 45.2,
        ...
    }
}
```

## 🔗 依赖项

```
pandas
numpy
torch
torch_geometric
scikit-learn
scipy
matplotlib
tqdm
```

## 📝 注意事项

1. **Ground Truth 保留**：原始 CSV 的行顺序通过 `order` 列保留，可用于后续评估
2. **GPU 加速**：如果有 CUDA 可用，图构建会自动使用 GPU
3. **中位数启发式**：谱聚类的高斯核带宽会自动根据数据分布计算，无需手动调参
4. **聚类数量**：默认动态确定为 `N / 50`，可在 `clustering.py` 中修改
