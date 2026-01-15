"""
数据集划分脚本：将 ships 文件夹中的 CSV 文件随机划分为训练集和测试集
"""

import os
import shutil
import random

# 配置参数
SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'ships')
TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'ships_train')
TEST_DIR = os.path.join(os.path.dirname(__file__), 'ships_test')

TRAIN_RATIO = 0.8  # 训练集比例
RANDOM_SEED = 42   # 随机种子，保证可复现


def split_data():
    """
    将 ships 目录下的 CSV 文件随机划分为训练集和测试集
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    if total_files == 0:
        print("错误：ships 目录中没有找到 CSV 文件！")
        return
    
    # 随机打乱
    random.shuffle(csv_files)
    
    # 计算划分点
    split_idx = int(total_files * TRAIN_RATIO)
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]
    
    # 创建目标目录
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # 复制文件到训练集目录
    print(f"正在复制 {len(train_files)} 个文件到训练集目录...")
    for f in train_files:
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(TRAIN_DIR, f)
        shutil.copy2(src, dst)
    
    # 复制文件到测试集目录
    print(f"正在复制 {len(test_files)} 个文件到测试集目录...")
    for f in test_files:
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(TEST_DIR, f)
        shutil.copy2(src, dst)
    
    # 输出统计信息
    print("\n划分完成！")
    print(f"  - 原始文件总数: {total_files}")
    print(f"  - 训练集文件数: {len(train_files)} ({len(train_files)/total_files*100:.1f}%)")
    print(f"  - 测试集文件数: {len(test_files)} ({len(test_files)/total_files*100:.1f}%)")
    print(f"  - 训练集目录: {TRAIN_DIR}")
    print(f"  - 测试集目录: {TEST_DIR}")
    print(f"  - 随机种子: {RANDOM_SEED}")


if __name__ == "__main__":
    split_data()
