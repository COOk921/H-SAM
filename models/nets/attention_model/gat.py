from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear


class GAT(nn.Module):
    def __init__(self, in_channels, embed_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3, heads=2):
       
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
       
        self.node_emb = nn.Linear(in_channels, out_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(out_dim, hidden_dim, heads=self.heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * self.heads, hidden_dim, heads=self.heads))
        if num_layers > 1:
            # 最后一层 GAT 不使用多头，所以 heads 设置为 1
            self.convs.append(GATConv(hidden_dim * self.heads, out_dim, heads=1))
        else:  
            self.convs[0] = GATConv(embed_dim, out_dim, heads=1)

    def forward(self, batch):

        node_features = batch.x
        edge_index = batch.edge_index
        
        update_node_feature = self.node_emb(node_features)  
        init_h =  update_node_feature
       
        for i, conv in enumerate(self.convs):
            update_node_feature = conv(update_node_feature, edge_index)
            if i != len(self.convs) - 1:
                update_node_feature = F.elu(update_node_feature)
                update_node_feature = F.dropout(update_node_feature, p=self.dropout, training=self.training)

        # update_node_feature = update_node_feature  + self.node_emb(init_h)
        return update_node_feature


# class ContainerHeteroGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
        
#         # 定义异构卷积
#         self.conv1 = HeteroConv({
#             # blocks: 强逻辑，使用多头注意力，且不需要加 self-loop (因为是压迫关系)
#             ('container', 'blocks', 'container'): GATConv(in_channels, hidden_channels, heads=4, add_self_loops=False),
            
#             # spatial: 弱逻辑，双向，可以用较少的头
#             ('container', 'spatial', 'container'): GATConv(in_channels, hidden_channels, heads=2, add_self_loops=True),
            
#             # similar: 语义补充
#             ('container', 'similar', 'container'): GATConv(in_channels, hidden_channels, heads=2, add_self_loops=True),
#         }, aggr='sum') # 将三种边的聚合结果相加
        
        
#         self.conv_refined = HeteroConv({
#             ('container', 'blocks', 'container'): GATConv(in_channels, hidden_channels, heads=4, concat=False),
#             ('container', 'spatial', 'container'): GATConv(in_channels, hidden_channels, heads=4, concat=False),
#             ('container', 'similar', 'container'): GATConv(in_channels, hidden_channels, heads=4, concat=False),
#         }, aggr='sum')

#         self.lin = Linear(hidden_channels, out_channels)

#     def forward(self, data):
#         x_dict = data.x_dict
#         edge_index_dict = data.edge_index_dict
        
#         # 卷积
#         x_dict = self.conv_refined(x_dict, edge_index_dict)
#         x = x_dict['container']
#         x = torch.relu(x)
        
#         out = self.lin(x)
#         return out
        
class ContainerHeteroGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # --- 第一层 GAT ---
        # 目标：提取特征，使用多头注意力并拼接 (concat=True)
        # 注意：为了使用 aggr='sum'，所有分支的输出维度必须一致！
        # 输出维度 = hidden_channels * heads
        self.conv1 = HeteroConv({
            # blocks: 强逻辑，不加自环
            ('container', 'blocks', 'container'): GATConv(
                in_channels, hidden_channels, 
                heads=4, add_self_loops=False, concat=True
            ),
            
            # spatial: 弱逻辑，加自环
            # 必须改为 heads=4 以匹配 blocks 的输出维度，否则无法 sum
            ('container', 'spatial', 'container'): GATConv(
                in_channels, hidden_channels, 
                heads=4, add_self_loops=True, concat=True
            ),
            
            # similar: 语义补充
            ('container', 'similar', 'container'): GATConv(
                in_channels, hidden_channels, 
                heads=4, add_self_loops=True, concat=True
            ),
        }, aggr='sum') 

        # --- 第二层 GAT (Refined) ---
        # 输入维度是上一层的输出： hidden_channels * 4 (因为上一层 heads=4)
        # 目标：整合特征，通常最后一层 GNN 不拼接 (concat=False)，做平均
        self.conv_refined = HeteroConv({
            ('container', 'blocks', 'container'): GATConv(
                hidden_channels * 4, hidden_channels, 
                heads=4, concat=False, add_self_loops=False
            ),
            ('container', 'spatial', 'container'): GATConv(
                hidden_channels * 4, hidden_channels, 
                heads=4, concat=False, add_self_loops=True
            ),
            ('container', 'similar', 'container'): GATConv(
                hidden_channels * 4, hidden_channels, 
                heads=4, concat=False, add_self_loops=True
            ),
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        # Data 已经是 HeteroData
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        
        # 1. 第一层卷积
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        
        # 3. 第二层卷积
        x_dict = self.conv_refined(x_dict, edge_index_dict)
        
        # 4. 提取 container 节点的特征
        x = x_dict['container']

        # 5. 再次激活 (可选，看你是否需要在 Linear 前再做一次非线性)
        x = torch.relu(x)
        
        # 6. 输出层
        out = self.lin(x)
        
        return out



# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# import torch
# import pdb

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GATConv

# class GATLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, heads, dropout=0.2, feed_forward_hidden=512):
#         super().__init__()
#         # GAT Attention部分
#         self.gat_conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout)
        
#         # Add & Norm
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(out_dim * heads)
        
#         # Add & Norm
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(out_dim * heads)

#     def forward(self, x, edge_index):

#         attended_x = self.gat_conv(x, edge_index)
      
#         return x


# class GAT(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_dim, num_layers=3, dropout=0.2, heads=2, feed_forward_hidden=256):
        
#         super().__init__()
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.input_proj = nn.Linear(in_channels, hidden_dim * heads)
        
#         self.gat_layers = nn.ModuleList()
#         for _ in range(num_layers -1):
#             self.gat_layers.append(
#                 GATLayer(
#                     in_dim= hidden_dim * heads, 
#                     out_dim=hidden_dim, 
#                     heads=heads,
#                     dropout=dropout,
#                     feed_forward_hidden=feed_forward_hidden
#                 )
#             )

#         self.final_gat = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout)
        
#     def forward(self, batch):
#         node_features = batch.x
#         edge_index = batch.edge_index

#         h = node_features
#         h = self.input_proj(node_features) # ->hidden_dim*heads 128*2=256

#         for layer in self.gat_layers:
#             h = layer(h, edge_index)
            
#         h_out = self.final_gat(h, edge_index)
        
#         return h_out

