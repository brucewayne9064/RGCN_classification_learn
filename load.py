import torch

# load graph data,8285个结点，只有140个有label，关系数91，分类数4，边数为65439
from dgl.contrib.data import load_data
data = load_data(dataset='aifb')
num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx  # 140个结点的编号
# split training and validation set
val_idx = train_idx[:len(train_idx) // 5]#前20%做验证
train_idx = train_idx[len(train_idx) // 5:]#剩下做训练

# edge type and normalization factor
# 获取边的类型
edge_type = torch.from_numpy(data.edge_type)  # 每一条边的关系类型
# 获取边的标准化因子
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

labels = torch.from_numpy(labels).view(-1)
