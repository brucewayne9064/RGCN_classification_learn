from dgl import DGLGraph
from load import data, edge_type, edge_norm, num_classes, num_rels, train_idx, labels, val_idx
from model import Model
import torch
import torch.nn.functional as F

# configurations
n_hidden = 16 # number of hidden units
n_bases = -1 # use number of relations as number of bases这里设置-1相当于没进行分解
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer#输出接输出，没有中间层
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

# create graph
g = DGLGraph((data.edge_src, data.edge_dst))  # src和dst分别指源点和目标点
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})  # 给每一条边附上对应的关系和norm
# create model
model = Model(len(g),  # 等于8285
              n_hidden,
              num_classes,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)

# optimizer优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()  # 清空梯度
    logits = model.forward(g)  # 前向传播
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())  # 计算损失
    loss.backward()  # 计算梯度

    optimizer.step()  # 反向传播

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx].long())
    val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
    val_acc = val_acc.item() / len(val_idx)
    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, loss.item()) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              val_acc, val_loss.item()))
