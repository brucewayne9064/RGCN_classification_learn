import torch
import torch.nn as nn
from RGCN import RGCNLayer
from functools import partial
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        # 先初始化参数
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        # 具体看下面函数
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        # torch.arange(start=0, end=5)的结果并不包含end，start默认是0
        features = torch.arange(self.num_nodes)
        return features

    # 输入num_nodes的独热编号
    # 输出h_dim
    # 激活函数是relu
    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    # 输入h_dim
    # 输出h_dim
    # 激活函数是relu
    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    # 输入h_dim
    # 输出out_dim
    # 激活函数是softmax后归一化
    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features  # 把图结点的id作为图结点的特征
        for layer in self.layers:
            layer(g)
        # 取出每个节点的隐藏层并且删除"h"特征，方便下一次进行训练
        return g.ndata.pop('h')
