import torch
import torch.nn as nn
import dgl.function as fn


class RGCNLayer(nn.Module):
    # 参数说明：
    # in_feat 输入维度
    # out_feat 输出维度
    # num_rels 边类型数量
    # num_bases W_r分解的数量，对应原文公式3的B（求和符号的上界）
    # bias 偏置
    # activation 激活函数
    # is_input_layer 是否是输入层（第一层）
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat  # 输入的特征维度
        self.out_feat = out_feat  # 输出的特征维度
        self.num_rels = num_rels  # 关系的数量
        self.num_bases = num_bases  # 基分解中W_r分解的数量，即B的大小
        self.bias = bias  # 是否带偏置b
        self.activation = activation
        self.is_input_layer = is_input_layer
        # sanity check
        # 矩阵分解的参数校验条件：不能小于0，不能比现有维度大（复杂度会变高，参数反而增加）
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        # 这里是根据公式3把W_r算出来，用V_b（weight）表示，共有num_bases个V_b累加得到
        # 得到的结果是Tensor，因此用 nn.Parameter将一个不可训练的类型Tensor
        # 转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # num_bases * (in_feat * out_feat)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))

        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            # 这里的w_comp是公式3里面的a_{rb}
            # 一个边类型对应一个W_r（那么就一共有num_rels种W_r），每个W_r分解为num_bases个组合
            # 因此w_comp这里的维度就是num_rels×num_bases
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        # 偏置要进行加法运算其维度要和输出维度大小一样
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        # 这里用的是xavier初始化，可以查下为什么
        # 因为我们用的是线性激活函数，网上有说用ReLU和Leaky ReLU
        # 可以考虑用别的初始化方法：He init 。
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:  # 分解就走公式3,如果B < 关系数
            # generate all weights from bases (equation (3))
            # weight的维度 = num_bases * in_feat * out_feat --> in_feat * num_bases * out_feat
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            # w_comp的维度 => num_rels * num_bases
            # torch.matmul(self.w_comp, weight)
            # w_comp(num_rels * num_bases)  weight(in_feat * num_bases * out_feat)
            #                             ||
            #                             V
            #              in_feat * num_rels * out_feat
            # 再经过view操作
            # weight --> num_rels * in_feat * out_feat
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat, self.out_feat)
        else:  # 不分解就直接用weight算
            weight = self.weight

        if self.is_input_layer:  # 如果是第一层
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                # 对于第一层，输入可以直接用独热编码进行aggregate
                # 信息的汇聚就可以直接写成矩阵相乘的形式

                # 就这个例子来说，weight: num_rels * in_feat * out_feat = 91 * 8285 * 16
                # embed = 753935 * 16
                # edges.data['rel_type']存的是所有的关系 共65439个关系
                # self.in_feat是输入的embedding，因为这里直接one-hot表示，因此大小为8285
                # edges.src['id']是每个关系的源节点id, 这个参数看了好久也不知道是怎么来的，我感觉可能是节点有了“id”，然后传进来便就会有这个参数了吧
                # 这个index还是很奇妙的
                # 因为一共91种关系，8285个节点，那么最开始输入需要赋值节点自身的特征
                # 那么每个节点对应的91种关系都有不同的表达
                # edges.data['rel_type'] * self.in_feat + edges.src['id']就是取自身节点对应的关系的那种表达
                # 从而获得节点的embedding表达  #对weight进行reshape，变成out_feat列，行数由列数决定
                embed = weight.view(-1, self.out_feat)  # embed维度整成out_feat维度一样
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:    # import numpy as np  a = np.array(index) 查看张量的办法
            def message_func(edges):
                # 根据边类型'rel_type'获取对应的w
                w = weight[edges.data['rel_type'].long()]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()  # 消息汇聚，就是w乘以src['h']（输入节点特征）
                # edges.src['h'].unsqueeze(1): 65439*in -> 65439*1*in
                # (65439*1*in) * (65439*in*out) -> 65439*1*out 广播，
                # 前一项提出65439，然后1*in与in*out作矩阵乘法，得1*out，然后与65439组合成65439*1*out
                # .squeeze() msg: 65439*1*out -> 65439*out
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:  # 有偏置加偏置
                h = h + self.bias
            if self.activation:  # 经过激活函数
                h = self.activation(h)
            return {'h': h}
        # 信息进行聚合(这里是求和)，并保存在节点的某一个特征里。1.计算所有邻居边给自己带来的信息msg2.聚合这些信息到点上h。
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
