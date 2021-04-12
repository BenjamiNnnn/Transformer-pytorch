import torch
import torch.nn as nn
'''
        随着网络层数的增加，通过多层的计算后参数可能开始出现过大或过小的情况
        这样可能导致学习过程出现异常，模型收敛过慢
        因此添加规范化层进行数值的规范化，使其特征数值在合理范围内
'''

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        '''

        :param features:  代表词嵌入的维度
        :param eps:
        '''
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        # 防止分母为0
        self.eps = eps

    def forward(self,x):
        # 对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致
        mean = x.mean(-1,keepdim=True)
        # 接着再求最后一个维度的标准差
        std = x.std(-1,keepdim=True)
        # 然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数
        return self.a2 * (x-mean) /(std + self.eps) + self.b2

if __name__ == '__main__':
    features = d_model = 512
    eps = 1e-6

    x = torch.randn(3,4,512)
    ln = LayerNorm(features,eps)

    ln_r = ln(x)
    print(ln_r)
    print(ln_r.shape)