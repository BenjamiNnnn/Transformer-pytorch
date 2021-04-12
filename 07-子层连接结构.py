import torch
import torch.nn as nn
from 规范化层 import LayerNorm

'''
        子层连接结构
'''

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        # size:  词嵌入维度的大小
        super(SublayerConnection, self).__init__()
        # 实例化规范化对象
        self.norm = LayerNorm(size)
        self.dpot = nn.Dropout(p=dropout)

    def forward(self,x,sublayer):
        '''
        接受上一个层或者子层的输入作为第一个参数
        将该子层连接中的子层函数作为第二个参数
        :param x:
        :param sublayer:
        :return:
        '''
        # 首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作
        # 随机停止一些网络中神经元的作用 防止过拟合
        # 残差连接
        return x + self.dpot(sublayer(self.norm(x)))