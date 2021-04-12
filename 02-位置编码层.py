'''
        Transformer结构中没有针对词汇位置信息的处理，因此需要在Embedding最后加入位置编码器
        将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失
'''
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

class PositionalEnconding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        '''

        :param d_model: 词嵌入维度
        :param dropout: 丢失率
        :param max_len: 每个句子的最长长度
        '''
        super(PositionalEnconding,self).__init__()
        # 实例化dropout层
        self.dpot = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len,d_model)

        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0,max_len).unsqueeze(1)

        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze((0))

        # 把pe位置编码矩阵注册成模型的buffer
        # 模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''

        :param x: 文本的词嵌入表示
        :return:
        '''

        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dpot(x)


if __name__ == '__main__':

    plt.figure(figsize=(15,5))

    pe = PositionalEnconding(20,0,max_len=100)

    y = pe(Variable(torch.zeros(1,100,20)))

    plt.plot(np.arange(100),y[0,:,4:8].data.numpy())

    plt.legend(["dim %d" %p for p in [4,5,6,7]])

    plt.savefig("1.jpg")