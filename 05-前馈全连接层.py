import torch
import torch.nn as nn
import torch.nn.functional as F
'''
        前馈全连接层：两层全连接层
        作用：考虑注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增强模型的能力
'''

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        '''

        :param d_model: 线性层输入维度
        :param d_ff: 线性层输出维度
        :param dropout:
        '''

        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dpot = nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dpot(F.relu(self.w1(x))))

if __name__ == '__main__':
    d_model = 512
    d_ff = 64
    dropout=0.2

    ff = FeedForward(d_model,d_ff,dropout)
    x = torch.randn(3,4,512)
    re = ff(x)
    print(x)
    print(re)