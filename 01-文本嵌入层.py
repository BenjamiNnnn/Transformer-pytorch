'''
        将文本中词汇的数字表示转变为向量表示
        希望在高维空间捕捉词汇间的关系
'''
import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self,vocab,d_model):
        '''

        :param vocab:  词表的大小
        :param d_model: 词嵌入的维度
        '''
        super(Embedding,self).__init__()

        self.lut = nn.Embedding(vocab,d_model,padding_idx=0)

        self.d_model = d_model

    def forward(self,x):
        '''

        :param x: 输入给模型的文本通过词汇映射后的张量
        :return:
        '''
        return self.lut(x) * math.sqrt(self.d_model)

if __name__ == '__main__':
    d_model = 512
    vocab = 1000

    x = torch.randint(low=0,high=500,size=(5,10),dtype=torch.long)

    embe = Embedding(vocab,d_model)
    ember = embe(x)

    print(ember)
    print(ember.shape)