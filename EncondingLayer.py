import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math

# 1.文本嵌入层
class TextEmbedding(nn.Module):
    def __init__(self,vocab,d_model):
        '''

        :param vocab:  词表的大小
        :param d_model: 词嵌入的维度
        '''
        super(TextEmbedding,self).__init__()
        self.lut = nn.Embedding(vocab,d_model,padding_idx=0)
        self.d_model = d_model

    def forward(self,x):
        '''

        :param x: 输入给模型的文本通过词汇映射后的张量
        :return:
        '''
        return self.lut(x) * math.sqrt(self.d_model)

# 2.位置编码层
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

# 3.注意力机制
def subsequent_mask(size):

    atten_shape = (1,size,size)
    # 对角线下就是负 对角线上就是正 对角线就是0
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    return torch.from_numpy(1-mask)

def attention(query,key,value,mask=None,dropout=None):
    '''

    :param query:
    :param key:
    :param value:
    :param mask:  掩码张量
    :param dropout:
    :return: query在key和value作用下的表示
    '''
    # 获取query的最后一维的大小，一般情况下就等同于我们的词嵌入维度
    d_k = query.size(-1)

    # 按照注意力公式，将query与key转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数
    # 得到注意力的得分张量score
    score = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较
        # 如果掩码张量处为0 则对应的score张量用-1e9替换
        score = score.masked_fill(mask==0,-1e9)

    p_atten= F.softmax(score,dim=-1)

    if dropout is not None:
        p_atten = dropout(p_atten)
    # 返回注意力表示
    return torch.matmul(p_atten,value.float()),p_atten

# 4.多头注意力机制
def clones(module,N):
    '''
    生成相同的网络层的克隆函数
    :param module:  目标网络层
    :param N: 克隆数量
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAtten(nn.Module):
    def __init__(self,head,embedding_dim,dropout=0.1):
        '''

        :param head: 头数
        :param embedding_dim: 词嵌入维度
        :param dropout:
        '''
        super(MultiHeadAtten,self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被此嵌入维度整除
        # 因为要给每个头分配等量的词特征
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_K
        self.d_k = embedding_dim // head
        # 获得头数
        self.head = head
        # 克隆四个全连接层对象，通过nn的Linear实例化
        self.linears = clones(nn.Linear(embedding_dim,embedding_dim),4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以未None
        self.attn = None
        # 最后就是一个self.dropout对象
        self.dpot = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        # 前向逻辑函数
        if mask is not None:
            # 扩展维度 代表对头中的第i个头
            mask = mask.unsqueeze(1)
        # 获取样本数
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个全连接层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头风格输入，使用view方法对线性变换的结果进行维度重塑，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度
        # 计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作
        # lis = [query,key,value]
        # r = []
        # for step,model in enumerate(self.linears):
        #     r.append(model(lis[step]))
        query,key,value = [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) for model,x in zip(self.linears,(query,key,value))]
        # 得到每个头的输入后，接下来就是将他们传入到attention中,
        # 这里直接attention函数
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dpot)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的四维张量，我们需要将其转换为输入的形式
        # 对 第2，3维进行转置 然后使用contiguous方法
        # 这个方法的作用就是能够让转置后的张量应用view方法
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)

        # 使用线性层列表中的最后一个线性层对输入进行线性变
        return self.linears[-1](x)

# 5.前馈全连接层
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

# 6.规范化层
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

# 7.子层连接结构
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


if __name__ == '__main__':
    vocab = 500
    d_model = 512
    dropout = 0.2
    inputs = torch.randint(low=0,high=100,size=(5,10),dtype=torch.long)

    # 实例化文本嵌入层对象
    TE = TextEmbedding(vocab,d_model)
    # 实例化位置编码层



    PE = PositionalEnconding(d_model,dropout,max_len=10)

    TER = TE(inputs)
    print(TER.shape)

    PER = PE(TER)
    print(PER.shape)

    # 实例化多头注意力机制
    head = 8
    MHA = MultiHeadAtten(head=head,embedding_dim=d_model,dropout=dropout)
    # MHAR = MHA(PER,PER,PER)
    # print(MHAR.shape)

    # 实例化规范化层
    # LN = LayerNorm(d_model)
    # LNR = LN(MHAR)
    # print(LNR.shape)
    MHAR = lambda x:MHA(x,x,x)

    # 实例化 子层连接结构
    SLC1 = SublayerConnection(d_model,dropout=dropout)
    SLC1R = SLC1(PER,MHAR)
    print(SLC1R.shape)

    # 实例化 前馈全连接层
    FF = FeedForward(d_model,1024,dropout=dropout)
    SLC2 = SublayerConnection(d_model,dropout=dropout)
    SLC2R = SLC2(SLC1R,FF)
    print(SLC2R.shape)