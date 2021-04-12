import torch
import torch.nn as nn
from Atten import attention
import copy
from torch.autograd import Variable
'''
    使用一组线性变化层对Q，K，V分别进行线性变换
    这些变换不会改变原有张量的尺寸，因此每个变换矩阵都是方阵
    每个头从词义层面分割输出张量  也就是每个头都想获得一组Q,K,V
    但是句子中的每个词的表示只获得一部分，也就是只分割了最后一维的词嵌入向量 这就是所谓的多头
    把每个头的获得的输入送到注意力机制中，就形成多头注意力机制
'''

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

if __name__ == '__main__':
    head = 8
    embedding_dim = 512
    dropout = 0.2

    pe_result = torch.randint(low=0, high=500, size=[5, 10, 512])
    query=key=value=pe_result
    mask = Variable(torch.zeros(8,4,4))
    mha = MultiHeadAtten(head,embedding_dim,dropout)
    mha_result = mha(query,key,value,mask)
    print(mha_result)
    print(mha_result.shape)