'''
        每个编码器部分由n个编码器层堆叠而成
        每个编码器层由两个子层连接结构组成
        第一个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
        第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
'''
import math
import numpy as np
import torch
import torch.nn.functional as F

'''
    掩码张量
    掩码张量一般只有1和0元素，代表位置被遮掩或者不被遮掩
    在transformer中，掩码张量的作用在应用attention时有一些生成的attention变量中的值有可能已知了未来信息而得到的
    未来信息被看到是因为训练时会把整个输出结果都一次性进行Embedding
    但是理论上解码器的输出不是一次就能产生最终结果，而是一次次通过上一次结果综合得出的
    因此，未来信息可能被提前利用
'''
def subsequent_mask(size):

    atten_shape = (1,size,size)
    # 对角线下就是负 对角线上就是正 对角线就是0
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    return torch.from_numpy(1-mask)

'''
    观察事物 大脑很快把注意力放在事物最具有辨识度的部分从而作出判断
    并非从而到尾的观察一遍事物后 才能有判断结果 
    正是基于这样的理论 产生了注意力机制
    注意力机制是 注意力计算规则能够应用的深度学习网络的载体
'''
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

if __name__ == '__main__':
    size = 10
    mask = subsequent_mask(size)
    print(mask)

    pe_result = torch.randint(low=0,high=500,size=[5,10,512])

    query = key = value = pe_result

    attn,p_attn = attention(query,key,value,mask=mask)

    print("attn:",attn)
    print(attn.shape)
    print("p_attn:",p_attn)
    print(p_attn.shape)