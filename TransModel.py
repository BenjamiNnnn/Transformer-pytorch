from TransLayer import TextEmbedding,PositionalEnconding,Encoder,Decoder,Generator
from TransLayer import MultiHeadAtten,FeedForward,EncoderLayer,DecoderLayer,subsequent_mask
import torch.nn as nn
import torch
import copy

class Transformer(nn.Module):
    def __init__(self,TextEmbdedding,PoisitionEmbedding,Encoder,Decoder,Generator):
        '''

        :param TextEmbdedding: 文本编码对象
        :param PoisitionEmbedding: 位置编码对象
        :param Encoder:
        :param Decoder:
        :param Generator:
        '''
        super(Transformer, self).__init__()
        self.TE = TextEmbdedding
        self.PE = PoisitionEmbedding
        self.EN = Encoder
        self.DE = Decoder
        self.G = Generator

    def forward(self,source_inputs,target_inputs,mask):

        source_out = self.TE(source_inputs)
        print("源文本嵌入:",source_out.shape)
        source_out = self.PE(source_out)
        print("源位置编码:", source_out.shape)

        target_out = self.TE(target_inputs)
        print("目的文本嵌入:", target_out.shape)
        target_out = self.PE(target_out)
        print("目的位置编码:", target_out.shape)

        out = self.EN(source_out,mask)
        print("编码器:", out.shape)

        out = self.DE(target_out,out,mask,mask)
        print("解码器:", out.shape)
        out = self.G(out)
        print("模型输出:", out.shape)
        return out

if __name__ == '__main__':
    vocab = 500
    d_model = 512
    dropout = 0.2
    source_inputs = torch.randint(low=0, high=100, size=(5, 10), dtype=torch.long)
    target_inputs = torch.randint(low=0, high=100, size=(5, 10), dtype=torch.long)
    c = copy.deepcopy

    mask = subsequent_mask(10)

    # 实例化文本嵌入层对象
    TE = TextEmbedding(vocab, d_model)
    # 实例化位置编码层
    PE = PositionalEnconding(d_model, dropout, max_len=10)

    N = 8
    head = 8
    attn = MultiHeadAtten(head, d_model)
    ff = FeedForward(d_model, 64, dropout)
    # 编码器部分
    Enlayer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    E = Encoder(Enlayer, N)
    # 解码器部分
    Delayer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    D = Decoder(Delayer, N)
    # 输出部分
    G = Generator(d_model, vocab)

    TransM = Transformer(TE,PE,E,D,G)
    print(TransM)
    out = TransM(source_inputs,target_inputs,mask)
    print(out)
