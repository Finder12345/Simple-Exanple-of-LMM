# -*- coding: utf-8 -*-
# @Date    :2025-03-11 18:30:48
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
import torch.nn as nn
import math


class GQAAttention(nn.Module):
    """docstring for GQAAttention"""

    def __init__(self, d_model=512, num_heads=8, n_group=4):
        super(GQAAttention, self).__init__()
        # 定义属性
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_group = n_group
        self.head_dim = d_model // num_heads

        # 定义计算
        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,self.n_group*self.head_dim)
        self.wv = nn.Linear(d_model,self.n_group*self.head_dim)

    def forward(self,h:torch.Tensor,mask=None):
        bs,seq,_ = h.shape
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        # 多头切分，并进行分组，分组是1,1,2,2，不能是1,2,1,2
        q = q.view(bs,seq,self.num_heads,-1).transpose(1,2)
        k = k.view(bs,seq,self.n_group,-1).repeat(1,1,1,self.num_heads//self.n_group).view(bs,seq,self.num_heads,-1).transpose(1,2)
        v = v.view(bs,seq,self.n_group,-1).repeat(1,1,1,self.num_heads//self.n_group).view(bs,seq,self.num_heads,-1).transpose(1,2)

        # 计算
        scores = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.head_dim)
        # print(scores.shape)
        attention = torch.softmax(scores,dim=3)
        out = torch.matmul(attention,v).transpose(1,2).contiguous().view(bs,seq,-1)

        return out

if __name__ == '__main__':
    gqa = GQAAttention()
    h = torch.randn(12,10,512)
    out = gqa(h)
    print(out.shape)