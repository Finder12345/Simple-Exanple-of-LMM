import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, num_heads, base=10000, max_len=512):
        super().__init__()
        self.head_dim = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        # 初始化时计算一次
        self.cos_pos_cache, self.sin_pos_cache = self._compute_pos_emb()

    def _compute_pos_emb(self):
        # theta_i=1/(10000^(2i/head_dim))i的范围是[0,head_dim//2]
        theta_i = 1. / \
            (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        # 根据最大长度创建位置索引序列，元素m对应第 m个位置
        positions = torch.arange(self.max_len)
        # max_len个位置对应m*theta->[max_len,head_dim//2]
        pos_emb = positions.unsqueeze(1)*theta_i.unsqueeze(0)
        # cos相邻位置复制一次，eg：123-》112233
        cos_pos = pos_emb.sin().repeat_interleave(2, dim=-1)
        # sin相邻位置复制一次,[max_len,head_dim]
        sin_pos = pos_emb.cos().repeat_interleave(2, dim=-1)

        return cos_pos, sin_pos

    def forward(self, q):
        bs, q_len = q.shape[0], q.shape[1]
        self.cos_pos = self.cos_pos_cache[:q_len]
        self.sin_pos = self.sin_pos_cache[:q_len]
        # q压缩出num_heads以便于在head_dim上施加rope位置编码
        q = q.reshape(bs, q_len, self.num_heads, -1).transpose(1, 2)

        # repeat沿着指定位置复制，bs,num_head纬度上复制以便和q,k计算，其他纬度为1不复制->[bs,num_heads,max_len,head_dim]
        self.cos_pos = self.cos_pos.repeat(
            bs, self.num_heads, *([1]*len(self.cos_pos.shape)))
        self.sin_pos = self.sin_pos.repeat(
            bs, self.num_heads, *([1]*len(self.sin_pos.shape)))

        # 有了sin_pos,还需对q进行负奇，偶交替处理,先抽取q的奇偶元素再stack扩展最后一个维度让奇偶相邻
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        # 再reshape压缩最后一个维度实现负奇，偶交替
        q2 = q2.reshape(bs, self.num_heads, q_len, -1)
        # q与位置编码相乘
        r_q = q*self.cos_pos+q2*self.sin_pos

        return r_q


class MLAAttention(nn.Module):
    """docstring for MLAAttention"""

    def __init__(self, d_model=512, down_dim=128,
                 up_dim=256, num_heads=8,
                 rope_head_dim=26, dropout_prod=0.1):
        super(MLAAttention, self).__init__()
        self.d_model = d_model  # 512
        self.down_dim = down_dim  # 128
        self.up_dim = up_dim  # 256
        self.num_heads = num_heads  # 8
        self.rope_head_dim = rope_head_dim  # 26,旋转每个头为维度
        self.head_dim = d_model//num_heads  # 64，q每个头的维度
        self.v_head_dim = up_dim//num_heads  # 32 v每个头的维度
        # 初始化kv联合基于相应的down,up,的projection matrix
        # 256->128->256
        self.down_proj_kv = nn.Linear(d_model, down_dim)  # 这里可以使用同一个矩阵
        self.up_proj_k = nn.Linear(down_dim, up_dim)  # 从参数吸收角度进行理解
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.down_proj_q = nn.Linear(d_model, down_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)
        # 初始化解耦q,k，并进行MQA的计算
        # 映射到相关性（correlation）,的q,k，进而方便计算RoPE
        self.proj_qr = nn.Linear(down_dim, rope_head_dim*num_heads)
        self.proj_kr = nn.Linear(d_model, rope_head_dim*1)
        # 初始化去q,k对应的rope类
        # 成倍数关系即可，这里使用MQA进行模拟实现
        self.rope_q = RotaryEmbedding(rope_head_dim*num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)
        # Drooout and Feed forward
        self.dropout = nn.Dropout(dropout_prod)
        self.fc = nn.Linear(num_heads*self.v_head_dim, d_model)  # 256->512
        self.res_dropout = nn.Dropout(dropout_prod)

    def forward(self, h: torch.Tensor, mask=None):
        bs, seq_len, _ = h.shape
        # step1:中间低秩转换:512->128->256
        c_t_kv = self.down_proj_kv(h)  # 128
        k_t_c = self.up_proj_k(c_t_kv)  # 256
        v_t_c = self.up_proj_v(c_t_kv)  # 256

        c_t_q = self.down_proj_q(h)  # 128
        q_t_c = self.up_proj_q(c_t_q)  # 256

        # step2:进行解耦q,k并添加RoPE
        q_t_r = self.rope_q(self.proj_qr(c_t_q))
        # 128->26*8->[8,26],head,r_head_dim
        k_t_r = self.rope_k(self.proj_kr(h))
        # 512->26*1->[1,26],head,r_head_dim

        # step3:分头，和计算注意力
        q_t_c = q_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        #[8,32]  #
        q = torch.cat([q_t_c, q_t_r], dim=-1)  # [8,:,58]

        k_t_c = k_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # [8,32]
        k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)  # 拼接时保持维度一致
        # [8,26]
        k = torch.cat([k_t_c, k_t_r], dim=-1)
        # [:,8:,58]

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = torch.softmax(
            scores/(math.sqrt(self.head_dim+self.rope_head_dim)), dim=-1)
        scores = self.dropout(scores)
        # print(scores.shape)
        # [:.:,seq,seq]

        v_t_c = v_t_c.reshape(bs, seq_len, self.num_heads,
                              self.v_head_dim).transpose(1, 2)
        # [:,8,:seq,32] 
        output = torch.matmul(scores, v_t_c) #[:,:,seq,32]

        print(output.shape)
        output = output.transpose(1,2).reshape(bs,seq_len,-1) # 等价与contiguous和view的组合
        output = self.fc(output)
        output = self.res_dropout(output)
        return output


if __name__ == '__main__':
    mla = MLAAttention()

    h = torch.randn(12, 10, 512)
    out = mla(h)
    print(out.shape)
