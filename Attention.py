import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dims):
        super(Self_Attn, self).__init__()
        self.embed_dim = embedding_dims
        # self.Wq = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        # self.Wk = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))  
        # self.Wv = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax  = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.linear_final = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v, batch_size, scale=None, attn_mask=None):
        # Q = torch.mm(q,self.Wq)
        # K = torch.mm(k,self.Wk)
        # V = torch.mm(v,self.Wv)
        Q = self.linear_q(q)
        K = self.linear_q(k)
        V = self.linear_q(v)
        residual = Q

        Q = Q.view(batch_size, 1, self.embed_dim)
        K = K.view(batch_size, -1, self.embed_dim)
        V = V.view(batch_size, -1, self.embed_dim)
        
        # print Q.size()
        # print K.size()
        # print V.size()
        attention = torch.bmm(Q, K.transpose(1,2))
        if scale:
            attention = attention * scale
        #if attn_mask:
        #    attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = F.dropout(attention,training=self.training)
        context = torch.bmm(attention, V)

        context = context.view(batch_size,self.embed_dim)
   
        output = self.linear_final(context)
        output = F.dropout(output,training=self.training)
        output = self.layer_norm(residual + output)


        return output, attention