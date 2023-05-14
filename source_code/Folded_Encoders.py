import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from Attention import Self_Attn
import time


class Folded_Encoder(nn.Module):

    def __init__(self, features, uv2e, embed_dim, seq_len, folded_seq, base_model=None, cuda="cpu"):
        super(Folded_Encoder, self).__init__()
        self.features = features
        self.uv2e = uv2e
        self.seq_len = seq_len
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.folded_seq = folded_seq
        self.Self_Attn = Self_Attn(embed_dim)
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        node_seq = []
        for node in nodes.cpu().numpy():
            seq = self.folded_seq[int(node)]
            while len(seq) < self.seq_len:
                seq.append(node)
            node_seq.append(seq[::-1])

        batch_size = len(node_seq)

        node_seq = [item for sublist in node_seq for item in sublist]

        # q = self.uv2e(nodes).to(self.device)#much faster
        q = self.uv2e.weight[nodes].to(self.device)
        #q = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device).t() # nodes * dim
        
        # k = self.uv2e(torch.LongTensor(node_seq).to(self.device))#much faster
        k = self.uv2e.weight[torch.LongTensor(node_seq).to(self.device)]
        #k = self.features(torch.LongTensor(node_seq)).to(self.device).t()#nodes * 32 * dim

        # v = self.uv2e(torch.LongTensor(node_seq).to(self.device)) #much faster
        v = self.uv2e.weight[torch.LongTensor(node_seq).to(self.device)]
        #v = self.features(torch.LongTensor(node_seq)).to(self.device).t()#nodes * 32 * dim

        folded_feats, attention = self.Self_Attn.forward(q, k, v, batch_size, scale = 0.125) 

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        combined = torch.cat([self_feats, folded_feats.view(batch_size, self.embed_dim)], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
