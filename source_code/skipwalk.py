import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Model(torch.nn.Module, size_vertex, d):
    def __init__(self):
        super(Model, self).__init__()
        self.phi = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))

    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out = torch.matmul(hidden, self.phi2)
        return out