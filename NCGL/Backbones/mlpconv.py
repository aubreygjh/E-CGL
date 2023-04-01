"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair
from dgl.base import DGLError
import math
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
import torch.autograd as autograd
# from re import S
# from xml.dom import xmlbuilder
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.utils import num_nodes, to_dense_adj
# import numpy as np
# from torch_geometric.utils import remove_self_loops, add_self_loops, degree
# import math

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')
class MLP_GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2):
        super(MLP_GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, feat, graph=None, use_conv=True):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        h = self.linear(feat)
        if use_conv and graph is not None:
            graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
            graph.ndata['h'] = h
            graph.update_all(gcn_msg, gcn_reduce)
            h = graph.ndata['h']
        return h

    def forward_batch(self, feat, block=None, use_conv=True):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        feat_src, feat_dst = expand_as_pair(feat)
        h = self.linear(feat_src)
        if use_conv and block is not None:
            block = block.local_var().to('cuda:{}'.format(feat.get_device()))
            block.srcdata['h'] = h
            block.update_all(gcn_msg, gcn_reduce)
            h = block.dstdata['h']
        return h

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)


'''
def gcn_conv(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_noloop(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5) # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_rw(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N) 

    deg_src = deg[src].pow(-0) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_diff(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N) 

    deg_src = deg[src].pow(-0)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1) 
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = heat_kernel(a, h, 10) 
    return h_prime

def heat_kernel(a, h, k):
    h_prime = h, h_temp = h
    for i in range(1, k+1):
        h_temp = a @ h_temp / i
        h_prime += h_temp
    return h_prime / math.e


def conv_resi(h, edge_index, h_ori, alpha):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5) # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = (1. - alpha) * a @ h + alpha * h_ori
    return h_prime
'''