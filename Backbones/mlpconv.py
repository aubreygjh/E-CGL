from torch import nn
from dgl.utils import expand_as_pair
import dgl.function as fn

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


