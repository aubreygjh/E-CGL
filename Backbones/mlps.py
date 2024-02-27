from .mlpconv import *
from .utils import *

class MLP_GCN(nn.Module):
    def __init__(self,
                 args):
        super(MLP_GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(MLP_GCNLayer(dims[l], dims[l+1]))

    def forward(self, g=None, features=None):
        use_conv = False if self.training else True
        h = features
        for layer in self.gat_layers[:-1]:
            h = layer(h, g, use_conv)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gat_layers[-1](h, g, use_conv)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        return logits, None

    def forward_batch(self, blocks=None, features=None):
        use_conv = False if self.training else True
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h = layer.forward_batch(h, blocks[i], use_conv)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gat_layers[-1].forward_batch(h, blocks[-1], use_conv)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        
        return logits, None

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()
