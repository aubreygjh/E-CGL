import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, SGC
from .mlps import MLP_GCN

def get_model(dataset, args):
    n_classes = args.n_cls_per_task
    print('n_classes', n_classes)
    if args.backbone.upper() == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone.upper() == 'GCN':
        model = GCN(args)
    elif args.backbone.upper() == 'GIN':
        model = GIN(args)
    elif args.backbone.upper() == 'MLP':
        model = MLP_GCN(args)
    elif args.backbone.upper() == 'SGC':
        model = SGC(args)
    return model
