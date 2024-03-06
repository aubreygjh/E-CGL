import torch
from .cat_utils import Condenser

class NET(torch.nn.Module):
    def __init__(self, model, task_manager, args):
        super(NET, self).__init__()
        
        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.condenser = Condenser(args)
        self.current_task = -1
        self.buffer_c_node = []
        self.buffer_all_nodes = []
        self.aux_g = None


    def forward(self, features):
        output = self.net(features)
        return output
    

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.aux_loss_w_ = []
        self.net.train()

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        budgets = self._assign_budget_per_cls(args.cat_args['budget'])
        if t!=self.current_task:
            self.current_task = t
            condensed_g = self.condenser(g, budgets)

            self.aux_g.append(condensed_g.to(device='cuda:{}'.format(features.get_device())))
            labels = condensed_g.dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

        if t!=0:
            # if not the first task, calculate aux loss with buffered data
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][offset1: offset2])
                loss += loss_aux

        loss.backward()
        self.opt.step()


    def _assign_budget_per_cls(self, budget):
        return budgets

