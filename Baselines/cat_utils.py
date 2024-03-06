import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import random

from Backbones.gnns import GCN

class Condenser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_cls = args.n_cls


    def forward(self, graph, budgets, ids_per_cls_train):
        labels_cond = []
        for i, cls in enumerate(self.n_cls):
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)

        feat_cond = torch.nn.Parameter(torch.FloatTensor(sum(budgets), graph.num_features))
        feat_cond = self._initialize_feature(graph, budgets, feat_cond, self.feat_init)

        replayed_graph = self._condense(graph, feat_cond, labels_cond, budgets)
        return replayed_graph
    

    def _initialize_feature(self, graph, budgets, feat_cond, method="randomChoice"):
        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)
        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(self.n_cls):
                train_mask = graph.train_mask
                train_mask_at_cls = (graph.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += random.choices(ids_at_cls, k=budgets[i])
            sampled_feat = graph.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        # elif method == "kMeans":
        #     sampled_ids = []
        #     for i, cls in enumerate(self.n_cls):
        #         train_mask = graph.train_mask
        #         train_mask_at_cls = (graph.y == cls).logical_and(train_mask)
        #         ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
        #         sampled_ids += query(graph, ids_at_cls, budgets[i], self.device)
        #     sampled_feat = graph.x[sampled_ids]
        #     feat_cond.data.copy_(sampled_feat)
        return feat_cond


    def _condense(self, graph, feat_cond, labels_cond, budgets):
        self_loops = SparseTensor.eye(sum(budgets), sum(budgets)).t()
        opt_feat = torch.optim.Adam([feat_cond], lr=self.feat_lr)

        cls_train_masks = []
        for cls in self.n_cls:
            cls_train_masks.append((graph.y == cls).logical_and(graph.train_mask))   
        
        encoder = GCN(graph.num_features, self.hid_dim, self.emb_dim, self.n_layers, self.hop, self.activation).to(self.device)
        for _ in range(self.n_encoders):
            encoder.initialize()
            with torch.no_grad():
                emb_real = encoder.encode(graph.x.to(self.device), graph.adj_t.to(self.device))
                emb_real = F.normalize(emb_real)
            emb_cond = encoder.encode_without_e(feat_cond.to(self.device))
            emb_cond = F.normalize(emb_cond)

            loss = torch.tensor(0.).to(self.device)
            for i, cls in enumerate(self.n_cls):
                real_emb_at_class = emb_real[cls_train_masks[i]]
                cond_emb_at_class = emb_cond[labels_cond == cls]
                
                dist = torch.mean(real_emb_at_class, 0) - torch.mean(cond_emb_at_class, 0)
                loss += torch.sum(dist ** 2)
        
            # Update the feature matrix
            opt_feat.zero_grad()
            loss.backward()
            opt_feat.step()

        # Wrap the graph data object
        replayed_graph = Data(x=feat_cond.detach().cpu(), 
                              y=labels_cond, 
                              adj_t=self_loops)
        replayed_graph.train_mask = torch.ones(sum(budgets), dtype=torch.bool)
        return replayed_graph
