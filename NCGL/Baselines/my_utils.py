import random
import torch
import dgl
import torch.nn as nn
import networkx as nx

class random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            ids_selected.extend(random.sample(ids,min(budget,len(ids))))
        return ids_selected
    

class PPR_sampler(nn.Module):
    #
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget):
        if self.plus:
            return self.sampling(subgraph, ids_per_cls_train, train_ids, budget)
        else:
            return self.sampling(subgraph, ids_per_cls_train, train_ids, budget)
                                 
    def sampling(self, g, ids_per_cls_train, train_ids, budget):
        ids_selected = []
        pagerank = nx.pagerank(dgl.to_networkx(g.to('cpu')))
        sorted_pagerank = sorted(pagerank.items(), key=lambda x:x[1], reverse=True)
        sorted_pagerank_lst = []
        for item in sorted_pagerank:
            if item[0] in train_ids:
                sorted_pagerank_lst.append(item[0])

        
        return sorted_pagerank_lst[:min(budget, len(train_ids))]
    

class NearestNeighbor_sampler(nn.Module):
    #
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)
                                 
    def sampling(self, ids_per_cls_train, budget, vecs):
        ids_selected = []
        
        return ids_selected
    
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x