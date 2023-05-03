import random
import torch
import dgl
import math
import torch.nn as nn
import networkx as nx
from AttriRank import AttriRank

class MFA_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus, random_ratio):
        super().__init__()
        self.plus = plus
        self.random_ratio = random_ratio

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats):
        return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self,ids_per_cls_train, budget, vecs):
        centers = [vecs[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1,-1).mm(vecs[ids_per_cls_train[i]].permute(1,0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        rank_des = [s.sort(descending=True)[1].tolist() for s in sim]
        ids_selected = []
        
        random_budget = math.floor(budget*self.random_ratio)
        sample_budget = budget-random_budget

        rb = math.floor(random_budget/len(ids_per_cls_train))
        sb = math.floor(sample_budget/(2*len(ids_per_cls_train)))
        for i,ids in enumerate(ids_per_cls_train):
            #a
            nearest = rank[i][0:min(sb, len(ids))]
            ids_selected.extend([ids[i] for i in nearest])
            #b
            farest = rank_des[i][0:min(sb, len(ids))]
            ids_selected.extend([ids[i] for i in farest])
            #c
            ids_selected.extend(random.sample(ids,min(rb,len(ids))))
        return list(set(ids_selected))
    

class AttriRank_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats):
        return self.sampling(subgraph, feats, ids_per_cls_train, train_ids, budget)

    def sampling(self, subgraph, feats, ids_per_cls_train, train_ids, budget):
        print(subgraph.number_of_nodes(),feats.shape,len(train_ids))
        print(subgraph, feats[0:5] )
        # subgraph_ = 
        # feats_ = 
        
        # AR = AttriRank(subgraph_, feats_, itermax=100, weighted=False, nodeCount=len(feat_))
        # scores = AR.runModel(factors=0.5, kernel='rbf_ap',
        #                      Matrix=False,TotalRank=False,alpha=1.0,beta=1.0,print_every=100)

        ids_selected = []
        # sorted_pagerank = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        # sorted_pagerank_lst = []
        # for item in sorted_pagerank:
        #     if item[0] in train_ids:
        #         sorted_pagerank_lst.append(item[0])
        # pr_list = sorted_pagerank_lst[:min(pr_budget, len(train_ids))]
 
        return ids_selected
    

class PageRank_sampler(nn.Module):
    #
    def __init__(self, plus, random_ratio):
        super().__init__()
        self.plus = plus
        self.random_ratio = random_ratio

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats=None):
        return self.sampling(subgraph, ids_per_cls_train, train_ids, budget)
                                 
    def sampling(self, g, ids_per_cls_train, train_ids, budget):
        random_budget = math.floor(budget*self.random_ratio)
        pr_budget = budget - random_budget

        #random sample
        random_list = []
        for i,ids in enumerate(ids_per_cls_train):
            random_list.extend(random.sample(ids,min(math.floor(random_budget/len(ids_per_cls_train)),len(ids))))

        #pagerank sample
        pagerank = nx.pagerank(dgl.to_networkx(g.to('cpu')))
        sorted_pagerank = sorted(pagerank.items(), key=lambda x:x[1], reverse=True)
        sorted_pagerank_lst = []
        for item in sorted_pagerank:
            if item[0] in train_ids:
                sorted_pagerank_lst.append(item[0])
        pr_list = sorted_pagerank_lst[:min(pr_budget, len(train_ids))]

        return list(set(random_list).union(set(pr_list)))


class Random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats=None):
        return self.sampling(ids_per_cls_train, train_ids, budget)

    def sampling(self,ids_per_cls_train, train_ids, budget):
        ids_selected = []
        # for i,ids in enumerate(ids_per_cls_train):
        #     ids_selected.extend(random.sample(ids,min(budget,len(ids))))
        ids_selected = random.sample(train_ids, min(budget, len(train_ids)))
        return ids_selected
    

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