import random
import torch
import dgl
import math
import torch.nn as nn
import networkx as nx
import dgl.function as fn
from .AttriRank import AttriRank

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


class AttriRank_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus, diversity_ratio):
        super().__init__()
        self.plus = plus
        self.diversity_ratio = diversity_ratio

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats):
        return self.sampling(subgraph, feats, ids_per_cls_train, train_ids, budget)

    def sampling(self, subgraph, feats, ids_per_cls_train, train_ids, budget):
        # budget = int(budget*len(train_ids))

        diversity_budget = math.floor(budget*self.diversity_ratio)
        importance_budget = budget - diversity_budget

        # ### importance sampling
        g = subgraph.local_var()
        src, dst = g.edges()
        edges = torch.cat([src.unsqueeze(1),dst.unsqueeze(1)],dim=1)# edges = np.array(g.edges().cpu()).T
        AR = AttriRank(edges.detach().cpu(), feats.detach().cpu(), itermax=1000, weighted=False, nodeCount=feats.shape[0])
        scores_attrirank = AR.runModel(dampFac=0.85)
        sorted_attrirank = sorted(range(len(scores_attrirank)),key=lambda x:scores_attrirank[x], reverse=True)
        
        # ### diversity sampling
        g.update_all(fn.copy_u('feat','m'), fn.mean('m', 'neighbor_feat'))
        scores_diversity = torch.norm(g.ndata['feat']-g.ndata['neighbor_feat'], dim=1).tolist()
        sorted_diversity = sorted(range(len(scores_diversity)),key=lambda x:scores_diversity[x], reverse=True)

        # ### CM
        # vecs = feats.half()
        # budget_dist_compute = 1000
        # d = 0.5
        # sorted_diversity_idx = []
        # for i,ids in enumerate(ids_per_cls_train):
        #     other_cls_ids = list(range(len(ids_per_cls_train)))
        #     other_cls_ids.pop(i)
        #     ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i])<budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)
        #     dist = []
        #     vecs_0 = vecs[ids_selected0]
        #     for j in other_cls_ids:
        #         chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))
        #         vecs_1 = vecs[chosen_ids]
        #         if len(chosen_ids) < 26 or len(ids_selected0) < 26:
        #             # torch.cdist throws error for tensor smaller than 26
        #             dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
        #         else:
        #             dist.append(torch.cdist(vecs_0,vecs_1))
        #     dist_ = torch.cat(dist,dim=-1) # include distance to all the other classes
        #     n_selected = (dist_<d).sum(dim=-1)
        #     rank = n_selected.sort()[1].tolist()
        #     current_ids_selected = rank[:budget]
        #     sorted_diversity_idx.extend([ids_per_cls_train[i][j] for j in current_ids_selected])

        train_ids_set = set(train_ids)
        sorted_attrirank_idx = [idx for idx in sorted_attrirank if idx in train_ids_set]
        sorted_diversity_idx = [idx for idx in sorted_diversity if idx in train_ids_set]


        ret = list(set(sorted_attrirank_idx[:min(importance_budget, len(train_ids))]).union(set(sorted_diversity_idx[:min(diversity_budget, len(train_ids))])))
        return ret


class Random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats=None):
        return self.sampling(ids_per_cls_train, train_ids, budget)

    def sampling(self,ids_per_cls_train, train_ids, budget):
        ids_selected = random.sample(train_ids, min(budget, len(train_ids)))
        # ids_selected = train_ids[:min(budget,len(train_ids))]
        return ids_selected


class MF_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats):
        return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self,ids_per_cls_train, budget, vecs):
        budget = int(budget/len(ids_per_cls_train))
        centers = [vecs[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1,-1).mm(vecs[ids_per_cls_train[i]].permute(1,0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            nearest = rank[i][0:min(budget, len(ids_per_cls_train[i]))]
            ids_selected.extend([ids[i] for i in nearest])
        return ids_selected
    

class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, subgraph, ids_per_cls_train, train_ids, budget, feats, d=0.5, using_half=True):
        return self.sampling(ids_per_cls_train, budget, feats, d, using_half=using_half)
  

    def sampling(self,ids_per_cls_train, budget, vecs, d, using_half=True):
        budget = int(budget/len(ids_per_cls_train))
        budget_dist_compute = 1000
        '''
        if using_half:
            vecs = vecs.half()
        '''
        vecs = vecs.half()
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i])<budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0,vecs_1))

            #dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist,dim=-1) # include distance to all the other classes
            n_selected = (dist_<d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x