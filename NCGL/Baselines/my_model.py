import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .my_utils import *
from dgl import DropEdge, FeatMask

samplers = {'ppr': PPR_sampler(plus=False), 'random':random_sampler(plus=False)}

def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

def kaiming_normal_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class NET(torch.nn.Module):
    """
        A template for implementing new methods for NCGL tasks. The major part for users to care about is the implementation of the function ``observe()``, which is how the implemented NCGL method learns each new task.

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.args = args
        self.activation = F.elu

        # setup network
        self.net = model
        self.net.apply(kaiming_normal_init)
        self.sampler = samplers['ppr']

        

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup buffer
        self.current_task = -1
        self.prev_model = None
        self.fisher = []
        self.budget = int(args.my_args['sample_budget'])
        self.buffer_node_ids = []
        self.aux_g = None
        self.ema_weight = args.my_args['ema_weight']
        self.epochs = 0

        # setup contrastive loss
        self.tau: float = 0.5
        self.drop_edge = DropEdge(0.1)
        self.mask_node = FeatMask(0.3, node_feat_names=[])
        self.fc1 = nn.Linear(256, 128, device='cuda:{}'.format(args.gpu))
        self.fc2 = nn.Linear(128, 256, device='cuda:{}'.format(args.gpu))
        self.con_weight = args.my_args['con_weight']

        # setup optimizer
        self.opt = torch.optim.Adam(list(self.net.parameters())+list(self.fc1.parameters())+list(self.fc2.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, g, features):
        output = self.net(g, features)
        return output
    
                
    def observe(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks. Each time a new task is presented, this function will be called to learn the task. Therefore, how the model adapts to new tasks and prevent forgetting on old tasks are all implemented in this function.
                More detailed comments accompanying the code can be found in the source code of this template in our GitHub repository.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param prev_model: The model obtained after learning the previous task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        offset1, offset2 = self.task_manager.get_label_offset(t) # Since the output dimensions of a model may correspond to multiple tasks, offset1 and offset2 denote the starting end ending dimension of the output for task t.
        self.net.train()    


        # if the given task is a new task, set self.current_task to denote the current task index. This is mainly designed for the mini-batch training scenario, in which the data of a task may not come in simultaneously, and each batch of data may either belong to an existing task or a new task..
        if t != self.current_task:            
            self.current_task = t  
            sampled_ids = self.sampler(g, ids_per_cls_train, train_ids, self.budget)
            old_ids = g.ndata['_ID'].cpu()# the original node indices before mapping, the original node indices in the whole graph.
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            if t > 0:
                # nodes_to_retrive = list(set(self.buffer_node_ids).intersection(set(old_ids.tolist())))
                nodes_to_retrive = self.buffer_node_ids
                aux_g, _, _ = dataset.get_graph(node_ids=nodes_to_retrive)
                self.aux_g = aux_g.to(device=f'cuda:{features.get_device()}')
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))


        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size/(buffer_size+n_nodes)
        self.net.zero_grad()
        output_labels = labels[train_ids]
        output, _ = self.net(g, features) # get the model output
        # choose whether to balance the loss with the class sizes
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        # Whether the total number of dimensions are pre-defined or increase with the incoming of new tasks
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])  #this is where task-IL and class-IL differs
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)


        if t!=0:
            # calculate auxiliary loss based on replay if not the first task
            output, _ = self.net(self.aux_g, self.aux_features)
            if args.classifier_increase:
                loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels, weight=self.aux_loss_w_[offset1: offset2]) #this is where task-IL and class-IL differs
            else:
                loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
            loss = beta*loss + (1-beta)*loss_aux
        
        loss.backward()
        self.opt.step()


    # original version
    def observe_task_IL(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset, cur_epoch):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        n_epochs = args.epochs

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]  #this is where task-IL and class-IL differs
        self.net.train()

        buffer_size = 0
        for k in self.buffer_node_ids:
            buffer_size+=len(self.buffer_node_ids[k])
        beta = buffer_size/(buffer_size+len(train_ids))
        # ###Learning the new incoming data
        self.net.zero_grad()
        output_labels = labels[train_ids] # get the labels
        output, _ = self.net(g, features) # get the model outputs
        second_last_h = self.net.second_last_h
        if args.cls_balance: # choose whether to balance the loss with the class sizes
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])



        ### representation contrastive loss
        # g_aug2 = self.drop_edge(g).to(device='cuda:{}'.format(args.gpu))
        features_aug = drop_feature(features, 0.3).to(device='cuda:{}'.format(args.gpu))
        output_aug, _ = self.net(g, features_aug)
        second_last_h_aug=self.net.second_last_h
        # loss_con = self._con_loss(output[train_ids], output_aug[train_ids])
        loss_con = self._con_loss(second_last_h[train_ids], second_last_h_aug[train_ids])
        loss = loss + self.con_weight * loss_con



        if t != 0: # calculate auxiliary loss based on replay if not the first task
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1: o2])
                loss = beta * loss + (1 - beta) * loss_aux
        loss.backward()
        self.opt.step()

        # if the given task is a new task, set self.current_task to denote the current task index. 
        # This is mainly designed for the mini-batch training scenario, in which the data of a task 
        # may not come in simultaneously, and each batch of data may either belong to an existing 
        # task or a new task..
        if last_epoch == 0: 
            # ###Weight Rugularization
            self.net.zero_grad()
            output, _ = self.net(g, features)
            self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2]).backward()
            self.fisher = []
            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                # shape = pg.shape
                # pg = F.normalize(pg.view(-1),p=2,dim=-1).reshape(shape)

                min_pg = pg.min()
                max_pg = pg.max()
                pg = (pg-min_pg)/(max_pg-min_pg)
                self.fisher.append(pg)
        
            # ###Replay Module           
            sampled_ids = self.sampler(g, ids_per_cls_train, train_ids, self.budget)
            old_ids = g.ndata['_ID'].cpu()# the original node indices before mapping, the original node indices in the whole graph.
            self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            nodes_to_retrive = self.buffer_node_ids[t]
            aux_g, _, _ = dataset.get_graph(node_ids=nodes_to_retrive)
            self.aux_g.append(aux_g.to(device=f'cuda:{features.get_device()}'))
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

            # self.current_task = t

        if t != 0: # use momentum update to alleviate forgetting if not the first task
            with torch.no_grad():
                for i, (param_new, param_old) in enumerate(zip(self.net.parameters(), prev_model.net.parameters())):
                    # reg = self.ema_weight
                    # reg = cur_epoch/n_epochs*self.ema_weight
                    reg = self.fisher[i] 
                    param_new.data = (1.0-reg) * param_new.data + reg * param_old.data


    def _con_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        h1 = self._projection(z1)
        h2 = self._projection(z2)
        l1 = self._semi_loss(h1, h2)
        l2 = self._semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


    def _projection(self, z):   
        z_proj = F.elu(self.fc1(z))
        return self.fc2(z_proj)
    
    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


    def _semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self._sim(z1, z1))
        between_sim = f(self._sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    


    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
            The method for learning the given tasks under the task-IL setting with mini-batch training.

            :param args: Same as the args in __init__().
            :param g: The graph of the current task.
            :param dataloader: The data loader for mini-batch training
            :param features: Node features of the current task.
            :param labels: Labels of the nodes in the current task.
            :param t: Index of the current task.
            :param train_ids: The indices of the nodes participating in the training.
            :param ids_per_cls: Indices of the nodes in each class (currently not in use).
            :param dataset: The entire dataset (currently not in use).

        """
        self.epochs += 1
        last_epoch = self.epochs % args.epochs

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        self.net.train()

        # now compute the grad on the current task
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = 0
            for k in self.buffer_node_ids:
                buffer_size += len(self.buffer_node_ids[k])
            beta = buffer_size / (buffer_size + n_nodes_current_batch)

            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels - offset1, weight=loss_w_[offset1: offset2])

            if t != 0:
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1:o2])
                    loss = beta * loss + (1 - beta) * loss_aux
            loss.backward()
            self.opt.step()

        # sample and store ids from current task
        # this block is for batch training, only execute once in the iteration of a dataloader
        if last_epoch == 0:
            # ###Weight Rugularization
            


            # ###Replay Module 
            sampled_ids = self.sampler(g, ids_per_cls_train, train_ids, self.budget)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            nodes_to_retrive = self.buffer_node_ids[t] #should have other versions
            aux_g, __, _ = dataset.get_graph(node_ids=nodes_to_retrive)
            self.aux_g.append(aux_g.to(device='cuda:{}'.format(args.gpu)))
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

            # self.current_task = t

        if t != 0: # use momentum update to alleviate forgetting if not the first task
            with torch.no_grad():
                for i, (param_new, param_old) in enumerate(zip(self.net.parameters(), prev_model.net.parameters())):
                    param_new.data = self.weight * param_new.data + (1.0 - self.weight) * param_old.data




'''
    # simplified version
    def observe_task_IL(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        n_tasks = args.n_tasks
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]  #this is where task-IL and class-IL differs
        self.net.train()

        if t != self.current_task: 
            self.current_task = t
            # ###Replay Module           
            sampled_ids = self.sampler(g, ids_per_cls_train, train_ids, self.budget)
            old_ids = g.ndata['_ID'].cpu()# the original node indices before mapping, the original node indices in the whole graph.
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            nodes_to_retrive = list(set(self.buffer_node_ids))
            aux_g, _, _ = dataset.get_graph(node_ids=nodes_to_retrive)
            self.aux_g = aux_g.to(device=f'cuda:{features.get_device()}')
            self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        buffer_size = self.aux_g.number_of_nodes()
        beta = buffer_size/(buffer_size+len(train_ids))
        # ###Learning the new incoming data
        self.net.zero_grad()
        output_labels = labels[train_ids] # get the labels
        output, _ = self.net(g, features) # get the model outputs
        if args.cls_balance: # choose whether to balance the loss with the class sizes
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t != 0: # calculate auxiliary loss based on replay if not the first task
            output, _ = self.net(self.aux_g, self.aux_features)
            loss_aux = self.ce(output[:, :offset2], self.aux_labels, weight=self.aux_loss_w_[: offset2])
            # loss = beta * loss + (1 - beta) * loss_aux
            loss = loss + loss_aux
        
        loss.backward()
        self.opt.step()

        if t != 0: # use momentum update to alleviate forgetting if not the first task
            with torch.no_grad():
                for i, (param_new, param_old) in enumerate(zip(self.net.parameters(), prev_model.net.parameters())):
                    reg = self.weight
                    param_new.data = reg * param_new.data + (1.0 - reg) * param_old.data
        

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
            The method for learning the given tasks under the task-IL setting with mini-batch training.

            :param args: Same as the args in __init__().
            :param g: The graph of the current task.
            :param dataloader: The data loader for mini-batch training
            :param features: Node features of the current task.
            :param labels: Labels of the nodes in the current task.
            :param t: Index of the current task.
            :param train_ids: The indices of the nodes participating in the training.
            :param ids_per_cls: Indices of the nodes in each class (currently not in use).
            :param dataset: The entire dataset (currently not in use).

        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
    

        # sample and store ids from current task
        # this block is for batch training, only execute once in the iteration of a dataloader
        if t != self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(g, ids_per_cls_train, train_ids, self.budget)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            nodes_to_retrive = list(set(self.buffer_node_ids))
            aux_g, __, _ = dataset.get_graph(node_ids=nodes_to_retrive)
            self.aux_g = aux_g.to(device='cuda:{}'.format(args.gpu))
            self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))


        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = self.aux_g.number_of_nodes()
            beta = buffer_size / (buffer_size + n_nodes_current_batch)

            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels - offset1, weight=loss_w_[offset1: offset2])


            if t != 0:
                output, _ = self.net(self.aux_g, self.aux_features)
                loss_aux = self.ce(output[:, :offset2], self.aux_labels, weight=self.aux_loss_w_[: offset2])
                # loss = beta * loss + (1 - beta) * loss_aux
                loss = loss + loss_aux

            loss.backward()
            self.opt.step()


            if t != 0: # use momentum update to alleviate forgetting if not the first task
                with torch.no_grad():
                    for i, (param_new, param_old) in enumerate(zip(self.net.parameters(), prev_model.net.parameters())):
                        param_new.data = self.weight * param_new.data + (1.0 - self.weight) * param_old.data
'''