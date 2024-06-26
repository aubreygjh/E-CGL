import os
import pickle
import numpy as np
import time
import torch
import torch.utils.data as thdata
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, evaluate_batch, NodeLevelDataset, NoGraphDataset
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl


### Simplied get_pipeline
def get_pipeline(args): 
    if args.minibatch:
        if args.ILmode == 'classIL':
            return pipeline_classIL_minibatch
        elif args.ILmode == 'taskIL':
            return pipeline_taskIL_minibatch
    elif not args.minibatch:
        if args.ILmode == 'classIL':
            return pipeline_classIL
        elif args.ILmode == 'taskIL':
            return pipeline_taskIL


def data_prepare(args):
    """
    check whether the processed data exist or create new processed data
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    n_cls_so_far = 0
    # check whether the preprocessed data exist and can be loaded
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            # if not exist or cannot be loaded correctly, create new processed data
            print(f'preparing data for task {task}')
            mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
            mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)


def pipeline_taskIL(args, valid=False):
    # valid=True denotes the evaluation is done on validation set, otherwise on testing set
    epochs = args.epochs if valid else 0 # training epochs is zero for testing mode
    torch.cuda.set_device(args.gpu)

    # Load Dataset
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)] # this line will remove the final task if only one class included
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    # Load Model
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    # model_params = sum(p.numel() for p in model.parameters())
    # life_model_params = sum(p.numel() for p in life_model_ins.parameters())
    # print(f"Model Params: {model_params}, {life_model_params}.")

    # Prepare Dataset for Continual Format
    data_prepare(args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    train_time = []
    infer_time = []

    # Iterate Over Task Sequences
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

        # Get Subgraph for Current Task
        if args.method == 'joint':
            subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
            for t in range(task + 1):
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
                subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
                features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
                subgraphs.append(subgraph)
                featuress.append(features)
                labelss.append(labels)
                train_idss.append(train_ids)
                ids_per_clss.append(ids_per_cls) 
        else:
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        # Train
        start_time = round(time.time()*1000,3)
        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model, train_ids,ids_per_cls, dataset)
            elif args.method == 'joint':
                life_model_ins.observe_task_IL(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
        end_time = round(time.time()*1000,3)
        train_time.append((end_time-start_time)/epochs)

        # Test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        start_time = round(time.time()*1000,3)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl','rb'))
            test_ids = valid_ids_ if valid else test_ids_  # whether use validation or test set
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                # deprecated
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")
        end_time = round(time.time()*1000,3)
        infer_time.append((end_time-start_time)/(task+1))
        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}|", end="")
        print(f"time:({round(np.mean(train_time),2)}/{round(np.mean(infer_time),2)})ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)  # save the best model for each hyperparameter composition
        prev_model = copy.deepcopy(life_model_ins).cuda(args.gpu)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)

    return acc_mean, mean_backward, acc_matrix


def pipeline_taskIL_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    
    # Load Dataset
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    # Load Model
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    # model_params = sum(p.numel() for p in model.parameters())
    # life_model_params = sum(p.numel() for p in life_model_ins.parameters())
    # print(f"Model Params: {model_params}, {life_model_params}.")
    
    # Prepare Dataset for Continual Format
    data_prepare(args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    train_time = []
    infer_time = []

    # Iterate Over Task Sequences
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

        if args.method == 'joint':
            subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
            for t in range(task + 1):
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
                features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
                subgraphs.append(subgraph)
                featuress.append(features)
                labelss.append(labels)
                train_idss.append(train_ids)
                ids_per_clss.append(ids_per_cls)

            # build the dataloader for mini batch training
            train_ids_entire = []
            n_existing_nodes = 0
            for ids, g in zip(train_idss, subgraphs):
                new_ids = [i + n_existing_nodes for i in ids]
                train_ids_entire.extend(new_ids)
                n_existing_nodes += g.num_nodes()
            graph_entire = dgl.batch(subgraphs)
            dataloader = dgl.dataloading.DataLoader(graph_entire, train_ids_entire, args.nb_sampler,
                                                        batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                        drop_last=False, device='cuda:{}'.format(args.gpu))
        else:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
                open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl',
                    'rb'))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()    

            # build the dataloader for mini batch training
            dataloader = dgl.dataloading.DataLoader(subgraph, train_ids, args.nb_sampler,
                                                        batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                        drop_last=False, device='cuda:{}'.format(args.gpu)) 
            # dataloader_feat = thdata.DataLoader(NoGraphDataset(features[train_ids], labels[train_ids]), 
            #                                     batch_size=args.batch_size,
            #                                     shuffle=args.batch_shuffle,
            #                                     drop_last=False)
     
        # Train
        start_time = round(time.time()*1000,3)
        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            elif args.method == 'joint':
                life_model_ins.observe_task_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()
        end_time = round(time.time()*1000,3)
        train_time.append((end_time-start_time)/epochs)

        # Test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        start_time = round(time.time()*1000,3)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            acc = evaluate_batch(args, model, subgraph, features, labels-label_offset1, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")
        end_time = round(time.time()*1000,3)
        infer_time.append((end_time-start_time)/(task+1))
        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}|", end="")
        print(f"time:({round(np.mean(train_time),2)}/{round(np.mean(infer_time),2)})ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(life_model_ins).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)

    return acc_mean, mean_backward, acc_matrix


def pipeline_classIL(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)

    # Load Dataset
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    # Load Model
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    # Prepare Dataset for Continual Format
    data_prepare(args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    train_time = []
    infer_time = []
    
    # Iterate Over Task Sequences
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

        if args.method == 'joint':
            subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
            for t in range(task + 1):
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
                subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
                features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
                subgraphs.append(subgraph)
                featuress.append(features)
                labelss.append(labels)
                train_idss.append(train_ids)
                ids_per_clss.append(ids_per_cls)
        else:
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
                open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        
        label_offset1, label_offset2 = task_manager.get_label_offset(task)

        # training
        start_time = round(time.time()*1000,3)
        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            elif args.method == 'joint':
                life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()
        end_time = round(time.time()*1000,3)
        train_time.append((end_time-start_time)/epochs)

        # test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        start_time = round(time.time()*1000,3)
        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")
        end_time = round(time.time()*1000,3)
        infer_time.append((end_time-start_time)/(task+1))
        accs = acc_mean[:task+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean: {acc_mean}|", end="")
        print(f"time:({round(np.mean(train_time),2)}/{round(np.mean(infer_time),2)})ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)

    return acc_mean, mean_backward, acc_matrix


def pipeline_classIL_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)

    # Load Dataset
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    # Load Model
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    # Prepare Dataset for Continual Format
    data_prepare(args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    train_time = []
    infer_time = []

    # Iterate Over Task Sequences
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

        if args.method == 'joint':
            subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
            for t in range(task + 1):
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
                features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
                subgraphs.append(subgraph)
                featuress.append(features)
                labelss.append(labels)
                train_idss.append(train_ids)
                ids_per_clss.append(ids_per_cls)

            # build the dataloader for mini batch training
            train_ids_entire = []
            n_existing_nodes = 0
            for ids, g in zip(train_idss, subgraphs):
                new_ids = [i + n_existing_nodes for i in ids]
                train_ids_entire.extend(new_ids)
                n_existing_nodes += g.num_nodes()
            graph_entire = dgl.batch(subgraphs)
            dataloader = dgl.dataloading.DataLoader(graph_entire, train_ids_entire, args.nb_sampler,
                                                        batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                        drop_last=False)
        else:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
                open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl',
                    'rb'))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

            # build the dataloader for mini batch training
            dataloader = dgl.dataloading.DataLoader(subgraph, train_ids, args.nb_sampler,
                                                        batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                        drop_last=False)
        # Train
        start_time = round(time.time()*1000,3)
        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            elif args.method == 'joint':
                life_model_ins.observe_class_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()
        end_time = round(time.time()*1000,3)
        train_time.append((end_time-start_time)/epochs)

        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        start_time = round(time.time()*1000,3)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")
        end_time = round(time.time()*1000,3)
        infer_time.append((end_time-start_time)/(task+1))
        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}|", end="")
        print(f"time:({round(np.mean(train_time),2)}/{round(np.mean(infer_time),2)})ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)

    return acc_mean, mean_backward, acc_matrix



 
""" 
def pipeline_classIL_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(epochs):
            life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_taskIL_joint(args, valid=False):
    print("taskIL_joint")
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)

    # Load Dataset
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    
    # Load Model
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    data_prepare(args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    train_time = []
    infer_time = []

    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)          

        #train
        start_time = round(time.time()*1000,3)
        for epoch in range(epochs):
            life_model_ins.observe_task_IL(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)
        end_time = round(time.time()*1000,3)     
        train_time.append(end_time-start_time)

        #evaluate
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        start_time = round(time.time()*1000,3)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")
        end_time = round(time.time()*1000,3)
        infer_time.append((end_time-start_time)/(task+1))
        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}|", end="")
        print(f"train_time:{round(np.mean(train_time),2)}ms|", end="")
        print(f"infer_time:{round(np.mean(infer_time),2)}ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)

    return acc_mean, mean_backward, acc_matrix

def pipeline_taskIL_minibatch_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    train_time = []
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        # build the dataloader for mini batch training
        train_ids_entire = []
        n_existing_nodes = 0
        for ids, g in zip(train_idss, subgraphs):
            new_ids = [i + n_existing_nodes for i in ids]
            train_ids_entire.extend(new_ids)
            n_existing_nodes += g.num_nodes()

        graph_entire = dgl.batch(subgraphs)
        dataloader = dgl.dataloading.NodeDataLoader(graph_entire, train_ids_entire, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        # train
        for epoch in range(epochs):
            start_time = round(time.time()*1000,3)
            life_model_ins.observe_task_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)
            end_time = round(time.time()*1000,3)
            train_time.append(end_time-start_time)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}|", end="")
        # print(f"train_time:{round(train_time[-1], 2)}s", end="")
        print(f"train_time:{round(np.mean(train_time),2)}ms", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    # print(f'Train Time: {round(np.sum(train_time), 2)}')
    # print(f"train_time:{round(np.mean(train_time),2)}ms")
    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix


def pipeline_classIL_minibatch_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        # build the dataloader for mini batch training
        train_ids_entire = []
        n_existing_nodes = 0
        for ids, g in zip(train_idss, subgraphs):
            new_ids = [i + n_existing_nodes for i in ids]
            train_ids_entire.extend(new_ids)
            n_existing_nodes += g.num_nodes()

        graph_entire = dgl.batch(subgraphs)
        dataloader = dgl.dataloading.NodeDataLoader(graph_entire, train_ids_entire, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(epochs):
            life_model_ins.observe_class_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        # test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    cls_retain = []
    for clss in args.task_seq:
        cls_retain.extend(clss)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        cls_retain = []
        for clss in args.task_seq[0:task+1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(
            open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        subgraphs = subgraph
        featuress = features
        labelss = labels
        train_idss = train_ids
        ids_per_clss_all = ids_per_cls_all

        for epoch in range(epochs):
            life_model_ins.observe_task_IL_crsedge(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss_all, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t, temp in enumerate(args.task_seq[0:task+1]):
            # cls_ids_new = args.task_seq[t]
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            if cls_ids_new != temp:
                print(
                    '-------------------------------sequence is not as default--------------------------------------------------------')
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels_current_task = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

        cls_retain = []
        for clss in args.task_seq[:task + 1]:
            cls_retain.extend(clss)

        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(
            open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        for epoch in range(epochs):
            life_model_ins.observe_class_IL_crsedge(args, subgraph, features, labels, task, train_ids, ids_per_cls_all, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            # cls_ids_new = args.task_seq[t]
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            if cls_ids_new != args.task_seq[t]:
                print(
                    '-------------------------------sequence is not as default--------------------------------------------------------')
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]

        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids_current_task, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
                torch.cuda.empty_cache()

        # test
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_minibatch_joint(args, valid=False):
    args.method = 'joint_replay_all'
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)

        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(
            open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(epochs):
            life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_all, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            #cls_ids_new = args.task_seq[t]
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            if cls_ids_new!=args.task_seq[t]:
                print('-------------------------------sequence is not as default--------------------------------------------------------')
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels_current_task = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        test_ids = valid_ids_ if valid else test_ids_
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids_current_task, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(epochs):
            if args.method == 'lwf':
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)

        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_minibatch_joint(args, valid=False):
    epochs = args.epochs if valid else 0
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    for task, task_cls in enumerate(args.task_seq):
        name, ite = args.current_model_save_path
        config_name = name.split('/')[-1]
        subfolder_c = name.split(config_name)[-2]
        save_model_name = f'{config_name}_{ite}_{task_cls}'
        save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(
            open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(epochs):
            life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_all, dataset)

        if not valid:
            model = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        test_ids = valid_ids_ if valid else test_ids_ # whether use validation or test set
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            # cls_ids_new = args.task_seq[t]
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            if cls_ids_new != args.task_seq[t]:
                print(
                    '-------------------------------sequence is not as default--------------------------------------------------------')
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        if valid:
            mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
 """