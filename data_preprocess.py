import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *


class Data_class(Dataset):

    def __init__(self, triple):
        self.ent1 = triple[:, 0]
        self.ent2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.ent1[index], self.ent2[index])

def load_data(args, val_ratio=0.1, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data
    print('Loading {0} seed{1} dataset ...'.format(args.in_file[5:8], args.seed))
    all_edges = np.loadtxt(args.in_file, dtype=np.int64)

    G = nx.Graph()
    G.add_edges_from(all_edges)
    num_ents = len(G.nodes)

    train_size = int(all_edges.shape[0] * args.train_ratio)  # 保留多大比例的链接作为训练样本
    np.random.seed(args.seed)
    np.random.shuffle(all_edges)
    remain_positive = all_edges[:train_size]

    G = nx.Graph()
    G.add_edges_from(remain_positive)
    print(nx.info(G))

    # sample negative
    """负采样的方式是否可以改进，因为其采样中可能存在大量非负样本"""
    negative_all = list(nx.non_edges(G))
    np.random.seed(args.seed)
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:all_edges.shape[0]])  # 采样与正样本同样数量的负样本
    print("positive examples: %d, negative examples: %d" % (all_edges.shape[0], negative.shape[0]))

    # split data
    val_size = int(val_ratio * all_edges.shape[0])
    test_size = int(test_ratio * all_edges.shape[0])

    positive = np.concatenate([all_edges, np.ones((all_edges.shape[0], 1), dtype=np.int64)], axis=1)
    negative = np.concatenate([negative, np.zeros((negative.shape[0], 1), dtype=np.int64)], axis=1)

    """train_data中的正样本与训练图中已知的链接信息是一致的"""
    train_data = np.vstack((positive[:-(val_size + test_size)], negative[:-(val_size + test_size)]))
    val_data = np.vstack((positive[-(val_size + test_size): -test_size], negative[-(val_size + test_size): -test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))

    # construct adjacency
    train_positive = positive[:-(val_size + test_size)]
    adj = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                        shape=(num_ents, num_ents), dtype=np.float32)

    # symmetrization
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # build 2-hop adjacency
    adj_s = adj.dot(adj)
    adj_s = adj_s.sign()  # sign函数用以返回稀疏矩阵元素的正负性，若为正，则对应位置为1，若为负，则对应位置为-1，否则为0


    # construct edges for the torch_geometric
    edges_o = adj_o.nonzero()
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

    edges_s = adj_s.nonzero()
    edge_index_s = torch.tensor(np.vstack((edges_s[0], edges_s[1])), dtype=torch.long)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}


    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    validation_set = Data_class(val_data)
    val_loader = DataLoader(validation_set, **params)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)

    # extract features
    print('Extracting features')
    if args.feature_type == 'one_hot':
        features = np.eye(num_ents)
    elif args.feature_type == 'uniform':
        np.random.seed(args.seed)
        features = np.random.uniform(low=0, high=1, size=(num_ents, args.dimensions))
    elif args.feature_type == 'normal':
        np.random.seed(args.seed)
        features = np.random.normal(loc=0, scale=1, size=(num_ents, args.dimensions))
    else:
        features = adj_o.todense()

    features_o = normalize(features)

    args.dimensions = features_o.shape[1]

    # adversarial nodes for DGI
    np.random.seed(args.seed)
    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]

    y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)

    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_index_o)

    data_s = Data(edge_index=edge_index_s)

    x_a = torch.tensor(features_a, dtype=torch.float)
    data_a = Data(x=x_a, y=y_a)  # 特征向量随机置换之后的特征矩阵与标签

    print('Loading finished!')
    return data_o, data_s, data_a, train_loader, val_loader, test_loader