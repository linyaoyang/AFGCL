import numpy as np
import torch
import copy
import faiss
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 一级编码器
class Encoder(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        self.stacked_gnn = nn.ModuleList(
            [GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)
        return x

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        self.device = args.device
        self.num_centroids = args.num_centroids
        self.num_kmeans = args.num_kmeans
        self.clus_num_iters = args.clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:, 0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long()

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn,
                                                              I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)

class MGCL(nn.Module):
    def __init__(self, layer_config, args):
        super(MGCL, self).__init__()
        self.student_encoder = Encoder(layer_config=layer_config)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        self.neighbor = Neighbor(args)

        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(), nn.Linear(args.pred_hid, rep_dim))

        self.student_predictor.apply(init_weights)
        self.topk = args.topk
        self.decoder1 = nn.Linear(rep_dim * 4, args.decoder_dim1)  # 将上层编码器的输出变换后用于链接预测的分类
        self.decoder2 = nn.Linear(args.decoder_dim1, 1)


    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, y, edge_index, inp=None):
        student = self.student_encoder(x=x, edge_index=edge_index)
        ent1 = student[inp[0]]
        ent2 = student[inp[1]]
        add = ent1 + ent2
        product = ent1 * ent2
        concatenate = torch.cat((ent1, ent2), dim=1)
        pred = self.student_predictor(student)
        feature = torch.cat((add, product, concatenate), dim=1)
        log = F.relu(self.decoder1(feature))
        log = self.decoder2(log)

        with torch.no_grad():
            teacher = self.teacher_encoder(x=x, edge_index=edge_index)


        adj = torch.sparse.FloatTensor(edge_index, torch.ones_like(edge_index[1]), [x.shape[0], x.shape[0]])

        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)
        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss = loss1 + loss2

        return pred, ind, k, log, loss.mean()