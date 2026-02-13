import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

EPS = 1e-15
MAX_LOGVAR = 10
class ProductOfExperts(torch.nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.
        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar, pd_var

class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.LeakyReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.linear(x)

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, **kwargs):
        super(Base_gcn, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)




class HyperGraphBasicConvolution(nn.Module):
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(2 * input_dim, input_dim)

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper):
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)
        msg = self.aggregation(torch.cat([user_msg, item_msg], dim=1) + torch.cat([group_emb, group_emb], dim=1))

        norm_emb = torch.mm(full_hyper, msg)

        return norm_emb, msg

class HyperGraphParameterEncoder(nn.Module):
    def __init__(self, edges, input_dim, device):
        super(HyperGraphParameterEncoder, self).__init__()
        # mu
        # self.encoder_i_1 = nn.Linear(input_dim, input_dim)
        # self.encoder_u_1 = nn.Linear(input_dim, input_dim)
        # self.encoder_1 = nn.Linear(2 * input_dim, input_dim)
        self.gcn1 = Base_gcn(input_dim, input_dim)
        self.encoder_1 = nn.Linear(input_dim, input_dim)

        # logvar
        # self.encoder_i_2 = nn.Linear(input_dim, input_dim)
        # self.encoder_u_2 = nn.Linear(input_dim, input_dim)
        # self.encoder_2 = nn.Linear(2 * input_dim, input_dim)
        self.gcn2 = Base_gcn(input_dim, input_dim)
        self.encoder_2 = nn.Linear(input_dim, input_dim)

        self.device = device

        self.edges = edges.to(self.device)

    def forward(self, g_emb, i_emb):
        # user_msg = torch.sparse.mm(self.user_hyper_graph, user_emb)
        # item_msg = torch.sparse.mm(self.item_hyper_graph, item_emb)
        x = torch.cat((g_emb, i_emb), dim=0)
        # print(x)
        x = F.normalize(x).to(self.device)

        x1 = F.leaky_relu(self.gcn1(x, self.edges))
        x2 = F.leaky_relu(self.gcn2(x, self.edges))
        mu = self.encoder_1(x1)
        logvar = self.encoder_2(x2)
        return mu, logvar



class HyperGraphConvolution(nn.Module):
    """Hyper-graph Convolution for Member-level hyper-graph"""

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]



    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        final_ui = [torch.cat([user_emb, item_emb], dim=0)]
        final_g = [group_emb]

        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final_ui.append(emb)
            final_g.append(he_msg)

        final_ui = torch.sum(torch.stack(final_ui), dim=0)
        final_g = torch.sum(torch.stack(final_g), dim=0)

        # for i in range(1):
        #     x = torch.sparse.mm(overlap_graph, final_g)
        # final_g = final_g + x

        return final_ui, final_g


class DGGVAE(nn.Module):
    """DGGVAE"""
    def __init__(self, num_users, num_items, num_groups, args, user_hyper_graph, item_hyper_graph,
                 full_hyper, overlap_graph, device, edges, kl_weight, g_layers, cl_weight, temp):
        super(DGGVAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        # Hyper-parameters
        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        self.experts = ProductOfExperts()
        self.kl_weight = kl_weight
        self.cl_weight = cl_weight
        self.g_layers = g_layers
        self.temp = temp

        self.interaction = edges.astype(np.float32)
        edge_index = self.pack_edge_index(self.interaction)
        self.edges = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edges = torch.cat((self.edges, self.edges[[1, 0]]), dim=1)

        self.overlap_graph = overlap_graph
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        # Embedding init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        # Hyper-graph Convolution
        self.hyper_graph_conv = HyperGraphConvolution(user_hyper_graph, item_hyper_graph, full_hyper, self.layers,
                                                      self.emb_dim, device)
        # encoder
        self.encode1 = HyperGraphParameterEncoder(self.edges, self.emb_dim, self.device).to(self.device)

        self.encode2 = HyperGraphParameterEncoder(self.edges, self.emb_dim, self.device).to(self.device)

        # Prediction Layer
        self.predict = PredictLayer(self.emb_dim)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.num_groups
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def reparametrize(self, mu, logvar):
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            return mu + torch.randn_like(logvar) * 0.1 * torch.exp(logvar.mul(0.5))
        else:
            return mu

    def forward(self, user_inputs, group_inputs, pos_item_inputs, neg_item_inputs, members, items, mode, type):
        if type == 'group':
            return self.group_forward(user_inputs, group_inputs, pos_item_inputs, neg_item_inputs, members, items, mode)
        else:
            return self.user_forward(user_inputs, pos_item_inputs, neg_item_inputs, members, items, mode)

    def group_forward(self, user_inputs, group_inputs, pos_item_inputs, neg_item_inputs, members, items, mode):
        # Member-level graph computation
        ui_emb, g_emb = self.hyper_graph_conv(self.user_embedding.weight, self.item_embedding.weight,
                                              self.group_embedding.weight, self.num_users, self.num_items)
        u_emb, i_emb = torch.split(ui_emb, [self.num_users, self.num_items])

        # means = self.get_means_u(members, u_emb)
        means = self.get_means_ui(members, u_emb, items, i_emb)

        for i in range(self.g_layers):
            g_emb = torch.sparse.mm(self.overlap_graph, g_emb)
            means = torch.sparse.mm(self.overlap_graph, means)

        mu1, logvar1 = self.encode1(g_emb, i_emb)
        mu2, logvar2 = self.encode2(means, i_emb)


        # entire DGGVAE
        mu = torch.stack([mu1, mu2], dim=0)
        logvar = torch.stack([logvar1, logvar2], dim=0)
        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar

        # reparametrize
        z = self.reparametrize(pd_mu, pd_logvar)
        z1 = self.reparametrize(mu1, logvar1)
        z2 = self.reparametrize(mu2, logvar2)
        z_g, z_i = torch.split(z, [self.num_groups, self.num_items])
        z1_g, z1_i = torch.split(z1, [self.num_groups, self.num_items])
        z2_g, z2_i = torch.split(z2, [self.num_groups, self.num_items])
        z_g_f = z_g + z1_g + z2_g
        z_i_f = z_i + z1_i + z2_i

        if mode == 'eval':
            loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs], z_i_f[neg_item_inputs])
            return loss, pos_prediction

        else:
            bpr_loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs], z_i_f[neg_item_inputs])
            kl_loss = self.kl_loss(pd_mu, pd_logvar)
            cl_loss_z = self.InfoNCE(z1, z2, temp=self.temp)
            cl_loss_e = self.InfoNCE(g_emb, means, temp=self.temp)
            cl_loss = cl_loss_e + cl_loss_z
            # kl_loss1 = self.kl_loss(mu1, logvar1)
            # kl_loss2 = self.kl_loss(mu2, logvar2)
            loss = bpr_loss + cl_loss * self.cl_weight + kl_loss * self.kl_weight
            return loss, pos_prediction
        del z1, z2, z, z_g, z1_g, z1_i, z2, z2_g, z2_i

        # coarse only
        # z = self.reparametrize(mu1, logvar1)
        # z_g_f, z_i_f = torch.split(z, [self.num_groups, self.num_items])
        # if mode == 'eval':
        #     loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs], z_i_f[neg_item_inputs])
        #     return loss, pos_prediction
        #
        # else:
        #     bpr_loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs], z_i_f[neg_item_inputs])
        #     kl_loss = self.kl_loss(mu1, logvar1)
        #     # cl_loss_z = self.InfoNCE(z1, z2, temp=self.temp)
        #     # cl_loss_e = self.InfoNCE(g_emb, means, temp=self.temp)
        #     # cl_loss = cl_loss_e + cl_loss_z
        #     # kl_loss1 = self.kl_loss(mu1, logvar1)
        #     # kl_loss2 = self.kl_loss(mu2, logvar2)
        #     loss = bpr_loss + kl_loss * self.kl_weight
        #     return loss, pos_prediction
        # del z

        # fine only
        # z = self.reparametrize(mu2, logvar2)
        # z_g_f, z_i_f = torch.split(z, [self.num_groups, self.num_items])
        # if mode == 'eval':
        #     loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs], z_i_f[neg_item_inputs])
        #     return loss, pos_prediction
        #
        # else:
        #     bpr_loss, pos_prediction = self.BPR_loss(z_g_f[group_inputs], z_i_f[pos_item_inputs],
        #                                              z_i_f[neg_item_inputs])
        #     kl_loss = self.kl_loss(mu1, logvar1)
        #     # cl_loss_z = self.InfoNCE(z1, z2, temp=self.temp)
        #     # cl_loss_e = self.InfoNCE(g_emb, means, temp=self.temp)
        #     # cl_loss = cl_loss_e + cl_loss_z
        #     # kl_loss1 = self.kl_loss(mu1, logvar1)
        #     # kl_loss2 = self.kl_loss(mu2, logvar2)
        #     loss = bpr_loss + kl_loss * self.kl_weight
        #     return loss, pos_prediction
        # del z

    def get_means_u(self, members, u_emb):
        u_means = torch.empty(0).to(self.device)
        for member in members:
            embedding_member = torch.index_select(u_emb, 0, member)
            u_mean = self.get_mean_u(embedding_member)
            u_means = torch.cat((u_means, torch.unsqueeze(u_mean, dim=0)), dim=0)
        return u_means

    def get_means_ui(self, members, u_emb, items, i_emb):
        means = torch.empty(0).to(self.device)
        for member, item in zip(members, items):
            embedding_member = torch.index_select(u_emb, 0, member)
            embedding_item = torch.index_select(i_emb, 0, item)
            u_mean = self.get_mean_u(embedding_member)
            i_mean = self.get_mean_i(embedding_item)
            mean = (u_mean + i_mean) / 2
            means = torch.cat((means, torch.unsqueeze(mean, dim=0)), dim=0)
        return means

    def get_mean_u(self, embedding_member):
        """Geometric bounding and projection for group representation"""
        # u_max = torch.mean(embedding_member, dim=0).values
        # u_min = torch.min(embedding_member, dim=0).values
        mean_var = torch.mean(embedding_member, dim=0)
        return mean_var

    def get_mean_i(self, embedding_item):
        """Geometric bounding and projection for group representation"""
        # u_max = torch.mean(embedding_member, dim=0).values
        # u_min = torch.min(embedding_member, dim=0).values
        mean_var = torch.mean(embedding_item, dim=0)
        return mean_var

    def BPR_loss(self, g_emb, i_emb_pos, i_emb_neg):
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(g_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(g_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(g_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(g_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction

    def kl_loss(self, mu, logvar):
        logvar = logvar.clamp(max = MAX_LOGVAR)
        return 0.5 * torch.mean(torch.sum(-1 - logvar + mu.pow(2) + (logvar.exp()).pow(2), dim=1))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def user_forward(self, user_inputs, pos_item_inputs, neg_item_inputs, members, items, mode):
        u_emb = self.user_embedding(user_inputs)
        i_emb_pos = self.item_embedding(pos_item_inputs)
        i_emb_neg = self.item_embedding(neg_item_inputs)
        if self.predictor_type == "MLP":
            pos_prediction = torch.sigmoid(self.predict(u_emb * i_emb_pos))
            neg_prediction = torch.sigmoid(self.predict(u_emb * i_emb_neg))
        else:
            pos_prediction = torch.sum(u_emb * i_emb_pos, dim=-1)
            neg_prediction = torch.sum(u_emb * i_emb_neg, dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        return loss, pos_prediction
