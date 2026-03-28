import torch
import torch.nn as nn


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(3)
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HGR(nn.Module):
    def __init__(self, num_users, num_items, num_groups, emb_dim, layers, drop_ratio, adj, D, A, group_member_dict, device):
        super(HGR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.layers = layers
        self.drop_ratio = drop_ratio
        self.adj = adj
        self.D = D
        self.A = A
        self.group_member_dict = group_member_dict
        member_index, member_mask = self._build_group_member_tensors(num_groups, group_member_dict)
        self.register_buffer('group_member_index', member_index)
        self.register_buffer('group_member_mask', member_mask)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)
        self.hyper_graph = HyperConv(self.layers)
        self.group_graph = GroupConv(self.layers)
        self.attention = AttentionLayer(2 * self.emb_dim, self.drop_ratio)
        self.predict = PredictLayer(3 * self.emb_dim, self.drop_ratio)
        self.device = device

        self.gate = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), nn.Sigmoid())

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

    @staticmethod
    def _build_group_member_tensors(num_groups, group_member_dict):
        max_len = max((len(members) for members in group_member_dict.values()), default=1)
        member_index = torch.zeros((num_groups, max_len), dtype=torch.long)
        member_mask = torch.ones((num_groups, max_len), dtype=torch.bool)

        for group_id in range(num_groups):
            members = group_member_dict.get(group_id, [])
            if not members:
                continue
            member_len = len(members)
            member_index[group_id, :member_len] = torch.tensor(members, dtype=torch.long)
            member_mask[group_id, :member_len] = False
        return member_index, member_mask


    def forward(self, group_inputs, user_inputs, item_inputs):


        if (group_inputs is not None) and (user_inputs is None):
            ui_embedding = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            ui_embedding = self.hyper_graph(self.adj, ui_embedding)
            user_embedding, item_embedding = torch.split(ui_embedding, [self.num_users, self.num_items], dim=0)
            item_emb = item_embedding[item_inputs]

            group_embedding = self.group_graph(self.group_embedding.weight, self.D, self.A)

            member_masked = self.group_member_index[group_inputs]
            mask = self.group_member_mask[group_inputs]
            bsz, max_len = member_masked.shape

            member_emb = user_embedding[member_masked]
            # attention aggregation
            item_emb_attn = item_emb.unsqueeze(1).expand(bsz, max_len, -1)
            at_emb = torch.cat((member_emb, item_emb_attn), dim=2)
            at_wt = self.attention(at_emb, mask)
            g_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), member_emb).squeeze()

            # mean aggregation
            # at_wt = torch.ones(bsz, max_len).to(self.device)
            # at_wt.masked_fill_(mask.bool(), -np.inf)
            # at_wt = torch.softmax(at_wt, dim=1)
            # g_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), member_emb).squeeze()

            g_emb_pure = group_embedding[group_inputs]
            group_emb = g_emb_with_attention + g_emb_pure
            # g_weight = self.gate(torch.cat((g_emb_with_attention, g_emb_pure), dim=1))
            # group_emb = g_weight * g_emb_with_attention + (1 - g_weight) * g_emb_pure
            # group_emb = g_emb_with_attention

            element_emb = torch.mul(group_emb, item_emb)

            new_emb = torch.cat((element_emb, group_emb, item_emb), dim=1)
            y = self.predict(new_emb)
            # y = torch.matmul(group_emb.unsqueeze(1), item_emb.unsqueeze(2)).squeeze()
            return y

        else:
            user_emb = self.user_embedding(user_inputs)
            item_emb = self.item_embedding(item_inputs)
            element_emb = torch.mul(user_emb, item_emb)
            new_emb = torch.cat((element_emb, user_emb, item_emb), dim=1)
            y = self.predict(new_emb)
            # y = torch.matmul(user_emb.unsqueeze(1), item_emb.unsqueeze(2)).squeeze()
            return y

class HyperConv(nn.Module):
    def __init__(self, layers):
        super(HyperConv, self).__init__()
        self.layers = layers

    def forward(self, adj, embedding):
        all_emb = embedding
        final = [all_emb]
        for i in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            final.append(all_emb)
        final_emb = torch.stack(final, dim=0).sum(dim=0)
        return final_emb

class GroupConv(nn.Module):
    def __init__(self, layers):
        super(GroupConv, self).__init__()
        self.layers = layers

    def forward(self, embedding, D, A):
        DA = torch.mm(D, A).float()
        group_emb = embedding
        final = [group_emb]
        for i in range(self.layers):
            group_emb = torch.mm(DA, group_emb)
            final.append(group_emb)
        final_emb = torch.stack(final, dim=0).sum(dim=0)
        return final_emb

class AttentionLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim / 2)),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(int(emb_dim / 2), 1)
        )

    def forward(self, x, mask):
        bsz = x.shape[0]
        out = self.linear(x)
        out = out.view(bsz, -1) # [bsz, max_len]
        out.masked_fill_(mask, float('-inf'))
        weight = torch.softmax(out, dim=1)
        return weight

class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

