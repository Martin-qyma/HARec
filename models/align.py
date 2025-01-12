import torch
import pickle
import manifolds
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HGCN(nn.Module):
    def __init__(self, c, args):
        super(HGCN, self).__init__()
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.c = c
        self.dim = args.dim
        self.num_gcn_layers = args.num_layers - 1
    
    def resSumGCN_encode(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            x = torch.spmm(adj, output[i])
            output.append(x)
        output_tan = sum(output[1:])

        h = self.manifold.proj(self.manifold.expmap0(output_tan, c=self.c), c=self.c)
        return h

    def resAddGCN_encode(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        output_tan = output[-1]
        h = self.manifold.proj(self.manifold.expmap0(output_tan, c=self.c), c=self.c)
        return h


class HARec(nn.Module):
    def __init__(self, users_items, args):
        super(HARec, self).__init__()

        self.c = torch.tensor([args.c]).to(device)
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.encoder = HGCN(self.c, args)
        self.num_users, self.num_items = users_items
        self.num_layers = args.num_layers
        self.args = args
        self.num_samples = args.num_neg

        with open(f'./data/{self.args.dataset}/user_semantic_embed.pkl', 'rb') as f:
            semantic_user_emb = pickle.load(f)
        with open(f'./data/{self.args.dataset}/item_semantic_embed.pkl', 'rb') as f:
            semantic_item_emb = pickle.load(f)
        self.semantic_embeddings = torch.cat((semantic_user_emb, semantic_item_emb), dim=0).float().cuda()
        
        self.mlp = nn.Sequential(
            nn.Linear(self.semantic_embeddings.shape[1], (self.semantic_embeddings.shape[1] + self.args.dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.semantic_embeddings.shape[1] + self.args.dim) // 2, self.args.dim)
        )

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items, embedding_dim=args.dim).to(device)
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def encode(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
            adj = adj.to(device)
            x = x.to(device)
        h = self.encoder.resSumGCN_encode(x, adj)
        return h

    def decode(self, h, idx):
        if isinstance(h, tuple):
            h = h[0]
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]

        sqdist_h = self.manifold.sqdist(emb_in, emb_out, self.c)
        return sqdist_h

    def hratio(self, user, item):
        # Separate the first column from the rest of the tensor
        user0 = user[:, 0]
        item0 = item[:, 0]

        # Calculate the factor to multiply with the score
        factor = 1.0 / (user0 * item0)
        # Calculate the score using the Minkowski dot product and subtract user0 and item0
        score = 1 - self.manifold.minkowski_dot(user, item, keepdim=False) - user0 - item0

        # Multiply by the factor calculated earlier and negate the resulting tensor
        score = score * factor
        # Also add an extra dimension to obtain a 2D tensor
        return -score.view(-1, 1)

    def decode_pos(self, h, idx):
        emb_in = h[idx[0], :]
        emb_out = h[idx[1], :]

        sqdist_h = self.manifold.sqdist(emb_in, emb_out, self.c)
        w = self.hratio(emb_in, emb_out).sigmoid()
        return w, sqdist_h

    def decode_neg(self, h, idx):
        # Unpack anchor, negative, and positive indices
        anchor, neg, pos = idx

        # Get embeddings for the anchor and positive examples
        emb_anchor = h[anchor, :]
        emb_pos = h[pos, :]

        # Compute pairwise L2 distances between each negative example and the positive example,
        # here we can also compute the hyperbolic distance
        pos_neg_dist = ((h[neg] - emb_pos.unsqueeze(1)) ** 2).sum(dim=-1)

        # Select the closest negative example (i.e., with the smallest L2 distance to the positive example)
        hard_idx = pos_neg_dist.min(dim=1).indices.view(-1, 1)
        hard_neg = torch.gather(neg.cuda(), 1, hard_idx).squeeze()
        emb_hard_neg = h[hard_neg,]

        # Compute squared distance between anchor and closest negative example in hyperbolic space
        sqdist_neg = self.manifold.sqdist(emb_anchor, emb_hard_neg, self.c)

        return sqdist_neg

    def negative_sampling(self, edge_index):
        # 1. get i,j as long tensor from the edge_index
        i, j = edge_index.long()
        num_nodes = self.num_users + self.num_items  # total number of nodes
        idx_1 = i * num_nodes + j  # The index for i and j paired node

        i = i.repeat(self.num_samples)  # repeat i based on number of samples
        # choose k randomly (uniform probability distribution) from the nodes
        k = torch.randint(self.num_users, num_nodes, (i.size(0),), dtype=torch.long)  # (e*t)
        idx_2 = i * num_nodes + k  # The index for i and k paired node, k is the index of possible negative samples

        # 2. after we get possible negative samples and then filter the invalid candidates
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(
            torch.bool)  # (e*t,), create a boolean tensor that checks for matching indices
        rest = mask.nonzero(as_tuple=False).view(-1)  # retain a tensor of all non-zero elements

        # 3. while still any zero element is found in the mask tensor
        while rest.numel() > 0:
            # Sample random number of items to replace indexes that conflicts
            tmp = torch.randint(self.num_users, num_nodes, (rest.numel(),),
                                dtype=torch.long)  # sample from the item set
            idx_2 = i[rest] * num_nodes + tmp
            mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)  # update boolean tensor again
            k[rest] = tmp  # update value of 'k'
            rest = rest[mask.nonzero(as_tuple=False).view(-1)]
        assert k.min() > self.num_users - 1 and k.max() < num_nodes

        # return new negative sample values
        # edge_index[0] is index for anchor, edge_index[1] is index for positive neighbor
        # k represents the index of negative samples
        return edge_index[0], edge_index[1], k.unsqueeze(0).reshape(self.num_samples, -1).transpose(1, 0).to(edge_index.device)

    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]
        anchor, pos, neg = self.negative_sampling(torch.from_numpy(np.array(train_edges)).transpose(1, 0))

        w, pos_scores = self.decode_pos(embeddings, (anchor, pos))
        neg_scores = self.decode_neg(embeddings, (anchor, neg, pos))
        loss_dist = pos_scores - neg_scores + self.args.margin * w
        loss_dist[loss_dist < 0] = 0
        loss_dist = torch.sum(loss_dist)

        # contrastive alignment loss
        semantic_embeddings = self.mlp(self.semantic_embeddings)
        semantic_embeddings = self.manifold.proj(self.manifold.expmap0(semantic_embeddings, c=self.c), c=self.c)
        loss_align = torch.sum(self.manifold.sqdist(embeddings[anchor, :], semantic_embeddings[anchor, :], self.c))
        loss_align += torch.sum(self.manifold.sqdist(embeddings[pos, :], semantic_embeddings[pos, :], self.c))
        loss_align *= self.args.align_weight

        loss = loss_dist + loss_align
        return loss, loss_dist, loss_align

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))

        for uid in range(num_users):
            emb_in = h[uid, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[uid] = np.reshape(probs, [-1, ])

        return probs_matrix

