import os
import time
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from utils.helper import sparse_mx_to_torch_sparse_tensor, normalize


class Data(object):
    def __init__(self, dataset):
        pkl_path = os.path.join('./data/' + dataset)
        self.dataset = dataset   
        self.user_item_list = pickle.load(open(os.path.join(pkl_path, 'user_item_list.pkl'), 'rb'))
        self.train_dict = pickle.load(open(os.path.join(pkl_path, 'train.pkl'), 'rb'))
        self.test_dict = pickle.load(open(os.path.join(pkl_path, 'test.pkl'), 'rb'))
        self.num_users, self.num_items = len(self.user_item_list), max([max(x) for x in self.user_item_list]) + 1
        self.adj_train, user_item = self.generate_adj()

        # normalize the adjacency matrix
        self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
        self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)

        tot_num_rating = sum([len(x) for x in self.user_item_list])
        self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)

    def generate_adj(self):
        user_item = np.zeros((self.num_users, self.num_items)).astype(int)
        for i, v in self.train_dict.items():
            user_item[i][v] = 1
        coo_user_item = sp.coo_matrix(user_item)
        start = time.time()
        start = time.time()
        rows = np.concatenate((coo_user_item.row, coo_user_item.transpose().row + self.num_users))
        cols = np.concatenate((coo_user_item.col + self.num_users, coo_user_item.transpose().col))
        data = np.ones((coo_user_item.nnz * 2,))
        adj_csr = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)
        return adj_csr, user_item

    def generate_inverse_mapping(self, mapping):
        inverse_mapping = dict()
        for inner_id, true_id in enumerate(mapping):
            inverse_mapping[true_id] = inner_id
        return inverse_mapping

    def generate_rating_matrix(self, train_set, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        return rating_matrix
    
    def generate_triples(self, neg_num):
        # adj_train is a sparse matrix, generate triples from it: [user, item, negtive_item]
        triples = []
        for i in range(self.num_users):
            for j in self.train_dict[i]:
                pair = [i, j]
                pair.extend(self.sample_negative(i, neg_num))
                triples.append(pair)
        return torch.tensor(triples).cuda()
    
    def sample_negative(self, uid, neg_num):
        neg_list = []
        for i in range(neg_num):
            neg_item = np.random.randint(self.num_items)
            while neg_item in self.train_dict[uid]:
                neg_item = np.random.randint(self.num_items)
            neg_list.append(neg_item)
        return neg_list
            
        

