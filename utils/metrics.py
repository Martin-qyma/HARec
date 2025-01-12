import math
import torch
import pickle
import numpy as np
from collections import Counter
from manifolds import Hyperboloid


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        if len(act_set) != 0:
            pred_set = set(predicted[i][:topk])
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_at_k(test_dict, pred_list, k):
    ndcg_scores = []
    
    for user_idx, user in enumerate(test_dict):
        true_relevant_items = set(test_dict[user])
        if len(true_relevant_items) == 0:
            continue
        pred_items = pred_list[user_idx][:k]  # Get top-k predictions for this user
        
        # Calculate DCG
        dcg = 0.0
        for rank, item in enumerate(pred_items):
            if item in true_relevant_items:
                dcg += 1 / np.log2(rank + 2)  # rank + 2 because log2 starts at rank 1
        
        # Calculate IDCG
        ideal_relevance_count = min(len(true_relevant_items), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_relevance_count))
        
        # Calculate NDCG
        if idcg != 0:
            ndcg_scores.append(dcg / idcg)

    # Return the average NDCG score across all users
    avg_ndcg = np.mean(ndcg_scores)
    
    return avg_ndcg 

def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = -np.inf
    ind = np.argpartition(pred_matrix, -topk)   
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]
    
    recall = []
    ndcg = []
    for k in [10, 20]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))
        ndcg.append(ndcg_at_k(data.test_dict, pred_list, k))
    return recall, ndcg

def Diversity(prediction, item_emb, k):
    '''
    prediction: dictionary, key: user_id, value: list of recommended items
    '''
    user_dist = {}
    manifold = Hyperboloid()
    c = torch.tensor([1]).cuda()
    for uid in prediction.keys():
        dist = 0
        rec_items = prediction[uid][:k]
        rec_items_emb = item_emb[rec_items]
        for x in rec_items_emb:
            tensor1_list = x.repeat(k, 1)
            tensor2_list = rec_items_emb
            dist += manifold.sqdist(tensor1_list, tensor2_list, c).mean()
        user_dist[uid] = (dist / rec_items_emb.shape[0]).item()
    # normalize the diversity
    min_diversity = min(user_dist.values())
    max_diversity = max(user_dist.values())
    diversity = sum(user_dist.values()) / len(user_dist)
    diversity = (diversity - min_diversity) / (max_diversity - min_diversity)

    return diversity

def Shannon_entropy(labels):
    """
    Calculate the Shannon Entropy for a list of labels.
    Args:
        labels (list): List of recommended items
    Returns:
        float: Shannon Entropy.
    """
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Calculate the probability of each label
    probabilities = [count / len(labels) for count in label_counts.values()]
    
    # Calculate the Shannon entropy
    entropy = -sum([p * math.log2(p) for p in probabilities if p > 0])
    
    return entropy

def EPC(recommended_items, item_popularity):
    """
    Calculate the Expected Popularity Complement (EPC) for a list of recommended items.
    Higher epc -> more diversity
    Args:
        recommended_items (list): List of recommended item IDs.
        item_popularity (dict): Dictionary mapping item IDs to their popularity (e.g., number of interactions).
    Returns:
        float: EPC value.
    
    # Example usage
    recommended_items = ['item1', 'item2', 'item3']
    item_popularity = {'item1': 100, 'item2': 50, 'item3': 25, 'item4': 200}
    epc_value = epc(recommended_items, item_popularity)
    """
    # Maximum popularity in the item popularity dictionary
    max_pop = max(item_popularity.values())
    
    # Calculate the EPC
    epc_value = 0
    for item in recommended_items:
        # Get the popularity of the current item
        pop = item_popularity.get(item, 0)
        # Calculate the complement of its popularity
        epc_value += 1 - (pop / max_pop)
    
    # Average over the number of recommended items
    epc_value /= len(recommended_items)
    
    return epc_value
