import torch
import numpy as np
import scipy.sparse as sp


def default_device(device_id=0) -> torch.device:
    return torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')


def normalize(mx):
    """Row-normalize sparse matrix."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """https://github.com/HazyResearch/hgcn/blob/a526385744da25fc880f3da346e17d0fe33817f8/utils/data_utils.py"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def min_max_scaling_with_clipping(values, lower_percentile=0.1, upper_percentile=99.9):
    values = values.cpu().detach().numpy()
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)

    # Clipping the values to the specified percentiles
    values_clipped = np.clip(values, lower_bound, upper_bound)

    # Normalizing the clipped values
    min_val = np.min(values_clipped)
    max_val = np.max(values_clipped)

    normalized_values = (values_clipped - min_val) / (max_val - min_val)
    return torch.tensor(normalized_values)

def test_output_format(recall, ndcg):
    output = '\tR@10\tR@20\tN@10\tN@20\n'
    for i in range(2):
        output += '\t{:.4f}'.format(recall[i])
    for i in range(2):
        output += '\t{:.4f}'.format(ndcg[i])
    return output

def hierarchy_output_format(recall, ndcg, diversity, shannon, epc):
    output = '\tR@10\tR@20\tN@10\tN@20\tDiv@10\tDiv@20\tH@10\tH@20\tEPC@10\tEPC@20\n'
    for i in range(2):
        output += '\t{:.4f}'.format(recall[i])
    for i in range(2):
        output += '\t{:.4f}'.format(ndcg[i])
    for i in range(2):
        output += '\t{:.4f}'.format(diversity[i])
    for i in range(2):
        output += '\t{:.4f}'.format(shannon[i])
    for i in range(2):
        output += '\t{:.4f}'.format(epc[i])
    return output


