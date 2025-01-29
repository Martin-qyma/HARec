'''This file allows users to manually balance exploration and exploitation by navigating the hierarchy tree.'''
import torch
import random
import pickle
import manifolds
import numpy as np
from utils.config import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Balance():
    def __init__(self, data):
        super(Balance, self).__init__()
        self.data = data
        self.manifold = manifolds.Hyperboloid()
        self.c = torch.tensor([args.c]).to(device)
        self.tree = pickle.load(open(f"./data/{args.dataset}/hierarchy_tree.pkl", "rb"))
        self.user_embed = pickle.load(open(f"./data/{args.dataset}/user_embed.pkl", "rb"))
        self.item_embed = pickle.load(open(f"./data/{args.dataset}/item_embed.pkl", "rb"))
        # Calculate the max layer of the tree
        n_samples = self.user_embed.shape[0] + self.item_embed.shape[0]
        self.max_layer = 0
        while n_samples > 1:
            n_samples = max(int(n_samples / args.k), 0)
            self.max_layer += 1
        self.index = {}  # To store (data, layer) -> node mappings

    def build_index(self, node):
        """Builds the index for quick lookups."""
        key = (tuple(node.data[0].tolist()), node.layer)
        self.index[key] = node
        for child in node.children:
            self.build_index(child)

    def predict(self, temperature, layer):
        '''
        Navigate the hierarchical tree structure for recommendation.

        Parameters:
        - prediction: Dictionary of recommendations, key: user index, value: list of item indices (50).
        - temperature: Proportion of exploration nodes in the recommendation.
        - layer: layer of the hierarchical tree to explore.
        '''
        prediction = pickle.load(open(f"./data/{args.dataset}/prediction.pkl", "rb"))
        self.build_index(self.tree)
        for uid in prediction.keys():
            user_embed = self.user_embed[uid]
            # Find the node corresponding to the user embedding
            key = (tuple(user_embed.tolist()), self.max_layer)
            user_node = self.index.get(key, None)
            if user_node is None:
                continue
            # Find the ancestors of the user node, return [bottom, ..., root]
            ancestors = self.tree.get_ancestors(user_node)
            explore_root = ancestors[self.max_layer - layer]
            # Find the item index corresponding to the explore root
            item_matches = (self.item_embed.unsqueeze(0) == explore_root.data.detach().unsqueeze(1)).all(dim=2)
            candidates = torch.nonzero(item_matches)[:, 1]
            # Calculate the number of exploration nodes, consider the first k=20 recommendations
            num_explore = min(int(temperature * 20), len(candidates))
            # Randomly select exploration nodes from the candidates except the original recommendations
            random_indices = torch.randperm(candidates.size(0))[:num_explore]
            explore_nodes = candidates[random_indices].cpu().tolist()
            # exclude the original recommendations
            original_recommendations = prediction[uid][:20]
            explore_nodes = [e for e in explore_nodes if e not in original_recommendations]
            # Replace the recommendations with the exploration nodes
            prediction[uid][20-len(explore_nodes):20] = explore_nodes
        return prediction

    def find_lca_layer(self, embed1, embed2):
        """
        Finds the layer of the lowest common ancestor (LCA) of two nodes in a cluster tree.
        """
        self.build_index(self.tree)
        node1 = self.index.get((tuple(embed1.tolist()), self.max_layer), None)
        node2 = self.index.get((tuple(embed2.tolist()), self.max_layer), None)
        # Trace paths to the root for both nodes
        
        path1 = []
        while node1:
            path1.append(node1)
            node1 = node1.parent

        path2 = []
        while node2:
            path2.append(node2)
            node2 = node2.parent

        # Reverse paths to start from root
        path1 = path1[::-1]
        path2 = path2[::-1]

        # Find the last common node
        lca_layer = 0
        for n1, n2 in zip(path1, path2):
            if n1 == n2:
                lca_layer = n1.layer
            else:
                break

        return lca_layer

    def find_index_with_lca(self, start_embed, lca_layer):
        """
        Finds an item index that shares the specified LCA layer with the start node.

        Returns:
            list: The index of the node that shares the specified LCA, or None if not found.
        """
        self.build_index(self.tree)
        start_node = self.index.get((tuple(start_embed.tolist()), self.max_layer), None)

        # Step 1: Traverse up to the node at the specified LCA layer
        current_node = start_node
        while current_node and current_node.layer != lca_layer:
            current_node = current_node.parent

        if not current_node:
            return None  # LCA at specified layer doesn't exist

        # Step 2: Search among the descendants of the LCA node for a sibling
        def find_sibling(node, target_node):
            if node == target_node:
                return None  # Skip the start node itself

            indices = np.arange(node.data.shape[0])
            np.random.shuffle(indices)
            for i in indices:
                matches = (self.item_embed == node.data[i]).all(dim=1)
                if matches.any():
                    return matches.nonzero(as_tuple=True)[0].item()

            for child in node.children:
                result = find_sibling(child, target_node)
                if result:
                    return result
            return None

        # Search siblings for an embedding that matches
        for child in current_node.children:
            index = find_sibling(child, start_node)
            if index:
                return index

        return None  # No matching sibling found

