'''This file builds a hierarchical tree structure by clustering on hyperbolic embeddings.'''
import torch
import pickle
import manifolds
import numpy as np
from utils.math_utils import arcosh

class ClusterNode:
    def __init__(self, data, layer=0, label=None, parent=None):
        self.data = data        # Data points in this cluster
        self.children = []      # Sub-clusters
        self.centroid = None    # Centroid of the cluster
        self.layer = layer      # Layer in the tree
        self.parent = parent    # Reference to the parent node
    
    def build_index(self):
        """Builds the index for quick lookups."""
        def add_to_index(node):
            key = (node.data, node.layer)
            self.index[key] = node
            for child in node.children:
                add_to_index(child)
        
        # Populate the index recursively starting from this node
        add_to_index(self)
    
    def find_node_by_data(self, target_data, target_layer):
        """Search using the pre-built index."""
        key = (target_data, target_layer)
        return self.index.get(key, None)
    
    def get_nodes_at_layer(self, target_layer):
        '''
        IMPORTANT: This function only works for the root node!
        '''
        nodes_at_layer = []
        if self.layer == target_layer:
            nodes_at_layer.append(self)

        # Recursively check the children nodes
        for child in self.children:
            nodes_at_layer.extend(child.get_nodes_at_layer(target_layer))
        
        return nodes_at_layer

    def get_ancestors(self, node):
        """
        Finds the node with the given data and traces its ancestors.
        Returns a list of ancestors (including the root node).
        """
        ancestors = []
        
        # Trace the ancestors if the node was found
        if node:
            while node:
                ancestors.append(node)
                node = node.parent
        return ancestors


class HierarchicalClustering:
    def __init__(self, X, k):
        self.k = k
        self.manifold = manifolds.Hyperboloid()
        n_samples, n_features = X.shape
        # Calculate the number of levels, level n has 2^(n-1) clusters
        layer = 0
        while n_samples > 1:
            n_samples = max(int(n_samples / self.k), 0)
            layer += 1
        self.max_layer = layer
        
        # Start the recursive hierarchical clustering
        self.tree = self.hierarchical_clustering(X, layer=0)

    def assign_labels(self, X):
        labels_list = []
        c = torch.tensor([1.]).cuda()
        if X.shape[0] % self.k != 0:
            target_size = X.shape[0] // self.k + 1
        else:
            target_size = X.shape[0] // self.k
        cluster_counts = torch.zeros(self.k).cuda()

        # Assign each point to the closest centroid
        for x in X:
            tensor1_list = x.repeat(self.centroids.shape[0], 1)
            tensor2_list = self.centroids
            dist = self.manifold.sqdist(tensor1_list, tensor2_list, c)
            labels = torch.argmin(dist, axis=0)
            while cluster_counts[labels.item()] >= target_size:
                # Change the label to other cluster randomly
                labels = torch.randint(0, self.k, (1,))
            cluster_counts[labels.item()] += 1
            labels_list.append(int(labels.item()))
        return torch.tensor(labels_list).cuda()

    def compute_centroids(self, X):
        centroids = []
        c = torch.tensor([1.]).cuda()
        for i in range(self.centroids.shape[0]):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                # Step 1: Log map to Euclidean space
                euclidean_cluster_points = self.manifold.logmap0(cluster_points, c)
                # Step 2: Compute the mean in Euclidean space
                euclidean_mean = torch.mean(euclidean_cluster_points, axis=0).unsqueeze(0)
                # Step 3: Exponential map back to the manifold
                centroids.append(self.manifold.expmap0(euclidean_mean, c))
            else:
                # Reinitialize centroid to a random point
                centroids.append(X[torch.randint(0, X.shape[0], (1,))])
        return torch.stack(centroids).squeeze(1).cuda()

    def k_means(self, X):
        # Initialize two centroids randomly from the data to ensure centroids are within the manifold
        torch.manual_seed(X.shape[0])
        random_indices = torch.randperm(X.shape[0])[:self.k]
        self.centroids = X[random_indices]

        for i in range(100):
            # Assign labels based on closest centroid
            self.labels = self.assign_labels(X)
            # Compute new centroids
            new_centroids = self.compute_centroids(X)
            # Check for convergence (if centroids do not change)
            if torch.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        return self.centroids, self.labels

    def hierarchical_clustering(self, X, layer):
        """
        Perform recursive hierarchical clustering using k-means on the hyperboloid manifold.

        Parameters:
        - X: Input data points at the current recursion level.
        - layer: Current layer of the hierarchical clustering.

        Returns:
        - ClusterNode: The root node of the hierarchical clustering tree.
        """
        # Create the current cluster node
        node = ClusterNode(data=X, layer=layer)

        # If the layer exceed the maximum layer, stop splitting
        if layer > self.max_layer:
            return node
        else:
            # Apply k-means to split the data into two balanced clusters
            centroids, labels = self.k_means(X)
            if layer == self.max_layer:
                node.centroid = centroids[0]
                return node
            else:
                # Create child nodes for each cluster
                for cluster_id in range(centroids.shape[0]):
                    # Extract the data points belonging to the current cluster
                    cluster_data = X[labels == cluster_id]
                    child_node = self.hierarchical_clustering(cluster_data, layer + 1)
                    child_node.parent = node
                    child_node.centroid = centroids[cluster_id]
                    node.children.append(child_node)
        return node

    def get_nodes_at_layer(self, target_layer):
        nodes_at_layer = []
        # Check if the current node is at the target layer
        if self.layer == target_layer:
            nodes_at_layer.append(self)

        # Recursively check the children nodes
        for child in self.children:
            nodes_at_layer.extend(child.get_nodes_at_layer(target_layer))
        return nodes_at_layer

    def print_tree(self):
        for layer in range(self.max_layer+1):
            if layer == 0:
                num_data = len(set(self.tree.data))
                print(f"Layer {layer}, Nodes in each cluster: {num_data}")
            else:
                num_data = len(set(self.tree.get_nodes_at_layer(layer)[0].data))
                same_layer_nodes = self.tree.get_nodes_at_layer(layer)
                num_centroids = len(same_layer_nodes)
                mean_norm = np.mean([torch.mean(arcosh(node.centroid[0])).item() for node in same_layer_nodes])
                print(f"Layer {layer}, Nodes in each cluster: {num_data}, Centroids: {num_centroids}, Centroid mean norm: {mean_norm:.4f}")