""" script containing class and functions for network loading and manipulation
"""
import numpy as np

import os
from os.path import dirname
import numpy as np
import networkx as nx
from ._graphml import load_graphml
from ._parcellation import Parcellation
# Paths
assetpath = os.path.join(dirname(dirname(dirname(os.path.abspath(__file__)))), 'pryonix/assets')

def connectome_path():
    return os.path.join(assetpath, 'Connectomes-hcp-scale1.xml')

# Connectome class
class Connectome:
    def __init__(self, parc, graph, n_matrix, l_matrix, weight_function):
        self.parc = parc
        self.graph = graph
        self.n_matrix = n_matrix
        self.l_matrix = l_matrix
        self.weight_function = weight_function

    @classmethod
    def from_graph_path(cls, graph_path: str, norm=True, weight_function=None):
        if weight_function is None:
            weight_function = lambda n, l: n

        _parc, n_matrix, l_matrix = load_graphml(graph_path)
        
        ids, labels, cortex, lobes, hemispheres, xs, ys, zs = _parc
        parc = Parcellation.from_lists(ids, labels, cortex, lobes, hemispheres, xs, ys, zs)
        sym_n = symmetrise(n_matrix)
        sym_l = symmetrise(l_matrix)
        weighted_graph = weight_function(sym_n, sym_l)
        A = np.nan_to_num(weighted_graph)

        if norm:
            A /= np.max(A)

        graph = nx.from_numpy_array(A)
        return cls(parc, graph, n_matrix, l_matrix, weight_function)

    def __str__(self):
        return f"Parcellation: {self.parc}\nAdjacency Matrix: {adjacency_matrix(self)}"

    def filter(self, cutoff=1e-2):
        filtered_matrix = filter_adjacency_matrix(self.graph, cutoff)
        return Connectome(self.parc, nx.from_numpy_array(filtered_matrix), self.n_matrix, self.l_matrix, self.weight_function)

    def slice(self, idx, norm=True):
        N = self.n_matrix[np.ix_(idx, idx)]
        L = self.l_matrix[np.ix_(idx, idx)]
        weighted_graph = self.weight_function(N, L)
        A = np.nan_to_num(weighted_graph)

        if norm:
            A /= np.max(A)

        graph = nx.from_numpy_array(A)
        parc = self.parc[idx.tolist()]
        return Connectome(parc, graph, N, L, self.weight_function)

    def reweight(self, norm=True, weight_function=None):
        if weight_function is None:
            weight_function = self.weight_function
        weighted_graph = weight_function(self.n_matrix, self.l_matrix)
        A = np.nan_to_num(weighted_graph)

        if norm:
            A /= np.max(A)

        graph = nx.from_numpy_array(A)
        return Connectome(self.parc, graph, self.n_matrix, self.l_matrix, weight_function)
        
    def __repr__(self):
        N = len(self.parc)
        return f"Connectome with {N} nodes."

def symmetrise(A):
    return (A + A.T) / 2

def filter_adjacency_matrix(graph, cutoff):
    adj_matrix = nx.to_numpy_array(graph)
    return adj_matrix * (adj_matrix > cutoff)

def adjacency_matrix(c: Connectome):
    return nx.to_numpy_array(c.graph)

def laplacian_matrix(c: Connectome):
    return nx.laplacian_matrix(c.graph).toarray()