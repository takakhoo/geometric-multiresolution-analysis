import numpy as np
from typing import List, Tuple, Union
import logging
from .helpers import node_function, rand_pca

class DyadicTreeNode:
    @staticmethod
    def get_idx_sublevel(node):
        """
        Get all the index of the nodes under node of a CoverTreeNode
        :param node: node
        :return: numpy 1d array of indices
        """
        if hasattr(node, 'idx'):
            return node.idx
        else:
            idxs = [DyadicTreeNode.get_idx_sublevel(child) for child in node.children]
            if idxs:
                return np.concatenate(idxs)
            else:
                return np.array([], dtype=int)

    def __init__(self, idxs, parent):
        self.idxs = idxs
        self.children = []
        self.parent = parent 
        self.fake_node = False 

        self.node_k = None  # this will be used to store the node in the CoverTree
        self.node_j = None  # this will be used to store the node in the CoverTree
        
        # Wavelet-related attributes
        self.is_leaf = False
        self.center = None
        self.size = None
        self.radius = None
        self.basis = None
        self.sigmas = None
        self.Z = None
        self.wav_basis = None
        self.wav_sigmas = None
        self.wav_consts = None
        self.CelWavCoeffs = {}

    def add_child(self, child_node):
        self.children.append(child_node)

    def __getitem__(self, index):
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
    def make_basis(self, X: np.ndarray, manifold_dim: int, max_dim: int, 
                   threshold: float = 0.5, precision: float = 1e-2, 
                   inverse: bool = False) -> None:
        '''
        Compute basis for this node using node_function
        '''
        logging.debug(f"Computing basis for node (j={self.node_j}, k={self.node_k}) with {len(self.idxs)} points, is_leaf={self.is_leaf}")
        
        self.center, self.size, self.radius, self.basis, self.sigmas, self.Z = node_function(
            np.atleast_2d(X[self.idxs,:]),
            manifold_dim,
            max_dim,
            self.is_leaf,
            threshold=threshold,
            precision=precision, 
            inverse=inverse
        )
        
        logging.debug(f"Node (j={self.node_j}, k={self.node_k}) basis shape: {self.basis.shape}, sigmas: {len(self.sigmas)}")
    
    def make_transform(self,
                       X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
                       threshold: float = 0.5,
                       precision: float = 1e-2) -> None:
        '''
        X: (n,d) - rows are data points, columns are features
        '''
        # from construct_localGeometricWavelets.m
        Phijx = self.basis

        for i, c in enumerate(self.children):
            if np.prod(c.basis.shape) >= 1:
                # this is (I-Pjx)V_{j+1,x}
                Phij1x = c.basis  # phi{j+1,x}
                
                Y: np.ndarray = Phij1x - Phij1x @ Phijx.T @ Phijx
            
                U,s,_ = rand_pca(Y.T, min(min(X.shape), max_dim))
                wav_dims = int((s > threshold).sum())

                if wav_dims > 0:
                    c.wav_basis = U[:,:wav_dims].T # (nxd)
                    c.wav_sigmas = s[:wav_dims]
                else:
                    c.wav_basis = np.zeros_like(U[:,:0].T)  # (nxd)
                    c.wav_sigmas = np.zeros_like(s[:0])

                tjx = c.center - self.center# (2.14)
                c.wav_consts = tjx - Phijx.T @ Phijx @ tjx
            else:
                c.wav_basis  = np.zeros((X.shape[1], 0)).T # (nxd)
                # Even a rank-zero leaf has an affine center displacement.
                tjx = c.center - self.center
                c.wav_consts = tjx - Phijx.T @ Phijx @ tjx
