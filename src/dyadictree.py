import numpy as np
from typing import List, Tuple, Union
from scipy.linalg import qr
import os
import logging
from .helpers import node_function, rand_pca
from .utils import *
from .dyadictreenode import DyadicTreeNode

class DyadicTree:
    def __init__(self, cover_tree, X=None, manifold_dims=None, max_dim=None, 
                 thresholds=0.5, precisions=1e-2, inverse=False):
        if cover_tree.n == 0:
            raise ValueError("Cover tree is empty")
        self.root = DyadicTreeNode(DyadicTreeNode.get_idx_sublevel(cover_tree.root), parent=None)
        self.height = 1

        # underlying cover tree
        self.cover_tree = cover_tree

        # Wavelet-related attributes
        self.j_k_to_node = {}
        self.inverse = inverse
        
        # Scikit-learn style attributes
        self._is_fitted = False
        self._X_shape = None
        
        self._setup_wavelet_params(manifold_dims, max_dim, thresholds, precisions)
            
        self.build_tree(self.root, cover_tree.root)
        
        if X is not None and manifold_dims is not None and max_dim is not None:
            self.fit(X)

    def _setup_wavelet_params(self, manifold_dims, max_dim, thresholds, precisions):
        if manifold_dims is not None:
            if not isinstance(manifold_dims, np.ndarray):
                self.manifold_dims = np.ones(self.height, dtype=int) * int(manifold_dims)
            else:
                self.manifold_dims = manifold_dims
        if thresholds is not None:
            if not isinstance(thresholds, np.ndarray):
                self.thresholds = np.ones(self.height, dtype=float) * thresholds
            else:
                self.thresholds = thresholds
        if precisions is not None:
            if not isinstance(precisions, np.ndarray):
                self.precisions = np.ones(self.height, dtype=float) * precisions
            else:
                self.precisions = precisions
        self.max_dim = max_dim

    def build_tree(self, node, cover_node, level=1):
        """
        Recursively build the DyadicTree from the CoverTree.
        """
        logging.debug(f"Building tree at level {level}, node indices: {DyadicTreeNode.get_idx_sublevel(cover_node)}")

        if level >= self.height:
            self.height = level+1
            logging.debug(f"Updated tree height to {self.height}")

        if hasattr(cover_node, 'idx'):
            # This cover_node is a leaf node - the current dyadic node should be the leaf
            logging.debug(f"Cover node is a leaf with indices: {cover_node.idx}")
            
            # Update the current node's indices to match the leaf indices
            node.idxs = cover_node.idx
            
            logging.debug(f"Created leaf node at level {level} with indices: {cover_node.idx}")
        else:
            logging.debug(f"Processing internal node at level {level} with {len(cover_node.children)} children")
            for i, child in enumerate(cover_node.children):
                child_node = DyadicTreeNode(DyadicTreeNode.get_idx_sublevel(child), parent=node)
                logging.debug(f"Created child {i+1}/{len(cover_node.children)} at level {level}")
                node.add_child(child_node)
                self.build_tree(child_node, child, level + 1)

    def traverse(self, only_print_level=None):
        """
        Traverse the tree and return nodes, optionally printing them.
        
        Parameters
        ----------
        only_print_level : int, optional
            If specified, only include nodes at this level
            
        Returns
        -------
        list of DyadicTreeNode
            List of nodes in the tree (filtered by level if specified)
        """
        nodes_list = []
        
        def traverse_from_node(node, level=0, only_print_level=None):
            """
            Traverse the tree, starting from the root node.
            At every level, use a single dash as level indicator.
            eg: root node has '-', child node has '--', etc.
            If only_print_level is set, only include nodes at that level.
            """
            if only_print_level is None or level == only_print_level:
                nodes_list.append(node)
                print("-" * (level + 1), node.idxs)
            
            for child in node.children:
                traverse_from_node(child, level + 1, only_print_level)

        traverse_from_node(self.root, level=0, only_print_level=only_print_level)
        return nodes_list
    
    def get_nodes_at_level(self, level):
        """
        Get all nodes at a specific level in the tree without printing.
        
        Parameters
        ----------
        level : int
            The level to get nodes from (0 = root level)
            
        Returns
        -------
        list of DyadicTreeNode
            List of nodes at the specified level
        """
        nodes_list = []
        
        def traverse_silent(node, current_level=0):
            if current_level == level:
                nodes_list.append(node)
            for child in node.children:
                traverse_silent(child, current_level + 1)
        
        traverse_silent(self.root)
        return nodes_list
    
    def get_all_nodes(self):
        """
        Get all nodes in the tree without printing.
        
        Returns
        -------
        list of DyadicTreeNode
            List of all nodes in the tree
        """
        nodes_list = []
        
        def traverse_silent(node):
            nodes_list.append(node)
            for child in node.children:
                traverse_silent(child)
        
        traverse_silent(self.root)
        return nodes_list
    
    def plot_tree(self, show_basis_dim=False, start_node=None):
        '''
        plot the tree using matplotlib
        start_node will be considered as the root of the tree to plot, root is used if None.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        def plot_node(node, x, y, ax, level=0):
            """
            Plot the node and its children.
            """
            color = 'red' if getattr(node, 'fake_node', False) else 'black'
            
            # Create the text to display
            if show_basis_dim:
                text_parts = [str(node.idxs)]
                
                # Add basis dimension if available
                if hasattr(node, 'basis') and node.basis is not None:
                    basis_dim = node.basis.shape[0]
                    text_parts.append(f"dim={basis_dim}")
                
                # Add wavelet basis dimension if available
                if hasattr(node, 'wav_basis') and node.wav_basis is not None:
                    wav_basis_dim = node.wav_basis.shape[0]
                    text_parts.append(f"wav={wav_basis_dim}")
                
                text = "\n".join(text_parts)
            else:
                text = str(node.idxs)
            
            ax.text(x, y, text, ha='center', va='center', fontsize=8, color=color)
            
            # Also plot (node_j, node_k) in red if the node has these attributes
            if hasattr(node, 'node_j') and hasattr(node, 'node_k'):
                jk_text = f"({node.node_j}, {node.node_k})"
                ax.text(x, y - 0.02, jk_text, ha='center', va='center', fontsize=6, color='red')
            for i, child in enumerate(node.children):
                child_x = x + (i - len(node.children) / 2) * 0.1
                child_y = y - 0.1
                ax.plot([x, child_x], [y, child_y], 'k-')
                plot_node(child, child_x, child_y, ax, level + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Structure')
        ax.axis('off')
        if start_node is None:
            plot_node(self.root, 0, 0, ax)
        else:
            plot_node(start_node, 0, 0, ax)
        plt.show()

    def plot_tree_graphviz(self, show_basis_dim=False, start_node=None, filename=None, format='png'):
        '''
        Plot the tree using graphviz
        start_node will be considered as the root of the tree to plot, root is used if None.
        
        Parameters
        ----------
        show_basis_dim : bool, default=False
            Whether to show basis dimensions in the node labels
        start_node : DyadicTreeNode, optional
            The node to use as root for plotting. If None, uses self.root
        filename : str, optional
            If provided, saves the plot to this file. Otherwise displays it
        format : str, default='png'
            Output format for the plot ('png', 'svg', 'pdf', etc.)
        '''
        try:
            import graphviz
        except ImportError:
            raise ImportError("graphviz is required for plot_tree_graphviz. Install it with: pip install graphviz")
        
        # Create a new directed graph
        dot = graphviz.Digraph(comment='DyadicTree Structure')
        dot.attr(rankdir='TB')  # Top to bottom layout
        
        # Counter for unique node IDs
        node_counter = [0]
        
        def add_node_to_graph(node, parent_id=None):
            """
            Add a node and its children to the graphviz graph.
            """
            # Create unique node ID
            node_id = f"node_{node_counter[0]}"
            node_counter[0] += 1
            
            # Create node label
            if show_basis_dim:
                label_parts = [str(node.idxs)]
                
                # Add basis dimension if available
                if hasattr(node, 'basis') and node.basis is not None:
                    basis_dim = node.basis.shape[0]
                    label_parts.append(f"dim={basis_dim}")
                
                # Add wavelet basis dimension if available
                if hasattr(node, 'wav_basis') and node.wav_basis is not None:
                    wav_basis_dim = node.wav_basis.shape[0]
                    label_parts.append(f"wav={wav_basis_dim}")
                
                label = "\\n".join(label_parts)
            else:
                label = str(node.idxs)
            
            # Add (node_j, node_k) if available
            if hasattr(node, 'node_j') and hasattr(node, 'node_k'):
                label += f"\\n({node.node_j}, {node.node_k})"
            
            # Set node color based on whether it's a fake node
            color = 'red' if getattr(node, 'fake_node', False) else 'black'
            fillcolor = 'lightcoral' if getattr(node, 'fake_node', False) else 'lightblue'
            
            # Add node to graph
            dot.node(node_id, label=label, 
                    color=color, 
                    fillcolor=fillcolor,
                    style='filled',
                    shape='box')
            
            # Add edge from parent if this is not the root
            if parent_id is not None:
                dot.edge(parent_id, node_id)
            
            # Recursively add children
            for child in node.children:
                add_node_to_graph(child, node_id)
        
        # Start from the specified node or root
        root_node = start_node if start_node is not None else self.root
        add_node_to_graph(root_node)
        
        # Handle output
        if filename is not None:
            # Save to file
            dot.render(filename, format=format, cleanup=True)
            print(f"Tree plot saved to {filename}.{format}")
        else:
            # Display in notebook or return source
            try:
                # Try to display in Jupyter notebook
                from IPython.display import display
                display(dot)
            except ImportError:
                # If not in notebook, print the source or save to temp file
                print("Graphviz source:")
                print(dot.source)
                print("\nTo view the graph, save it to a file using the filename parameter")
        
        return dot

    def grow_tree(self):
        """
        Grow tree so that every leave have the same levels = tree height.
        if a leave is not enough height, we duplicate node until it reaches the height of the tree.
        set the node.fake_node = True
        """
        
        def grow_node(node, current_level):
            if current_level >= self.height:
                return

            if len(node.children) == 0:
                # duplicate node
                new_node = DyadicTreeNode(node.idxs, parent=node)
                new_node.fake_node = True

                # register and grow
                node.add_child(new_node)
                grow_node(new_node, current_level + 1)
            else:
                for child in node.children:
                    grow_node(child, current_level + 1)

        grow_node(self.root, 0)

        return self
    
    def query_leaf(self, X):
        """
        Query for the leaf that is closest to the data point using self.cover_tree.
        Since we're using brute force query now, this method is deprecated.
        Use query_leaf_by_center instead.
        """
        raise NotImplementedError("query_leaf is deprecated. Use query_leaf_by_center instead.")

    def query_leaf_by_center(self, X):
        """
        Query for the leaf that is closest to each data point by comparing distances
        to leaf centers directly using vectorized operations.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Data points to query
            
        Returns
        -------
        leaf_nodes : list of DyadicTreeNode
            List of leaf nodes closest to each data point
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get all leaf nodes
        leafs = self.get_all_leafs()
        
        if not leafs:
            raise ValueError("No leaf nodes found in the tree")
        
        # Extract centers from leaf nodes - shape (n_leafs, n_features)
        leaf_centers = []
        for leaf in leafs:
            if hasattr(leaf, 'center') and leaf.center is not None:
                leaf_centers.append(leaf.center.flatten())
            else:
                # If no center is available, use mean of data points at this leaf
                # This requires accessing the original data, which might not be available
                # For now, we'll raise an error
                raise ValueError(f"Leaf node {leaf.idxs} does not have a center attribute")
        
        leaf_centers = np.array(leaf_centers)  # Shape: (n_leafs, n_features)
        
        # Method 1: Using scipy.spatial.distance.cdist (most efficient for large datasets)
        try:
            from scipy.spatial.distance import cdist
            # distances shape: (n_samples, n_leafs)
            distances = cdist(X, leaf_centers, metric='euclidean')
        except ImportError:
            # Method 2: Using numpy broadcasting (fallback if scipy not available)
            # X shape: (n_samples, n_features)
            # leaf_centers shape: (n_leafs, n_features)
            # distances shape: (n_samples, n_leafs)
            distances = np.linalg.norm(X[:, np.newaxis, :] - leaf_centers[np.newaxis, :, :], axis=2)
        
        # Find the index of closest leaf for each data point
        closest_leaf_indices = np.argmin(distances, axis=1)
        
        # Get the corresponding leaf nodes
        leaf_nodes = [leafs[idx] for idx in closest_leaf_indices]
        
        return leaf_nodes

    def make_basis(self, X: np.ndarray) -> None:
        '''
        Recursively compute basis for all nodes in the tree
        '''
        logging.info("Starting basis construction for DyadicTree")
        self.j_k_to_node = {}
        
        # Setup root node
        self.root.is_leaf = len(self.root.children) == 0
        self.root.node_j = 0
        self.root.node_k = 0
        
        # Compute basis for root using node's make_basis method
        self.root.make_basis(X, self.manifold_dims[0], self.max_dim,
                            self.thresholds[0], self.precisions[0], self.inverse)
        
        self.j_k_to_node[(0, 0)] = self.root
        
        # Recursively process all levels
        self._make_basis_recursive(self.root, X, level=0)

    def _make_basis_recursive(self, node: DyadicTreeNode, X: np.ndarray, level: int) -> None:
        '''
        Recursive helper for make_basis
        '''
        if level + 1 >= self.height:
            return
            
        logging.debug(f"Processing level {level+1} with {len(node.children)} children")
        
        k_counter = 0
        if level == 0:  # Start counting from level 1
            k_counter = 0
        else:
            # Find the current k_counter for this level
            existing_nodes_at_level = [n for (j, k), n in self.j_k_to_node.items() if j == level + 1]
            k_counter = len(existing_nodes_at_level)
        
        for child in node.children:
            child.is_leaf = len(child.children) == 0
            child.node_j = level + 1
            child.node_k = k_counter
            
            # Compute basis for child using node's make_basis method
            child.make_basis(X, 
                           self.manifold_dims[level + 1] if level + 1 < len(self.manifold_dims) else self.manifold_dims[-1],
                           self.max_dim,
                           self.thresholds[level + 1] if level + 1 < len(self.thresholds) else self.thresholds[-1],
                           self.precisions[level + 1] if level + 1 < len(self.precisions) else self.precisions[-1],
                           self.inverse)

            self.j_k_to_node[(level + 1, k_counter)] = child
            k_counter += 1
            
            # Recursively process children
            self._make_basis_recursive(child, X, level + 1)

    def make_wavelets(self, X: np.ndarray) -> None:
        logging.debug("Starting wavelet construction")
        
        nodes_at_layers = [[self.root]]
        current_layer = nodes_at_layers[0]
        for level in range(1, self.height):
            next_layer = list()
            for node in current_layer:
                for child in node.children:
                    next_layer.append(child)
            nodes_at_layers.append(next_layer)
            current_layer = next_layer
            logging.debug(f"Layer {level} has {len(next_layer)} nodes")

        for j in range(self.height-1, -1, -1):
            nodes = nodes_at_layers[j]
            logging.debug(f"Processing wavelets for level {j} with {len(nodes)} nodes")
            for i, node in enumerate(nodes):
                logging.debug(f"Making transform for node {i+1}/{len(nodes)} at level {j}")
                node.make_transform(X, 
                                  self.manifold_dims[j] if j < len(self.manifold_dims) else self.manifold_dims[-1], 
                                  self.max_dim,
                                  self.thresholds[j] if j < len(self.thresholds) else self.thresholds[-1], 
                                  self.precisions[j] if j < len(self.precisions) else self.precisions[-1])
        
        # For convenience, set Ψ0,k := Φ0,k and w0,k := c0,k for k ∈K0.
        # logging.debug("Setting Ψ0,k and w0,k for convenience")
        # for i,c in enumerate(self.root.children):
        #     c.wav_basis = self.root.basis
        #     c.wav_consts = self.root.center.T

    
    def get_all_leafs(self) -> List[DyadicTreeNode]:
        """
        Get all leaf nodes in the tree by traversing from the root.
        """
        logging.debug("Retrieving all leaf nodes from the DyadicTree")
        
        def traverse(node):
            if len(node.children) == 0:
                return [node]
            else:
                leafs = []
                for child in node.children:
                    leafs.extend(traverse(child))
                return leafs
        
        all_leafs = traverse(self.root)
        logging.debug(f"Found {len(all_leafs)} leaf nodes")
        return all_leafs
    
    # def fgwt_all_node(self, X):
    #     '''
    #     Compute the forward gmra wavelet transform for all nodes in the tree.
    #     fgwt is only depends on computation from the leaf has x, traversing the tree to the root.
    #     fgwt_all_node will compute basically the same thing, but for node that not in the path, it's 0
    #     '''
    #     logging.debug("Starting forward GMRA wavelet transform for all nodes")

    #     # get all the leafs in the tree by traverse
    #     leafs = self.get_all_leafs()

    def fgwt_batch(self, X):
        """
        Compute the forward gmra wavelet transform for all nodes in the tree.
        Each data point will have coefficients for all nodes in the tree.
        """
        out = []

        leafs = self.get_all_leafs()

        # log the leafs j&k
        logging.debug(f"Found {len(leafs)} leaf nodes for batch processing")
        logging.debug(str([f"(j={leaf.node_j}, k={leaf.node_k})" for leaf in leafs]))
        for leaf in leafs:
            # batch processing
            pjx = (X - leaf.center.T) @ leaf.basis.T   # ~> (B, d)
            qjx = pjx @ leaf.basis @ leaf.wav_basis.T #  # ~> (B, d)
            out.append(qjx)
            logging.debug(f"Processed leaf (j={leaf.node_j}, k={leaf.node_k})")
            pJx = pjx # ~> (B, d)
            p = path(leaf)
            for n in reversed(p[1:-1]):
                pjx = pJx @ leaf.basis @ n.basis.T + \
                    (leaf.center.T - n.center.T) @ n.basis.T # ~> (B, d)
                qjx = pjx @ n.basis @ n.wav_basis.T # ~> (B, d)
                out.append(qjx)
                logging.debug(f"Processed node (j={n.node_j}, k={n.node_k}), pjx shape: {pjx.shape}, qjx shape: {qjx.shape}")
            n = p[0] 
            pjx = (leaf.center.T - n.center.T) @ n.basis.T + \
                pJx @ leaf.basis @ n.basis.T # ~> (B, d)
            qjx = pjx # ~> (B, d)
            out.append(qjx)
        
        # remove tensor with 0 dimensions
        out = [q for q in out if q.shape[1] > 0]
        # stacking  (B, d1) (B d2) -> (B, d1+d2+...)
        out = np.hstack(out)
        return out

    def fgwt(self, X):
        '''
        Compute the forward gmra wavelet transform
        '''
        logging.debug(f"Starting forward GMRA wavelet transform for {X.shape[0]} data points")
        
        # Use brute force query by center
        leafs = self.query_leaf_by_center(X)

        leafs_jk = [(leaf.node_j, leaf.node_k) for leaf in leafs]
        
        logging.debug(f"Found {len(leafs)} leaf nodes, levels range: j={min(jk[0] for jk in leafs_jk)} to j={max(jk[0] for jk in leafs_jk)}")
        
        Qjx = [None] * X.shape[0]

        for idx, leaf in enumerate(leafs):
            if idx % 20 == 0:  # Log every 20th point to avoid spam
                logging.debug(f"Processing point {idx+1}/{len(leafs)}, leaf at (j={leaf.node_j}, k={leaf.node_k})")
            
            x = X[idx].reshape(1, -1)  #  a row
            if leaf.parent is None:
                Qjx[idx] = [leaf.basis @ (x.T - leaf.center)]
                continue
            pjx = leaf.basis @ (x.T - leaf.center) 
            qjx = leaf.wav_basis @ leaf.basis.T @ pjx
            # log qjx
            logging.debug(f"qjx: {qjx}")

            Qjx[idx]=[qjx]
            pJx = pjx
            
            p = path(leaf)
            logging.debug(f"Point {idx}: path length {len(p)}, leaf->root traversal")
            logging.debug(f"Point {idx}: starting from leaf (j={leaf.node_j}, k={leaf.node_k}), pjx shape: {pjx.shape}, qjx shape: {qjx.shape}")

            for n in reversed(p[1:-1]):
                pjx = n.basis @ leaf.basis.T @ pJx + \
                        n.basis @ ( leaf.center - n.center ) 
                qjx = n.wav_basis @ n.basis.T @ pjx
                logging.debug(f"qjx: {qjx}")
                Qjx[idx].append(qjx)
                logging.debug(f"Point {idx}: processed node at (j={n.node_j}, k={n.node_k}), qjx shape: {qjx.shape}")

            n = p[0]
            pjx = n.basis @ leaf.basis.T @ pJx + n.basis @ ( leaf.center - n.center ) 
            qjx = pjx
            logging.debug(f"qjx: {qjx}")
            Qjx[idx].append(qjx)
            Qjx[idx] = list(reversed(Qjx[idx]))
            
            logging.debug(f"Point {idx}: completed, total coefficients at {len(Qjx[idx])} levels")
            logging.debug(f"***")
        
        logging.debug("Forward GMRA wavelet transform completed")
        return Qjx, leafs_jk

    def igwt_batch(self, Y, leaves_j_k, shape):
        """
        Compute the inverse gmra wavelet transform for all nodes in the tree.
        Each data point will have coefficients for all nodes in the tree.

        we determine the coefficent for each node by checking the basis's dimension
        """

        leafs =  self.get_all_leafs()

        # log the leafs j&k
        logging.debug(f"Found {len(leafs)} leaf nodes for batch processing")
        logging.debug(str([f"(j={leaf.node_j}, k={leaf.node_k})" for leaf in leafs]))
 
        coeff_dim = 0
        X_hat = 0

        X_hat_each_leaf = []
        for leaf in leafs:
            logging.debug(f"Processing leaf (j={leaf.node_j}, k={leaf.node_k})")
            coeff = Y[:, coeff_dim:coeff_dim + leaf.wav_basis.shape[0]]
            coeff_dim += leaf.wav_basis.shape[0]
            logging.debug(f"Multiplying jk (j={leaf.node_j}, k={leaf.node_k}),\
                with coeff from dimension {coeff_dim - leaf.wav_basis.shape[0]} to {coeff_dim}")

            Qjx = coeff @ leaf.wav_basis + leaf.wav_consts.T
            leaf = leaf.parent
            while leaf.parent is not None:
                coeff = Y[:, coeff_dim:coeff_dim + leaf.wav_basis.shape[0]]
                coeff_dim += leaf.wav_basis.shape[0]
                Qjx += (coeff @ leaf.wav_basis + leaf.wav_consts.T -
                        Qjx @ leaf.parent.basis.T @ leaf.parent.basis)
                logging.debug(f"Multiplying node (j={leaf.node_j}, k={leaf.node_k}),\
                    with coeff from dimension {coeff_dim - leaf.wav_basis.shape[0]} to {coeff_dim}")
                leaf = leaf.parent

            coeff = Y[:, coeff_dim:coeff_dim + leaf.basis.shape[0]]
            coeff_dim += leaf.basis.shape[0]
            logging.debug(f"Multiplying root (j={leaf.node_j}, k={leaf.node_k}),\
                with coeff from dimension {coeff_dim - leaf.basis.shape[0]} to {coeff_dim}")
            Qjx = coeff @ leaf.basis + leaf.center.T + Qjx
            X_hat += Qjx
            X_hat_each_leaf.append(Qjx)
        
        return X_hat / len(leafs), X_hat_each_leaf
 
    def igwt(self, gmra_q_coeff, leaves_j_k, shape):
        '''
        Compute the inverse gmra wavelet transform
        '''
        logging.debug(f"Starting inverse GMRA wavelet transform for {len(gmra_q_coeff)} data points")
        logging.debug(f"Reconstruction target shape: {shape}")
        
        X_recon = np.zeros(shape, dtype=np.float64)

        # iterate over data points
        for i in range(len(gmra_q_coeff)):
            if i % 20 == 0:  # Log every 20th point to avoid spam
                logging.debug(f"Reconstructing point {i+1}/{len(gmra_q_coeff)}")
            
            # coefficient and leaf node for this data
            coeffs = list(reversed(gmra_q_coeff[i]))# leaf -> root
            leaf = self.j_k_to_node[leaves_j_k[i]]
            if leaf.parent is None:
                X_recon[i] = (leaf.basis.T @ coeffs[0] + leaf.center).ravel()
                continue
            lvl_from_leaf = 0
            
            logging.debug(f"Point {i}: starting from leaf (j={leaf.node_j}, k={leaf.node_k}), {len(coeffs)} coefficient levels")

            # begin reconstruct
            # leaves level treat differently
            Qjx = leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts
            logging.debug(f"Point {i}: leaf reconstruction, Qjx shape: {Qjx.shape}")
            
            leaf = leaf.parent
            lvl_from_leaf += 1

            while leaf.parent is not None:
                Qjx += (leaf.wav_basis.T @ coeffs[lvl_from_leaf] + leaf.wav_consts -
                        leaf.parent.basis.T @ leaf.parent.basis @ Qjx)
                logging.debug(f"Point {i}: intermediate level (j={leaf.node_j}, k={leaf.node_k}), Qjx shape: {Qjx.shape}")
                leaf = leaf.parent
                lvl_from_leaf += 1
            
            # root level also treat differently
            Qjx += leaf.basis.T @ coeffs[lvl_from_leaf] + leaf.center
            logging.debug(f"Point {i}: root level reconstruction, final Qjx shape: {Qjx.shape}")

            X_recon[i:i+1,:] = Qjx.T
            
            if i % 20 == 0:
                recon_norm = np.linalg.norm(X_recon[i])
                logging.debug(f"Point {i}: reconstruction norm: {recon_norm:.6f}")
            logging.debug(f"***")
        
        logging.debug("Inverse GMRA wavelet transform completed")
        return X_recon

    def prune_tree_min_point(self, num_point):
        """
        Prune the tree by removing children that have fewer points than num_point.
        
        Parameters
        ----------
        num_point : int
            Minimum number of points required for a node to remain in the tree.
            Children with fewer points will be removed.
            
        Returns
        -------
        self : DyadicTree
            Returns the instance itself after pruning.
        """
        def _prune_node(node):
            """Recursively prune children from a node."""
            # Create a copy of children list to avoid modification during iteration
            children_to_remove = []
            
            for child in node.children:
                if len(child.idxs) < num_point:
                    children_to_remove.append(child)
                else:
                    # Child has enough points, continue recursing
                    _prune_node(child)
            
            # Remove children that don't meet the threshold
            for child in children_to_remove:
                logging.debug(f"Pruning child node with indices {child.idxs} from parent node with indices {node.idxs}")
                
                # Remove child from parent's children list
                node.children.remove(child)
                child.parent = None
        
        # Start pruning from the root
        _prune_node(self.root)
        
        # Update tree height after pruning
        self._update_tree_height()
        
        return self
    
    def prune_max_level(self, max_level):
        """
        Prune the tree by removing nodes that exceed the maximum level.
        
        Parameters
        ----------
        max_level : int
            Maximum level allowed in the tree. Level 0 is the root.
            Nodes at levels greater than max_level will be removed.
            
        Returns
        -------
        self : DyadicTree
            Returns the instance itself after pruning.
        """
        def _prune_node_by_level(node, current_level):
            """Recursively prune children that exceed max_level."""
            if current_level >= max_level:
                # Remove all children at this level or beyond
                for child in node.children:
                    child.parent = None
                    logging.debug(f"Pruning child node at level {current_level + 1} with indices {child.idxs}")
                node.children.clear()
            else:
                # Continue recursing for children that don't exceed max_level
                for child in node.children:
                    _prune_node_by_level(child, current_level + 1)
        
        # Start pruning from the root (level 0)
        _prune_node_by_level(self.root, 0)
        
        # Update tree height after pruning
        self._update_tree_height()
        
        return self
    
    def _update_tree_height(self):
        """Update the tree height after pruning."""
        def _get_max_depth(node, current_depth=0):
            if len(node.children) == 0:
                return current_depth
            max_child_depth = 0
            for child in node.children:
                child_depth = _get_max_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            return max_child_depth
        
        self.height = _get_max_depth(self.root) + 1
    
    
    # ========== Scikit-learn Style API ==========
    
    def fit(self, X):
        """
        Learn the basis and wavelet basis from the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        if X.ndim != 2 or not min(X.shape) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite 2D array")
        if X.shape != self.cover_tree.data.shape or not np.array_equal(X, self.cover_tree.data):
            raise ValueError("Fit data must match the cover-tree data and row ordering; build a new cover tree for new data")
        self._X_shape = X.shape
        
        # Learn basis and wavelets
        self.make_basis(X)
        self.make_wavelets(X)

        # # compute the dimension of the transformed data
        # self._return_shape = self._compute_return_shape(X.shape[0])
        
        self._is_fitted = True
        return self
    
    def _compute_return_shape(self):
        """
        Compute the shape of the transformed data based on the wavelet coefficients.
        This would be the sum of the dimensions of the wavelet basis for all nodes.
        
        Returns
        -------
        return_shape : tuple
            Shape of the transformed data.
        """
        # Assuming each leaf node contributes a fixed number of coefficients

        return
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the forward GMRA wavelet transform to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) where:
            - coefficients: list of wavelet coefficients for each data point
            - leaf_indices: list of (j, k) tuples indicating leaf node for each point
        """
        return self.fit(X).transform(X)
    
    def transform(self, X):
        """
        Apply the forward GMRA wavelet transform to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) where:
            - coefficients: list of wavelet coefficients for each data point
            - leaf_indices: list of (j, k) tuples indicating leaf node for each point
        """
        if not self._is_fitted:
            raise ValueError("This DyadicTree instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        X = np.asarray(X)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array")
        
        if X.shape[1] != self._X_shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, but DyadicTree is expecting "
                           f"{self._X_shape[1]} features as seen in fit.")
        
        if len(X) == 0:
            return [], []
        return self.fgwt(X)
    
    def inverse_transform(self, X_transformed):
        """
        Apply the inverse GMRA wavelet transform to reconstruct the original data.
        
        Parameters
        ----------
        X_transformed : tuple
            Tuple containing (coefficients, leaf_indices) as returned by transform().
            
        Returns
        -------
        X_reconstructed : array of shape (n_samples, n_features)
            Reconstructed data.
        """
        if not self._is_fitted:
            raise ValueError("This DyadicTree instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this estimator.")
        
        if not isinstance(X_transformed, tuple) or len(X_transformed) != 2:
            raise ValueError("X_transformed must be a tuple of (coefficients, leaf_indices) "
                           "as returned by transform().")
        
        coefficients, leaf_indices = X_transformed
        
        if len(coefficients) != len(leaf_indices):
            raise ValueError("Length of coefficients and leaf_indices must match.")
        
        # Determine the shape for reconstruction
        n_samples = len(coefficients)
        n_features = self._X_shape[1]
        shape = (n_samples, n_features)
        
        return self.igwt(coefficients, leaf_indices, shape)
    
    # ========== Additional utility methods ==========
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'cover_tree': self.cover_tree,
            'manifold_dims': getattr(self, 'manifold_dims', None),
            'max_dim': self.max_dim,
            'thresholds': getattr(self, 'thresholds', None),
            'precisions': getattr(self, 'precisions', None),
            'inverse': self.inverse
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

    # ========== Original methods ==========
