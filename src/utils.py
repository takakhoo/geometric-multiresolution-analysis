from collections import Counter
import numpy as np
import math
from scipy.spatial.distance import pdist

#This file contains util helper functions that aid in traversing trees, gaining info from trees, and operations over trees

def check_wav_vars(root):
	if root.is_leaf:
		return {'wav_basis':0, 'wav_consts':0, 'wav_sigmas': 0}
	total = {'wav_basis':0, 'wav_consts':0, 'wav_sigmas': 0}
	for child in root.children:
		res = check_wav_vars(child)
		total['wav_basis'] += res['wav_basis']
		total['wav_sigmas'] += res['wav_sigmas']
		total['wav_consts'] += res['wav_consts']
	if not root.wav_basis is None:
		total['wav_basis'] +=1
	if not root.wav_sigmas is None:
		total['wav_sigmas'] +=1
	if not root.wav_consts is None:
		total['wav_consts'] +=1
	return total


# Given a tree structure root node, return a list of the leaf nodes in the tree
def get_leafs(node):
    if len(node.idxs) == 1:
        return [node]
    leafs = []
    for child in node.children:
        leafs += get_leafs(child)
    return leafs

# Given a tree structure leaf node, return a path of nodes (list) to the root
def path(node):
    if node.parent is None:
        return [node]
    return path(node.parent) + [node]

# Given a tree structure, return the top 15 best dimensions with their counts
def get_dim_dists(tree):
    leafs = get_leafs(tree.root)
    dims = [{node.basis.shape[1] for node in path(leaf)} for leaf in leafs]

    # print(dims)
    
    all_dims = []
    for dim in dims:
        all_dims += list(dim)

    cnt = Counter(all_dims)
    best_15 = sorted(cnt.items(),key=lambda x: x[1])[-15:]

    return best_15

    # print(best_15)

    # plt.hist(all_dims)
    # plt.show()

# Given an input data matrix X, a wavelettree, and the desired dim returns a matrix in the same order/shape as X that finds 
# embeddings of nodes at that dim and returns an embedding matrix. If the nodes are not found at that dim in the tree, entries will be 0
def get_embeddings_at_dim(X, tree, dim):
    embeddings = np.zeros((X.shape[0],dim))

    count = 0
    #TODO: refactor this to traverse tree rather than leafs, current way is inefficient
    for leaf in get_leafs(tree.root):
        pt_idx = leaf.idxs[0]
        #check if we've found this embedding yet
        if any(embeddings[pt_idx]):
            continue
        #if not, we need to search for the node
        for node in path(leaf):
            if node.basis.shape[1] == dim:
                #If this point in leaf can be represented by dim at this node, extract its embedding and all others at this node
                for idx in range(len(node.idxs)):
                    new_idx = node.idxs[idx]
                    embeddings[new_idx] = node.basis[idx]
                count += len(node.idxs)
                print(count)
                #we are done, don't need to explore the path more
                break
    print(f"{count} nodes found at dimension {dim}")
    return embeddings

#Given a root node from a tree, compute the depth of the tree
def depth(node):
    if node.is_leaf:
        return 1
    return 1 + max([depth(child) for child in node.children])

# depth has higher values for nodes closer to the root. 
# level will tell you what scale, or j you are at in the tree
# Input to level need not be the root
def level(node):
    if node.parent is None:
        return 1
    else:
        return 1 + level(node.parent)

#Return a list of all nodes at depth in tree
#subroutine to best_depth, start by passing in root node and depth you want
def get_nodes_at_depth(node, depth):
    #TODO: what happens when depth > max depth of node (return [])
    if depth == 0:
        return [node]
    result = []
    for child in node.children:
        result+=get_nodes_at_depth(child,depth-1)
    return result

#Find the list of WaveletNodes that exist at the deepest level where all nodes have the same dimension
def best_depth(tree):
    #TODO: What happens if node.basis is empty, does this work still?
    depth_counter = 1
    #root will satisfy best depth parameters since all nodes are present in root
    best_nodes = [tree]
    best_dim = tree.basis.shape[1]
    while True:
        nodes = get_nodes_at_depth(tree,depth_counter)

        #check if this set is "good" - all nodes have the same dimension
        dims = list({node.basis.shape[1] for node in nodes})
        # print("dims", dims)
        # print("depth counter", depth_counter)

        num_dims = len(dims)

        #if num_dims is 1, then all nodes have the same dimension (good). need to make sure its not 0
        if num_dims == 1 and not dims[0] == 0:
            best_nodes = nodes
            best_dim = dims[0]
        else:
            #if this is a bad depth, the previous depth was BEST. return those nodes
            print(depth_counter)
            return best_nodes, best_dim
        depth_counter += 1

# Given the wavelettree and input X, traverse to the deepest level where all nodes are the same dim and return the embeddings at that level
# All pts will have embeddings, and the embeddings will be in the same order as X (first row of output maps to pt in first row of X)
def get_embeddings(tree, X):
    #returns the embeddings matrix and the idxs map
    nodes, dim = best_depth(tree.root)

    print(dim)

    #aggregate nodes along depth
    #need basis, idxs, and sigmas
    basis = np.vstack([node.basis for node in nodes])
    idxs = np.hstack([node.idxs for node in nodes])
    sigmas = np.hstack([node.sigmas[:-1] for node in nodes])

    #TODO check dimensions of basis, idxs, sigmas
    #expect: basis nxd where n is 1000 (num nodes) and d is the best dim
    #idxs is a column vector of length 1000 (node idxs corresponding to the elements in the basis)
    #sigmas is a column vector of length 1000 (scaling factors for each basis vector)
    try:
        embeddings = np.multiply(basis, sigmas.reshape((basis.shape[0],1)))
    except:
        embeddings = basis
    # print(embeddings.shape)
    #we need to reorder embeddings based on sigmas 
    reordered_embs = np.zeros((X.shape[0],embeddings.shape[1]))
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        reordered_embs[new_idx] = embeddings[idx]
    return reordered_embs

#Compute max scale for the covertree, taking in input X. Computes log of max of pairwise pt distances
def calculate_max_scale(dataset):
     # Check if the input is a PyTorch tensor
    if hasattr(dataset, 'detach'):
        # Convert PyTorch tensor to NumPy array
        dataset = dataset.detach().cpu().numpy()
    # Check if the input is a NumPy array
    if not isinstance(dataset, np.ndarray):
        raise ValueError("Invalid type for dataset. Must be a PyTorch tensor or a NumPy array.")
    # Computes the squared Euclidean distance between the vectors.
    distances = pdist(dataset, 'sqeuclidean')
    # Check if the distances array is not empty
    if len(distances) == 0:
        raise ValueError("No distances computed from the dataset.")
    # Return the maximum of an array or maximum along an axis.
    max_distance = np.max(distances)
    # Check if max_distance is a valid, positive numerical value
    if not (np.isfinite(max_distance) and max_distance > 0):
        raise ValueError("Invalid value for max_distance. Must be a positive numerical value.")
    # returns the base 2 logarithm of a number.
    log_value = math.log2(max_distance)
    # Check if log_value is a valid numerical value
    if not np.isfinite(log_value):
        raise ValueError("Invalid value for log_value.")
    # rounds up and returns the smallest integer greater than or equal to a given number.
    max_scale = math.ceil(log_value)
    # Check if max_scale is a valid numerical value
    if not np.isfinite(max_scale):
        raise ValueError("Invalid value for max_scale.")
    return max_scale
