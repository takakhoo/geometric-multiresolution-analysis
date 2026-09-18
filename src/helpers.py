import numpy as np
from typing import List, Tuple, Union

def mindim(sigmas: np.ndarray,
           errortype: str,
           err: float) -> int:
    s2: float = np.sum(sigmas**2)

    tol: float = None
    if errortype.lower() == "absolute":
        tol = err**2
    else:
        tol = err*s2

    dim: int = 0
    while dim < sigmas.shape[0]-1 and s2>tol:
        s2 = s2 - sigmas[dim]**2

        dim += 1
    return dim


def rand_pca(A, k, its=2, l=None, shelf=None, inverse=False, random_state=7):
    """Seeded truncated SVD with a stable randomized range finder.

    Returns U (n,k), singular values (k,), Vh (k,m). The legacy inverse
    flag no longer changes dimensions. Disk-backed shelves are unsupported.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not min(A.shape) or not np.isfinite(A).all():
        raise ValueError("A must be a nonempty finite matrix")
    if not isinstance(k, (int, np.integer)) or k < 0 or its < 0:
        raise ValueError("k and power iterations must be nonnegative")
    if shelf is not None:
        raise NotImplementedError("Disk-backed PCA shelves are not supported")
    k = min(k, min(A.shape))
    width = min(min(A.shape), k + 8 if l is None else l)
    if width < k:
        raise ValueError("Sketch width cannot be smaller than k")
    if k == 0:
        return np.empty((A.shape[0], 0)), np.empty(0), np.empty((0, A.shape[1]))
    if width >= min(A.shape) or min(A.shape) <= 32:
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
    else:
        rng = np.random.default_rng(random_state)
        Q, _ = np.linalg.qr(A @ rng.standard_normal((A.shape[1], width)), mode="reduced")
        for _ in range(its):
            Z, _ = np.linalg.qr(A.T @ Q, mode="reduced")
            Q, _ = np.linalg.qr(A @ Z, mode="reduced")
        small_U, s, Vh = np.linalg.svd(Q.T @ A, full_matrices=False)
        U = Q @ small_U
    return U[:, :k], s[:k], Vh[:k]

def easy_pca(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt

def easy_node_function(X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
                       is_leaf: bool,
                       errortype: str = "relative",
                       shelf=None,
                       
                       threshold: float = 0.5,
                       precision: float = 1e-2) -> Tuple[np.ndarray, int, float,
                                                            np.ndarray, np.ndarray]:
    # X have shape (n, d)
    mu: np.ndarray = np.mean(X, axis=0, keepdims=True)
    X_norm = X - mu

    radius: float = np.sqrt(np.max((X_norm**2).sum(axis=-1)))
    size: int = max(1, X.shape[0])  
    max_dim = max(X_norm.shape)
    basis, sigmas, Z = easy_pca(X_norm, min(min(X_norm.shape), max_dim))
    rem_energy: float = max(np.sum(np.sum(X_norm**2) - np.sum(sigmas**2)), 0)
    return mu, X.shape[0], radius, basis, sigmas, Z
    
def node_function(X, manifold_dim, max_dim, is_leaf, errortype="relative",
                  shelf=None, threshold=0.5, precision=1e-2, inverse=False):
    """Local affine model for row-sample input, independent of legacy inverse flag.

    Center is (features,1), basis is (rank,features), size counts samples.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not min(X.shape) or not np.isfinite(X).all():
        raise ValueError("Node data must be a nonempty finite matrix")
    if max_dim is None or max_dim < 1 or manifold_dim < 0:
        raise ValueError("Positive max_dim and nonnegative manifold_dim required")
    center = X.mean(axis=0)[:, None]
    centered = X - center.T
    size = len(X)
    radius = float(np.linalg.norm(centered, axis=1).max())
    adaptive = is_leaf or manifold_dim == 0
    limit = min(min(centered.shape), max_dim if adaptive else min(max_dim, manifold_dim))
    U, singular, Vh = rand_pca(centered.T, limit, shelf=shelf)
    numerical_rank = int(np.sum(singular > (singular[0] * max(centered.shape) * np.finfo(float).eps))) if len(singular) else 0
    remainder = max(float(np.sum(centered ** 2) - np.sum(singular ** 2)), 0)
    sigmas = np.r_[singular, np.sqrt(remainder)] / np.sqrt(size)
    rank = min(numerical_rank, mindim(sigmas, errortype, precision if is_leaf else threshold)) if adaptive else numerical_rank
    return center, size, radius, U[:, :rank].T, sigmas, Vh
