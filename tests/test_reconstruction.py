import random
import numpy as np
import pytest
from scipy.spatial.distance import euclidean
from src.covertree import CoverTree
from src.dyadictree import DyadicTree
from src.helpers import rand_pca, node_function


def model(X, leafsize=8, inverse=False):
    return DyadicTree(CoverTree(X, euclidean, leafsize=leafsize, random_state=7),
                      X=X, manifold_dims=2, max_dim=3, thresholds=1e-9,
                      precisions=1e-6, inverse=inverse)


def test_randomized_svd_tall_and_wide():
    rng = np.random.default_rng(7)
    for shape in [(90, 60), (60, 90)]:
        A = rng.normal(size=(shape[0], 4)) @ rng.normal(size=(4, shape[1]))
        U, s, Vh = rand_pca(A, 4)
        assert U.shape == (shape[0], 4)
        assert Vh.shape == (4, shape[1])
        np.testing.assert_allclose((U * s) @ Vh, A, atol=1e-10)
        np.testing.assert_array_equal(rand_pca(A, 4)[1], s)


def test_inverse_agrees_with_leaf_affine_projection_on_held_out_data():
    rng = np.random.default_rng(7)
    X, query = rng.normal(size=(50, 8)), rng.normal(size=(13, 8))
    tree = model(X)
    transformed = tree.transform(query)
    reconstructed = tree.inverse_transform(transformed)
    leaves = tree.query_leaf_by_center(query)
    expected = np.array([(x - leaf.center.ravel()) @ leaf.basis.T @ leaf.basis + leaf.center.ravel()
                         for x, leaf in zip(query, leaves)])
    np.testing.assert_allclose(reconstructed, expected, atol=1e-8)
    assert reconstructed.shape == query.shape


def test_single_root_and_rank_zero_leaf():
    for X in [np.ones((1, 4)), np.random.default_rng(7).normal(size=(5, 4))]:
        tree = model(X, leafsize=20)
        reconstructed = tree.inverse_transform(tree.transform(X))
        expected = (X - tree.root.center.T) @ tree.root.basis.T @ tree.root.basis + tree.root.center.T
        np.testing.assert_allclose(reconstructed, expected, atol=1e-10)


def test_legacy_inverse_flag_does_not_change_basis_orientation():
    X = np.random.default_rng(7).normal(size=(40, 8))
    a, b = model(X), model(X, inverse=True)
    np.testing.assert_allclose(a.root.basis, b.root.basis)
    assert a.root.center.shape == (8, 1)
    assert a.root.size == 40


def test_tree_seed_and_fit_row_identity():
    X = np.random.default_rng(7).normal(size=(40, 8))
    first = model(X)
    random.seed(999)
    second = model(X)
    assert [list(n.idxs) for n in first.get_all_leafs()] == [list(n.idxs) for n in second.get_all_leafs()]
    with pytest.raises(ValueError):
        first.fit(X[::-1])
    assert first.transform(np.empty((0, 8))) == ([], [])
    assert first.inverse_transform(([], [])).shape == (0, 8)


def test_local_radius_and_size_count_samples_not_features():
    X = np.array([[0., 0., 0.], [2., 0., 0.]])
    center, size, radius, basis, _, _ = node_function(X, 1, 2, True)
    assert size == 2
    assert radius == 1
    assert basis.shape == (1, 3)
    np.testing.assert_array_equal(center.ravel(), [1, 0, 0])
