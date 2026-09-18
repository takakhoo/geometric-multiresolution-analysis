"""Held-out multiscale reconstruction of a curved manifold versus global PCA."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import euclidean
from src.covertree import CoverTree
from src.dyadictree import DyadicTree


def run(output="results"):
    rows = []
    illustration = None
    for seed in [7, 19, 41]:
        rng = np.random.default_rng(seed)
        rotation, _ = np.linalg.qr(rng.normal(size=(12, 3)))
        def sample(count):
            t = rng.uniform(0, 4 * np.pi, count)
            latent = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
            clean = latent @ rotation.T
            return clean + rng.normal(scale=0.015, size=clean.shape), clean, t
        train, _, _ = sample(300)
        query, clean, t = sample(150)
        center = train.mean(axis=0)
        _, _, Vh = np.linalg.svd(train - center, full_matrices=False)
        global_reconstruction = (query - center) @ Vh[:2].T @ Vh[:2] + center
        rows.append(dict(seed=seed, method="global_pca_rank2", rmse_clean=float(np.sqrt(np.mean((global_reconstruction - clean) ** 2)))))
        for leafsize in [8, 24, 64]:
            tree = DyadicTree(CoverTree(train, euclidean, leafsize=leafsize, random_state=seed),
                              X=train, manifold_dims=2, max_dim=2, thresholds=1e-9, precisions=1e-4)
            reconstruction = tree.inverse_transform(tree.transform(query))
            leaves = tree.query_leaf_by_center(query)
            projection = np.array([(x - n.center.ravel()) @ n.basis.T @ n.basis + n.center.ravel()
                                   for x, n in zip(query, leaves)])
            error = float(np.max(np.abs(projection - reconstruction)))
            if error > 1e-7:
                raise AssertionError("Wavelet inverse disagrees with direct affine projection")
            rows.append(dict(seed=seed, method=f"gmra_leafsize{leafsize}", leaves=len(tree.get_all_leafs()),
                             rmse_clean=float(np.sqrt(np.mean((reconstruction - clean) ** 2))),
                             inverse_projection_max_error=error))
            if seed == 7 and leafsize == 24:
                illustration = (clean @ rotation, global_reconstruction @ rotation, reconstruction @ rotation, t)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(description="Synthetic noisy helix in 12 dimensions; 300 train and 150 independent held-out points per seed.",
                  seeds=[7, 19, 41], noise_std=0.015, local_max_rank=2, rows=rows,
                  caveat="Local models have more parameters and center/leaf overhead than one global PCA model; this is not a matched-storage compression benchmark.")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    for index, (title, points) in enumerate(zip(["Held-out clean helix", "One global rank-2 PCA", "Local rank ≤ 2 GMRA"], illustration[:3]), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(*points.T, c=illustration[3], s=9, cmap="viridis")
        ax.set(title=title, xlabel="Latent x", ylabel="Latent y", zlabel="Latent z",
               xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-0.1, 2.1))
        ax.view_init(elev=25, azim=40)
    fig.savefig(out / "held-out-reconstruction.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
