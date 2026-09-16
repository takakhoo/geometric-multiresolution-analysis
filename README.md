# Geometric Multi-Resolution Analysis

A research implementation of Geometric Multi-Resolution Analysis (GMRA) for
building multiscale approximations of high-dimensional point clouds.

## What is here

- `src/covertree.py` — metric cover tree and nearest-neighbor operations.
- `src/dyadictree.py` — multiscale partition construction.
- `src/wavelettree.py` — local affine models and wavelet transforms.
- `experiments/` — classification and image-dataset studies.
- `tests/` — numerical tests for the core tree operations.
- `COMPREHENSIVE_GMRA_EXPLANATION.tex` — mathematical background and design notes.

This is research code, not a packaged production library. Several experiments
require large external datasets and additional machine-learning dependencies.

## Quick start

```bash
git clone https://github.com/takakhoo/geometric-multiresolution-analysis.git
cd geometric-multiresolution-analysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install numpy scipy pytest
python -m pytest -q
```

## Minimal example

```python
import numpy as np
from scipy.spatial.distance import euclidean

from src.covertree import CoverTree
from src.dyadictree import DyadicTree

rng = np.random.default_rng(7)
samples = rng.standard_normal((500, 32))
cover_tree = CoverTree(samples, euclidean, leafsize=10)
gmra = DyadicTree(
    cover_tree,
    X=samples,
    manifold_dims=6,
    max_dim=12,
    thresholds=0.5,
    precisions=1e-2,
)
gmra.fit(samples)
coefficients, leaf_indices = gmra.transform(samples)
```

## Experiment dependencies

The experiment folders cover MNIST, CIFAR-10, medical imaging, and classifier
comparisons. Install only the dependencies needed for the experiment you plan
to run; common additions include PyTorch, scikit-learn, pandas, Matplotlib,
Hydra, and TensorBoard. CUDA can accelerate larger image experiments but is not
required by the core tree implementation.

## Validation

The core suite exercises nearest-neighbor queries, ball queries, pair queries,
neighbor counting, and sparse distance matrices:

```bash
python -m pytest -q
```

Verified on September 16, 2026: **53 tests passed** across the core cover-tree
and multiscale geometry operations.

## Limitations

- Experiment scripts are exploratory and do not share one locked runtime.
- Dataset paths and compute requirements vary by experiment.
- Benchmark claims should be reproduced in a controlled environment before
  drawing performance conclusions.

## Acknowledgments

The cover-tree tests retain their original SciPy-style copyright notices. See
individual source files for attribution and licensing notes.
