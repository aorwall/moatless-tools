**0 - /tmp/repos/scikit-learn/sklearn/semi_supervised/_label_propagation.py**:
```python
# coding=utf8
"""
Label propagation in the context of this module refers to a set of
semi-supervised classification algorithms. At a high level, these algorithms
work by forming a fully-connected graph between all points given and solving
for the steady-state distribution of labels at each point.

These algorithms perform very well in practice. The cost of running can be very
expensive, at approximately O(N^3) where N is the number of (labeled and
unlabeled) points. The theory (why they perform so well) is motivated by
intuitions from random walk algorithms and geometric relationships in the data.
For more information see the references below.

Model Features
--------------
Label clamping:
  The algorithm tries to learn distributions of labels over the dataset given
  label assignments over an initial subset. In one variant, the algorithm does
  not allow for any errors in the initial assignment (hard-clamping) while
  in another variant, the algorithm allows for some wiggle room for the initial
  assignments, allowing them to change by a fraction alpha in each iteration
  (soft-clamping).

Kernel:
  A function which projects a vector into some higher dimensional space. This
  implementation supports RBF and KNN kernels. Using the RBF kernel generates
  a dense matrix of size O(N^2). KNN kernel will generate a sparse matrix of
  size O(k*N) which will run much faster. See the documentation for SVMs for
  more info on kernels.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelPropagation
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelPropagation(...)

Notes
-----
References:
[1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
Learning (2006), pp. 193-216

[2] Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient
Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005
"""

# Authors: Clay Woolam <clay@woolam.org>
#          Utkarsh Upadhyay <mail@musicallyut.in>
# License: BSD
from abc import ABCMeta, abstractmethod

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from ..base import BaseEstimator, ClassifierMixin
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_X_y, check_is_fitted, check_array
from ..exceptions import ConvergenceWarning
```
