# scikit-learn__scikit-learn-13828

| **scikit-learn/scikit-learn** | `f23e92ed4cdc5a952331e597023bd2c9922e6f9d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 673 |
| **Any found context length** | 673 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/cluster/affinity_propagation_.py b/sklearn/cluster/affinity_propagation_.py
--- a/sklearn/cluster/affinity_propagation_.py
+++ b/sklearn/cluster/affinity_propagation_.py
@@ -364,7 +364,11 @@ def fit(self, X, y=None):
         y : Ignored
 
         """
-        X = check_array(X, accept_sparse='csr')
+        if self.affinity == "precomputed":
+            accept_sparse = False
+        else:
+            accept_sparse = 'csr'
+        X = check_array(X, accept_sparse=accept_sparse)
         if self.affinity == "precomputed":
             self.affinity_matrix_ = X
         elif self.affinity == "euclidean":

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/cluster/affinity_propagation_.py | 367 | 367 | 2 | 1 | 673


## Problem Statement

```
sklearn.cluster.AffinityPropagation doesn't support sparse affinity matrix
<!--
If your issue is a usage question, submit it here instead:
- StackOverflow with the scikit-learn tag: https://stackoverflow.com/questions/tagged/scikit-learn
- Mailing List: https://mail.python.org/mailman/listinfo/scikit-learn
For more information, see User Questions: http://scikit-learn.org/stable/support.html#user-questions
-->

<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
sklearn.cluster.AffinityPropagation doesn't support sparse affinity matrix.
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->
A similar question is at #4051. It focuses on default affinity.
#### Steps/Code to Reproduce
\`\`\`python
from sklearn.cluster import AffinityPropagation
from scipy.sparse import csr
affinity_matrix = csr.csr_matrix((3,3))
AffinityPropagation(affinity='precomputed').fit(affinity_matrix)
\`\`\`


#### Expected Results
no error raised since it works for dense matrix.

#### Actual Results

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "D:\Miniconda\lib\site-packages\sklearn\cluster\affinity_propagation_.py", line 381, in fit
    copy=self.copy, verbose=self.verbose, return_n_iter=True)
  File "D:\Miniconda\lib\site-packages\sklearn\cluster\affinity_propagation_.py", line 115, in affinity_propagation
    preference = np.median(S)
  File "D:\Miniconda\lib\site-packages\numpy\lib\function_base.py", line 3336, in median
    overwrite_input=overwrite_input)
  File "D:\Miniconda\lib\site-packages\numpy\lib\function_base.py", line 3250, in _ureduce
    r = func(a, **kwargs)
  File "D:\Miniconda\lib\site-packages\numpy\lib\function_base.py", line 3395, in _median
    return mean(part[indexer], axis=axis, out=out)
  File "D:\Miniconda\lib\site-packages\numpy\core\fromnumeric.py", line 2920, in mean
    out=out, **kwargs)
  File "D:\Miniconda\lib\site-packages\numpy\core\_methods.py", line 85, in _mean
    ret = ret.dtype.type(ret / rcount)
ValueError: setting an array element with a sequence.

#### Versions
System:
    python: 3.6.7 |Anaconda, Inc.| (default, Oct 28 2018, 19:44:12) [MSC v.1915 64 bit (AMD64)]
executable: D:\Miniconda\python.exe
   machine: Windows-7-6.1.7601-SP1

BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: D:/Miniconda\Library\lib
cblas_libs: mkl_rt

Python deps:
       pip: 18.1
setuptools: 40.6.2
   sklearn: 0.20.1
     numpy: 1.15.4
     scipy: 1.1.0
    Cython: None
    pandas: 0.23.4


<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/cluster/affinity_propagation_.py** | 207 | 231| 278 | 278 | 3385 | 
| **-> 2 <-** | **1 sklearn/cluster/affinity_propagation_.py** | 337 | 386| 395 | 673 | 3385 | 
| 3 | **1 sklearn/cluster/affinity_propagation_.py** | 234 | 335| 879 | 1552 | 3385 | 
| 4 | **1 sklearn/cluster/affinity_propagation_.py** | 121 | 205| 796 | 2348 | 3385 | 
| 5 | 2 examples/cluster/plot_affinity_propagation.py | 1 | 64| 533 | 2881 | 3918 | 
| 6 | **2 sklearn/cluster/affinity_propagation_.py** | 1 | 30| 171 | 3052 | 3918 | 
| 7 | **2 sklearn/cluster/affinity_propagation_.py** | 33 | 119| 730 | 3782 | 3918 | 
| 8 | 3 sklearn/cluster/spectral.py | 446 | 499| 462 | 4244 | 8431 | 
| 9 | 3 sklearn/cluster/spectral.py | 159 | 249| 871 | 5115 | 8431 | 
| 10 | 4 benchmarks/bench_sparsify.py | 83 | 106| 153 | 5268 | 9338 | 
| 11 | 5 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 5779 | 9871 | 
| 12 | 5 sklearn/cluster/spectral.py | 275 | 426| 1469 | 7248 | 9871 | 
| 13 | 5 sklearn/cluster/spectral.py | 250 | 272| 248 | 7496 | 9871 | 
| 14 | 6 sklearn/random_projection.py | 262 | 290| 292 | 7788 | 15008 | 
| 15 | 7 sklearn/cluster/hierarchical.py | 10 | 26| 126 | 7914 | 24016 | 
| 16 | 8 sklearn/metrics/pairwise.py | 1 | 33| 163 | 8077 | 38805 | 
| 17 | 9 examples/linear_model/plot_lasso_dense_vs_sparse_data.py | 1 | 67| 511 | 8588 | 39316 | 
| 18 | 10 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 95 | 109| 174 | 8762 | 40300 | 
| 19 | 11 sklearn/linear_model/base.py | 310 | 332| 182 | 8944 | 45015 | 
| 20 | 12 benchmarks/bench_feature_expansions.py | 1 | 50| 467 | 9411 | 45482 | 
| 21 | 12 benchmarks/bench_sparsify.py | 1 | 82| 754 | 10165 | 45482 | 
| 22 | 13 sklearn/semi_supervised/label_propagation.py | 405 | 491| 703 | 10868 | 49632 | 
| 23 | 14 sklearn/utils/estimator_checks.py | 1311 | 1367| 575 | 11443 | 71442 | 
| 24 | 14 sklearn/utils/estimator_checks.py | 502 | 550| 491 | 11934 | 71442 | 
| 25 | 14 sklearn/utils/estimator_checks.py | 1 | 62| 452 | 12386 | 71442 | 
| 26 | 15 sklearn/utils/sparsefuncs.py | 6 | 25| 173 | 12559 | 75381 | 
| 27 | 16 sklearn/manifold/spectral_embedding_.py | 426 | 486| 536 | 13095 | 80391 | 
| 28 | 17 examples/cluster/plot_agglomerative_clustering.py | 1 | 81| 671 | 13766 | 81086 | 
| 29 | 18 benchmarks/bench_random_projections.py | 66 | 82| 174 | 13940 | 82821 | 
| 30 | 19 sklearn/decomposition/sparse_pca.py | 1 | 125| 934 | 14874 | 85923 | 
| 31 | 20 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 15433 | 87058 | 
| 32 | 21 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 15796 | 88176 | 
| 33 | 22 examples/cluster/plot_cluster_comparison.py | 82 | 92| 180 | 15976 | 89859 | 
| 34 | 23 sklearn/cluster/bicluster.py | 7 | 26| 125 | 16101 | 94764 | 
| 35 | 24 sklearn/utils/fixes.py | 42 | 155| 798 | 16899 | 96604 | 
| 36 | 24 sklearn/cluster/hierarchical.py | 28 | 78| 411 | 17310 | 96604 | 
| 37 | 24 sklearn/linear_model/base.py | 334 | 361| 256 | 17566 | 96604 | 
| 38 | 24 sklearn/utils/estimator_checks.py | 469 | 499| 295 | 17861 | 96604 | 
| 39 | 25 sklearn/preprocessing/data.py | 11 | 59| 309 | 18170 | 121793 | 
| 40 | 26 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 361 | 18531 | 135721 | 
| 41 | 27 sklearn/metrics/cluster/bicluster.py | 1 | 27| 228 | 18759 | 136438 | 
| 42 | 28 sklearn/linear_model/coordinate_descent.py | 8 | 27| 149 | 18908 | 157531 | 
| 43 | 28 sklearn/random_projection.py | 617 | 650| 211 | 19119 | 157531 | 
| 44 | 28 sklearn/random_projection.py | 196 | 260| 574 | 19693 | 157531 | 
| 45 | 29 sklearn/linear_model/ridge.py | 37 | 113| 635 | 20328 | 174132 | 
| 46 | 30 examples/inspection/plot_partial_dependence.py | 72 | 138| 653 | 20981 | 175473 | 
| 47 | 31 sklearn/utils/random.py | 4 | 98| 802 | 21783 | 176301 | 
| 48 | 32 examples/plot_johnson_lindenstrauss_bound.py | 94 | 168| 709 | 22492 | 178144 | 
| 49 | 33 sklearn/linear_model/least_angle.py | 475 | 632| 1591 | 24083 | 194638 | 
| 50 | 33 sklearn/manifold/spectral_embedding_.py | 338 | 424| 830 | 24913 | 194638 | 


### Hint

```
Yes, it should be providing a better error message. A pull request doing so
is welcome.

I don't know affinity propagation well enough to comment on whether we
should support a sparse graph as we do with dbscan.. This is applicable
only when a sample's nearest neighbours are all that is required to cluster
the sample.

For DBSCAN algorithm, sparse distance matrix is supported.
I will make a pull request to fix the support of sparse affinity matrix of Affinity Propagation.
It seems numpy does not support calculate the median value of a sparse matrix:
\`\`\`python
from scipy.sparse import csr
a=csr.csr_matrix((3,3))
np.mean(a)
# 0.0
np.median(a)
# raise Error similar as above
\`\`\`
How to fix this ?
DBSCAN supports sparse distance matrix because it depends on nearest
neighborhoods, not on all distances.

I'm not convinced this can be done.

Affinity Propagation extensively use dense matrix operation in its implementation.
I think of two ways to handle such situation.
1. sparse matrix be converted to dense in implementation 
2. or disallowed as input when affinity = 'precomputed'
The latter. A sparse distance matrix is a representation of a sparsely
connected graph. I don't think affinity propagation can be performed on it

raise an error when user provides a sparse matrix as input (when affinity = 'precomputed')?

Yes please

```

## Patch

```diff
diff --git a/sklearn/cluster/affinity_propagation_.py b/sklearn/cluster/affinity_propagation_.py
--- a/sklearn/cluster/affinity_propagation_.py
+++ b/sklearn/cluster/affinity_propagation_.py
@@ -364,7 +364,11 @@ def fit(self, X, y=None):
         y : Ignored
 
         """
-        X = check_array(X, accept_sparse='csr')
+        if self.affinity == "precomputed":
+            accept_sparse = False
+        else:
+            accept_sparse = 'csr'
+        X = check_array(X, accept_sparse=accept_sparse)
         if self.affinity == "precomputed":
             self.affinity_matrix_ = X
         elif self.affinity == "euclidean":

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_affinity_propagation.py b/sklearn/cluster/tests/test_affinity_propagation.py
--- a/sklearn/cluster/tests/test_affinity_propagation.py
+++ b/sklearn/cluster/tests/test_affinity_propagation.py
@@ -63,7 +63,8 @@ def test_affinity_propagation():
     assert_raises(ValueError, affinity_propagation, S, damping=0)
     af = AffinityPropagation(affinity="unknown")
     assert_raises(ValueError, af.fit, X)
-
+    af_2 = AffinityPropagation(affinity='precomputed')
+    assert_raises(TypeError, af_2.fit, csr_matrix((3, 3)))
 
 def test_affinity_propagation_predict():
     # Test AffinityPropagation.predict

```


## Code snippets

### 1 - sklearn/cluster/affinity_propagation_.py:

Start line: 207, End line: 231

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    # ... other code

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn("Affinity propagation did not converge, this model "
                      "will not have any cluster centers.", ConvergenceWarning)
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels
```
### 2 - sklearn/cluster/affinity_propagation_.py:

Start line: 337, End line: 386

```python
class AffinityPropagation(BaseEstimator, ClusterMixin):

    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def fit(self, X, y=None):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.

        y : Ignored

        """
        X = check_array(X, accept_sparse='csr')
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.cluster_centers_indices_, self.labels_, self.n_iter_ = \
            affinity_propagation(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self
```
### 3 - sklearn/cluster/affinity_propagation_.py:

Start line: 234, End line: 335

```python
###############################################################################

class AffinityPropagation(BaseEstimator, ClusterMixin):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, optional, default: 0.5
        Damping factor (between 0.5 and 1) is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, optional, default: 200
        Maximum number of iterations.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : boolean, optional, default: True
        Make a copy of input data.

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : string, optional, default=``euclidean``
        Which affinity to use. At the moment ``precomputed`` and
        ``euclidean`` are supported. ``euclidean`` uses the
        negative squared euclidean distance between points.

    verbose : boolean, optional, default: False
        Whether to be verbose.


    Attributes
    ----------
    cluster_centers_indices_ : array, shape (n_clusters,)
        Indices of cluster centers

    cluster_centers_ : array, shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : array, shape (n_samples,)
        Labels of each point

    affinity_matrix_ : array, shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation().fit(X)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
              damping=0.5, max_iter=200, preference=None, verbose=False)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
    array and all training samples will be labelled as ``-1``. In addition,
    ``predict`` will then label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
```
### 4 - sklearn/cluster/affinity_propagation_.py:

Start line: 121, End line: 205

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    # ... other code

    if (n_samples == 1 or
            _equal_similarities_and_preferences(S, preference)):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn("All samples have mutually equal similarities. "
                      "Returning arbitrary cluster center(s).")
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return ((np.arange(n_samples), np.arange(n_samples), 0)
                    if return_n_iter
                    else (np.arange(n_samples), np.arange(n_samples)))
        else:
            return ((np.array([0]), np.array([0] * n_samples), 0)
                    if return_n_iter
                    else (np.array([0]), np.array([0] * n_samples)))

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars
    # ... other code
```
### 5 - examples/cluster/plot_affinity_propagation.py:

Start line: 1, End line: 64

```python
"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```
### 6 - sklearn/cluster/affinity_propagation_.py:

Start line: 1, End line: 30

```python
"""Affinity Propagation clustering algorithm."""

import numpy as np
import warnings

from ..exceptions import ConvergenceWarning
from ..base import BaseEstimator, ClusterMixin
from ..utils import as_float_array, check_array
from ..utils.validation import check_is_fitted
from ..metrics import euclidean_distances
from ..metrics import pairwise_distances_argmin


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()
```
### 7 - sklearn/cluster/affinity_propagation_.py:

Start line: 33, End line: 119

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    """Perform Affinity Propagation Clustering of data

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations

    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    verbose : boolean, optional, default: False
        The verbosity level

    return_n_iter : bool, default False
        Whether or not to return the number of iterations.

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    n_iter : int
        number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it returns an empty array as
    ``cluster_center_indices`` and ``-1`` as label for each training sample.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    preference = np.array(preference)
    # ... other code
```
### 8 - sklearn/cluster/spectral.py:

Start line: 446, End line: 499

```python
class SpectralClustering(BaseEstimator, ClusterMixin):

    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)

        y : Ignored

        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"
```
### 9 - sklearn/cluster/spectral.py:

Start line: 159, End line: 249

```python
def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    affinity : array-like or sparse matrix, shape: (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : integer, optional
        Number of clusters to extract.

    n_components : integer, optional, default is n_clusters
        Number of eigen vectors to use for the spectral embedding

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities

    random_state : int, RandomState instance or None (default)
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when eigen_solver == 'amg' and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    eigen_tol : float, optional, default: 0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : {'kmeans', 'discretize'}, default: 'kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are two ways to assign labels after the laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization. See
        the 'Multiclass spectral clustering' paper referenced below for
        more details on the discretization approach.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    -----
    The graph should contain only one connect component, elsewhere
    the results make little sense.

    This algorithm solves the normalized cut for k=2: it is a
    normalized spectral clustering.
    """
    # ... other code
```
### 10 - benchmarks/bench_sparsify.py:

Start line: 83, End line: 106

```python
print("model sparsity: %f" % sparsity_ratio(clf.coef_))


def benchmark_dense_predict():
    for _ in range(300):
        clf.predict(X_test)


def benchmark_sparse_predict():
    X_test_sparse = csr_matrix(X_test)
    for _ in range(300):
        clf.predict(X_test_sparse)


def score(y_test, y_pred, case):
    r2 = r2_score(y_test, y_pred)
    print("r^2 on test data (%s) : %f" % (case, r2))

score(y_test, clf.predict(X_test), 'dense model')
benchmark_dense_predict()
clf.sparsify()
score(y_test, clf.predict(X_test), 'sparse model')
benchmark_sparse_predict()
```
