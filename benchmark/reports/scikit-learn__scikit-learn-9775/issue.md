# scikit-learn__scikit-learn-9775

* repo: scikit-learn/scikit-learn
* base_commit: `5815bd58667da900814d8780d2a5ebfb976c08b1`

## Problem statement

sklearn.manifold.t_sne.trustworthiness should allow custom metric
`precomputed` boolean parameter should be replaced by more standard `metric='precomputed'`.


## Patch

```diff
diff --git a/sklearn/manifold/t_sne.py b/sklearn/manifold/t_sne.py
--- a/sklearn/manifold/t_sne.py
+++ b/sklearn/manifold/t_sne.py
@@ -9,6 +9,7 @@
 #   http://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf
 from __future__ import division
 
+import warnings
 from time import time
 import numpy as np
 from scipy import linalg
@@ -394,7 +395,8 @@ def _gradient_descent(objective, p0, it, n_iter,
     return p, error, i
 
 
-def trustworthiness(X, X_embedded, n_neighbors=5, precomputed=False):
+def trustworthiness(X, X_embedded, n_neighbors=5,
+                    precomputed=False, metric='euclidean'):
     r"""Expresses to what extent the local structure is retained.
 
     The trustworthiness is within [0, 1]. It is defined as
@@ -431,15 +433,28 @@ def trustworthiness(X, X_embedded, n_neighbors=5, precomputed=False):
     precomputed : bool, optional (default: False)
         Set this flag if X is a precomputed square distance matrix.
 
+        ..deprecated:: 0.20
+            ``precomputed`` has been deprecated in version 0.20 and will be
+            removed in version 0.22. Use ``metric`` instead.
+
+    metric : string, or callable, optional, default 'euclidean'
+        Which metric to use for computing pairwise distances between samples
+        from the original input space. If metric is 'precomputed', X must be a
+        matrix of pairwise distances or squared distances. Otherwise, see the
+        documentation of argument metric in sklearn.pairwise.pairwise_distances
+        for a list of available metrics.
+
     Returns
     -------
     trustworthiness : float
         Trustworthiness of the low-dimensional embedding.
     """
     if precomputed:
-        dist_X = X
-    else:
-        dist_X = pairwise_distances(X, squared=True)
+        warnings.warn("The flag 'precomputed' has been deprecated in version "
+                      "0.20 and will be removed in 0.22. See 'metric' "
+                      "parameter instead.", DeprecationWarning)
+        metric = 'precomputed'
+    dist_X = pairwise_distances(X, metric=metric)
     ind_X = np.argsort(dist_X, axis=1)
     ind_X_embedded = NearestNeighbors(n_neighbors).fit(X_embedded).kneighbors(
         return_distance=False)

```

## Test Patch

```diff
diff --git a/sklearn/manifold/tests/test_t_sne.py b/sklearn/manifold/tests/test_t_sne.py
--- a/sklearn/manifold/tests/test_t_sne.py
+++ b/sklearn/manifold/tests/test_t_sne.py
@@ -14,6 +14,8 @@
 from sklearn.utils.testing import assert_greater
 from sklearn.utils.testing import assert_raises_regexp
 from sklearn.utils.testing import assert_in
+from sklearn.utils.testing import assert_warns
+from sklearn.utils.testing import assert_raises
 from sklearn.utils.testing import skip_if_32bit
 from sklearn.utils import check_random_state
 from sklearn.manifold.t_sne import _joint_probabilities
@@ -288,11 +290,39 @@ def test_preserve_trustworthiness_approximately_with_precomputed_distances():
                     early_exaggeration=2.0, metric="precomputed",
                     random_state=i, verbose=0)
         X_embedded = tsne.fit_transform(D)
-        t = trustworthiness(D, X_embedded, n_neighbors=1,
-                            precomputed=True)
+        t = trustworthiness(D, X_embedded, n_neighbors=1, metric="precomputed")
         assert t > .95
 
 
+def test_trustworthiness_precomputed_deprecation():
+    # FIXME: Remove this test in v0.23
+
+    # Use of the flag `precomputed` in trustworthiness parameters has been
+    # deprecated, but will still work until v0.23.
+    random_state = check_random_state(0)
+    X = random_state.randn(100, 2)
+    assert_equal(assert_warns(DeprecationWarning, trustworthiness,
+                              pairwise_distances(X), X, precomputed=True), 1.)
+    assert_equal(assert_warns(DeprecationWarning, trustworthiness,
+                              pairwise_distances(X), X, metric='precomputed',
+                              precomputed=True), 1.)
+    assert_raises(ValueError, assert_warns, DeprecationWarning,
+                  trustworthiness, X, X, metric='euclidean', precomputed=True)
+    assert_equal(assert_warns(DeprecationWarning, trustworthiness,
+                              pairwise_distances(X), X, metric='euclidean',
+                              precomputed=True), 1.)
+
+
+def test_trustworthiness_not_euclidean_metric():
+    # Test trustworthiness with a metric different from 'euclidean' and
+    # 'precomputed'
+    random_state = check_random_state(0)
+    X = random_state.randn(100, 2)
+    assert_equal(trustworthiness(X, X, metric='cosine'),
+                 trustworthiness(pairwise_distances(X, metric='cosine'), X,
+                                 metric='precomputed'))
+
+
 def test_early_exaggeration_too_small():
     # Early exaggeration factor must be >= 1.
     tsne = TSNE(early_exaggeration=0.99)

```
