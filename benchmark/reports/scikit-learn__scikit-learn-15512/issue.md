# scikit-learn__scikit-learn-15512

* repo: scikit-learn/scikit-learn
* base_commit: `b8a4da8baa1137f173e7035f104067c7d2ffde22`

## Problem statement

Return values of non converged affinity propagation clustering
The affinity propagation Documentation states: 
"When the algorithm does not converge, it returns an empty array as cluster_center_indices and -1 as label for each training sample."

Example:
```python
from sklearn.cluster import AffinityPropagation
import pandas as pd

data = pd.DataFrame([[1,0,0,0,0,0],[0,1,1,1,0,0],[0,0,1,0,0,1]])
af = AffinityPropagation(affinity='euclidean', verbose=True, copy=False, max_iter=2).fit(data)

print(af.cluster_centers_indices_)
print(af.labels_)

```
I would expect that the clustering here (which does not converge) prints first an empty List and then [-1,-1,-1], however, I get [2] as cluster center and [0,0,0] as cluster labels. 
The only way I currently know if the clustering fails is if I use the verbose option, however that is very unhandy. A hacky solution is to check if max_iter == n_iter_ but it could have converged exactly 15 iterations before max_iter (although unlikely).
I am not sure if this is intended behavior and the documentation is wrong?

For my use-case within a bigger script, I would prefer to get back -1 values or have a property to check if it has converged, as otherwise, a user might not be aware that the clustering never converged.


#### Versions
System:
    python: 3.6.7 | packaged by conda-forge | (default, Nov 21 2018, 02:32:25)  [GCC 4.8.2 20140120 (Red Hat 4.8.2-15)]
executable: /home/jenniferh/Programs/anaconda3/envs/TF_RDKit_1_19/bin/python
   machine: Linux-4.15.0-52-generic-x86_64-with-debian-stretch-sid
BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /home/jenniferh/Programs/anaconda3/envs/TF_RDKit_1_19/lib
cblas_libs: mkl_rt, pthread
Python deps:
    pip: 18.1
   setuptools: 40.6.3
   sklearn: 0.20.3
   numpy: 1.15.4
   scipy: 1.2.0
   Cython: 0.29.2
   pandas: 0.23.4




## Patch

```diff
diff --git a/sklearn/cluster/_affinity_propagation.py b/sklearn/cluster/_affinity_propagation.py
--- a/sklearn/cluster/_affinity_propagation.py
+++ b/sklearn/cluster/_affinity_propagation.py
@@ -194,17 +194,19 @@ def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
             unconverged = (np.sum((se == convergence_iter) + (se == 0))
                            != n_samples)
             if (not unconverged and (K > 0)) or (it == max_iter):
+                never_converged = False
                 if verbose:
                     print("Converged after %d iterations." % it)
                 break
     else:
+        never_converged = True
         if verbose:
             print("Did not converge")
 
     I = np.flatnonzero(E)
     K = I.size  # Identify exemplars
 
-    if K > 0:
+    if K > 0 and not never_converged:
         c = np.argmax(S[:, I], axis=1)
         c[I] = np.arange(K)  # Identify clusters
         # Refine the final set of exemplars and clusters and return results
@@ -408,6 +410,7 @@ def predict(self, X):
             Cluster labels.
         """
         check_is_fitted(self)
+        X = check_array(X)
         if not hasattr(self, "cluster_centers_"):
             raise ValueError("Predict method is not supported when "
                              "affinity='precomputed'.")

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_affinity_propagation.py b/sklearn/cluster/tests/test_affinity_propagation.py
--- a/sklearn/cluster/tests/test_affinity_propagation.py
+++ b/sklearn/cluster/tests/test_affinity_propagation.py
@@ -152,6 +152,14 @@ def test_affinity_propagation_predict_non_convergence():
     assert_array_equal(np.array([-1, -1, -1]), y)
 
 
+def test_affinity_propagation_non_convergence_regressiontest():
+    X = np.array([[1, 0, 0, 0, 0, 0],
+                  [0, 1, 1, 1, 0, 0],
+                  [0, 0, 1, 0, 0, 1]])
+    af = AffinityPropagation(affinity='euclidean', max_iter=2).fit(X)
+    assert_array_equal(np.array([-1, -1, -1]), af.labels_)
+
+
 def test_equal_similarities_and_preferences():
     # Unequal distances
     X = np.array([[0, 0], [1, 1], [-2, -2]])

```
