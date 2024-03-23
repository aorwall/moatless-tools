# scikit-learn__scikit-learn-10870

* repo: scikit-learn/scikit-learn
* base_commit: `b0e91e4110942e5b3c4333b1c6b6dfefbd1a6124`

## Problem statement

In Gaussian mixtures, when n_init > 1, the lower_bound_ is not always the max
#### Description
In Gaussian mixtures, when `n_init` is set to any value greater than 1, the `lower_bound_` is not the max lower bound across all initializations, but just the lower bound of the last initialization.

The bug can be fixed by adding the following line just before `return self` in `BaseMixture.fit()`:

```python
self.lower_bound_ = max_lower_bound
```

The test that should have caught this bug is `test_init()` in `mixture/tests/test_gaussian_mixture.py`, but it just does a single test, so it had a 50% chance of missing the issue. It should be updated to try many random states.

#### Steps/Code to Reproduce
```python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.rand(1000, 10)
for random_state in range(100):
    gm1 = GaussianMixture(n_components=2, n_init=1, random_state=random_state).fit(X)
    gm2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state).fit(X)
    assert gm2.lower_bound_ > gm1.lower_bound_, random_state
```

#### Expected Results
No error.

#### Actual Results
```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AssertionError: 4
```

#### Versions

```
>>> import platform; print(platform.platform())
Darwin-17.4.0-x86_64-i386-64bit
>>> import sys; print("Python", sys.version)
Python 3.6.4 (default, Dec 21 2017, 20:33:21)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.38)]
>>> import numpy; print("NumPy", numpy.__version__)
NumPy 1.14.2
>>> import scipy; print("SciPy", scipy.__version__)
SciPy 1.0.0
>>> import sklearn; print("Scikit-Learn", sklearn.__version__)
Scikit-Learn 0.19.1
```
In Gaussian mixtures, when n_init > 1, the lower_bound_ is not always the max
#### Description
In Gaussian mixtures, when `n_init` is set to any value greater than 1, the `lower_bound_` is not the max lower bound across all initializations, but just the lower bound of the last initialization.

The bug can be fixed by adding the following line just before `return self` in `BaseMixture.fit()`:

```python
self.lower_bound_ = max_lower_bound
```

The test that should have caught this bug is `test_init()` in `mixture/tests/test_gaussian_mixture.py`, but it just does a single test, so it had a 50% chance of missing the issue. It should be updated to try many random states.

#### Steps/Code to Reproduce
```python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.rand(1000, 10)
for random_state in range(100):
    gm1 = GaussianMixture(n_components=2, n_init=1, random_state=random_state).fit(X)
    gm2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state).fit(X)
    assert gm2.lower_bound_ > gm1.lower_bound_, random_state
```

#### Expected Results
No error.

#### Actual Results
```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AssertionError: 4
```

#### Versions

```
>>> import platform; print(platform.platform())
Darwin-17.4.0-x86_64-i386-64bit
>>> import sys; print("Python", sys.version)
Python 3.6.4 (default, Dec 21 2017, 20:33:21)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.38)]
>>> import numpy; print("NumPy", numpy.__version__)
NumPy 1.14.2
>>> import scipy; print("SciPy", scipy.__version__)
SciPy 1.0.0
>>> import sklearn; print("Scikit-Learn", sklearn.__version__)
Scikit-Learn 0.19.1
```


## Patch

```diff
diff --git a/sklearn/mixture/base.py b/sklearn/mixture/base.py
--- a/sklearn/mixture/base.py
+++ b/sklearn/mixture/base.py
@@ -172,11 +172,14 @@ def _initialize(self, X, resp):
     def fit(self, X, y=None):
         """Estimate model parameters with the EM algorithm.
 
-        The method fits the model `n_init` times and set the parameters with
+        The method fits the model ``n_init`` times and sets the parameters with
         which the model has the largest likelihood or lower bound. Within each
-        trial, the method iterates between E-step and M-step for `max_iter`
+        trial, the method iterates between E-step and M-step for ``max_iter``
         times until the change of likelihood or lower bound is less than
-        `tol`, otherwise, a `ConvergenceWarning` is raised.
+        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
+        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
+        initialization is performed upon the first call. Upon consecutive
+        calls, training starts where it left off.
 
         Parameters
         ----------
@@ -232,27 +235,28 @@ def fit_predict(self, X, y=None):
 
             if do_init:
                 self._initialize_parameters(X, random_state)
-                self.lower_bound_ = -np.infty
+
+            lower_bound = (-np.infty if do_init else self.lower_bound_)
 
             for n_iter in range(1, self.max_iter + 1):
-                prev_lower_bound = self.lower_bound_
+                prev_lower_bound = lower_bound
 
                 log_prob_norm, log_resp = self._e_step(X)
                 self._m_step(X, log_resp)
-                self.lower_bound_ = self._compute_lower_bound(
+                lower_bound = self._compute_lower_bound(
                     log_resp, log_prob_norm)
 
-                change = self.lower_bound_ - prev_lower_bound
+                change = lower_bound - prev_lower_bound
                 self._print_verbose_msg_iter_end(n_iter, change)
 
                 if abs(change) < self.tol:
                     self.converged_ = True
                     break
 
-            self._print_verbose_msg_init_end(self.lower_bound_)
+            self._print_verbose_msg_init_end(lower_bound)
 
-            if self.lower_bound_ > max_lower_bound:
-                max_lower_bound = self.lower_bound_
+            if lower_bound > max_lower_bound:
+                max_lower_bound = lower_bound
                 best_params = self._get_parameters()
                 best_n_iter = n_iter
 
@@ -265,6 +269,7 @@ def fit_predict(self, X, y=None):
 
         self._set_parameters(best_params)
         self.n_iter_ = best_n_iter
+        self.lower_bound_ = max_lower_bound
 
         return log_resp.argmax(axis=1)
 
diff --git a/sklearn/mixture/gaussian_mixture.py b/sklearn/mixture/gaussian_mixture.py
--- a/sklearn/mixture/gaussian_mixture.py
+++ b/sklearn/mixture/gaussian_mixture.py
@@ -512,6 +512,8 @@ class GaussianMixture(BaseMixture):
         If 'warm_start' is True, the solution of the last fitting is used as
         initialization for the next call of fit(). This can speed up
         convergence when fit is called several times on similar problems.
+        In that case, 'n_init' is ignored and only a single initialization
+        occurs upon the first call.
         See :term:`the Glossary <warm_start>`.
 
     verbose : int, default to 0.
@@ -575,7 +577,8 @@ class GaussianMixture(BaseMixture):
         Number of step used by the best fit of EM to reach the convergence.
 
     lower_bound_ : float
-        Log-likelihood of the best fit of EM.
+        Lower bound value on the log-likelihood (of the training data with
+        respect to the model) of the best fit of EM.
 
     See Also
     --------

```

## Test Patch

```diff
diff --git a/sklearn/mixture/tests/test_gaussian_mixture.py b/sklearn/mixture/tests/test_gaussian_mixture.py
--- a/sklearn/mixture/tests/test_gaussian_mixture.py
+++ b/sklearn/mixture/tests/test_gaussian_mixture.py
@@ -764,7 +764,6 @@ def test_gaussian_mixture_verbose():
 
 
 def test_warm_start():
-
     random_state = 0
     rng = np.random.RandomState(random_state)
     n_samples, n_features, n_components = 500, 2, 2
@@ -806,6 +805,25 @@ def test_warm_start():
     assert_true(h.converged_)
 
 
+@ignore_warnings(category=ConvergenceWarning)
+def test_convergence_detected_with_warm_start():
+    # We check that convergence is detected when warm_start=True
+    rng = np.random.RandomState(0)
+    rand_data = RandomData(rng)
+    n_components = rand_data.n_components
+    X = rand_data.X['full']
+
+    for max_iter in (1, 2, 50):
+        gmm = GaussianMixture(n_components=n_components, warm_start=True,
+                              max_iter=max_iter, random_state=rng)
+        for _ in range(100):
+            gmm.fit(X)
+            if gmm.converged_:
+                break
+        assert gmm.converged_
+        assert max_iter >= gmm.n_iter_
+
+
 def test_score():
     covar_type = 'full'
     rng = np.random.RandomState(0)
@@ -991,14 +1009,14 @@ def test_sample():
 @ignore_warnings(category=ConvergenceWarning)
 def test_init():
     # We check that by increasing the n_init number we have a better solution
-    random_state = 0
-    rand_data = RandomData(np.random.RandomState(random_state), scale=1)
-    n_components = rand_data.n_components
-    X = rand_data.X['full']
+    for random_state in range(25):
+        rand_data = RandomData(np.random.RandomState(random_state), scale=1)
+        n_components = rand_data.n_components
+        X = rand_data.X['full']
 
-    gmm1 = GaussianMixture(n_components=n_components, n_init=1,
-                           max_iter=1, random_state=random_state).fit(X)
-    gmm2 = GaussianMixture(n_components=n_components, n_init=100,
-                           max_iter=1, random_state=random_state).fit(X)
+        gmm1 = GaussianMixture(n_components=n_components, n_init=1,
+                               max_iter=1, random_state=random_state).fit(X)
+        gmm2 = GaussianMixture(n_components=n_components, n_init=10,
+                               max_iter=1, random_state=random_state).fit(X)
 
-    assert_greater(gmm2.lower_bound_, gmm1.lower_bound_)
+        assert gmm2.lower_bound_ >= gmm1.lower_bound_

```
