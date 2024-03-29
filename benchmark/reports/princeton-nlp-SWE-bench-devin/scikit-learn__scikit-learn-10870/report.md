# scikit-learn__scikit-learn-10870

| **scikit-learn/scikit-learn** | `b0e91e4110942e5b3c4333b1c6b6dfefbd1a6124` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 5624 |
| **Any found context length** | 5624 |
| **Avg pos** | 163.0 |
| **Min pos** | 9 |
| **Max pos** | 114 |
| **Top file pos** | 2 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/mixture/base.py | 175 | 179 | 114 | 4 | 50522
| sklearn/mixture/base.py | 235 | 255 | 97 | 4 | 42164
| sklearn/mixture/base.py | 268 | 268 | 97 | 4 | 42164
| sklearn/mixture/gaussian_mixture.py | 515 | 515 | 9 | 2 | 5624
| sklearn/mixture/gaussian_mixture.py | 578 | 578 | 9 | 2 | 5624


## Problem Statement

```
In Gaussian mixtures, when n_init > 1, the lower_bound_ is not always the max
#### Description
In Gaussian mixtures, when `n_init` is set to any value greater than 1, the `lower_bound_` is not the max lower bound across all initializations, but just the lower bound of the last initialization.

The bug can be fixed by adding the following line just before `return self` in `BaseMixture.fit()`:

\`\`\`python
self.lower_bound_ = max_lower_bound
\`\`\`

The test that should have caught this bug is `test_init()` in `mixture/tests/test_gaussian_mixture.py`, but it just does a single test, so it had a 50% chance of missing the issue. It should be updated to try many random states.

#### Steps/Code to Reproduce
\`\`\`python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.rand(1000, 10)
for random_state in range(100):
    gm1 = GaussianMixture(n_components=2, n_init=1, random_state=random_state).fit(X)
    gm2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state).fit(X)
    assert gm2.lower_bound_ > gm1.lower_bound_, random_state
\`\`\`

#### Expected Results
No error.

#### Actual Results
\`\`\`
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AssertionError: 4
\`\`\`

#### Versions

\`\`\`
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
\`\`\`
In Gaussian mixtures, when n_init > 1, the lower_bound_ is not always the max
#### Description
In Gaussian mixtures, when `n_init` is set to any value greater than 1, the `lower_bound_` is not the max lower bound across all initializations, but just the lower bound of the last initialization.

The bug can be fixed by adding the following line just before `return self` in `BaseMixture.fit()`:

\`\`\`python
self.lower_bound_ = max_lower_bound
\`\`\`

The test that should have caught this bug is `test_init()` in `mixture/tests/test_gaussian_mixture.py`, but it just does a single test, so it had a 50% chance of missing the issue. It should be updated to try many random states.

#### Steps/Code to Reproduce
\`\`\`python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.rand(1000, 10)
for random_state in range(100):
    gm1 = GaussianMixture(n_components=2, n_init=1, random_state=random_state).fit(X)
    gm2 = GaussianMixture(n_components=2, n_init=10, random_state=random_state).fit(X)
    assert gm2.lower_bound_ > gm1.lower_bound_, random_state
\`\`\`

#### Expected Results
No error.

#### Actual Results
\`\`\`
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AssertionError: 4
\`\`\`

#### Versions

\`\`\`
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
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/mixture/bayesian_mixture.py | 702 | 749| 470 | 470 | 7272 | 
| 2 | **2 sklearn/mixture/gaussian_mixture.py** | 586 | 600| 197 | 667 | 13530 | 
| 3 | **2 sklearn/mixture/gaussian_mixture.py** | 602 | 623| 211 | 878 | 13530 | 
| 4 | 2 sklearn/mixture/bayesian_mixture.py | 369 | 393| 207 | 1085 | 13530 | 
| 5 | 2 sklearn/mixture/bayesian_mixture.py | 413 | 452| 419 | 1504 | 13530 | 
| 6 | **2 sklearn/mixture/gaussian_mixture.py** | 677 | 692| 148 | 1652 | 13530 | 
| 7 | **2 sklearn/mixture/gaussian_mixture.py** | 625 | 656| 309 | 1961 | 13530 | 
| 8 | 2 sklearn/mixture/bayesian_mixture.py | 66 | 307| 2354 | 4315 | 13530 | 
| **-> 9 <-** | **2 sklearn/mixture/gaussian_mixture.py** | 435 | 584| 1309 | 5624 | 13530 | 
| 10 | 2 sklearn/mixture/bayesian_mixture.py | 309 | 329| 270 | 5894 | 13530 | 
| 11 | 2 sklearn/mixture/bayesian_mixture.py | 357 | 367| 123 | 6017 | 13530 | 
| 12 | 2 sklearn/mixture/bayesian_mixture.py | 395 | 411| 169 | 6186 | 13530 | 
| 13 | **2 sklearn/mixture/gaussian_mixture.py** | 694 | 710| 196 | 6382 | 13530 | 
| 14 | 3 examples/mixture/plot_gmm_sin.py | 103 | 155| 644 | 7026 | 15104 | 
| 15 | **3 sklearn/mixture/gaussian_mixture.py** | 1 | 51| 326 | 7352 | 15104 | 
| 16 | 3 sklearn/mixture/bayesian_mixture.py | 331 | 355| 236 | 7588 | 15104 | 
| 17 | 3 sklearn/mixture/bayesian_mixture.py | 687 | 700| 177 | 7765 | 15104 | 
| 18 | **4 sklearn/mixture/base.py** | 124 | 158| 261 | 8026 | 18882 | 
| 19 | 4 sklearn/mixture/bayesian_mixture.py | 751 | 786| 363 | 8389 | 18882 | 
| 20 | **4 sklearn/mixture/base.py** | 88 | 122| 289 | 8678 | 18882 | 
| 21 | 5 examples/mixture/plot_gmm_selection.py | 1 | 77| 657 | 9335 | 19773 | 
| 22 | 5 sklearn/mixture/bayesian_mixture.py | 674 | 685| 154 | 9489 | 19773 | 
| 23 | **5 sklearn/mixture/gaussian_mixture.py** | 712 | 724| 171 | 9660 | 19773 | 
| 24 | 5 sklearn/mixture/bayesian_mixture.py | 454 | 468| 125 | 9785 | 19773 | 
| 25 | 5 sklearn/mixture/bayesian_mixture.py | 649 | 672| 219 | 10004 | 19773 | 
| 26 | 5 sklearn/mixture/bayesian_mixture.py | 488 | 500| 119 | 10123 | 19773 | 
| 27 | 6 examples/mixture/plot_gmm.py | 68 | 89| 235 | 10358 | 20606 | 
| 28 | **6 sklearn/mixture/base.py** | 389 | 436| 389 | 10747 | 20606 | 
| 29 | **6 sklearn/mixture/gaussian_mixture.py** | 726 | 754| 181 | 10928 | 20606 | 
| 30 | **6 sklearn/mixture/base.py** | 512 | 530| 235 | 11163 | 20606 | 
| 31 | 7 examples/mixture/plot_concentration_prior.py | 1 | 40| 331 | 11494 | 22068 | 
| 32 | 8 sklearn/naive_bayes.py | 429 | 444| 169 | 11663 | 30477 | 
| 33 | 8 sklearn/mixture/bayesian_mixture.py | 1 | 18| 142 | 11805 | 30477 | 
| 34 | 8 examples/mixture/plot_concentration_prior.py | 87 | 137| 571 | 12376 | 30477 | 
| 35 | 9 examples/mixture/plot_gmm_pdf.py | 1 | 51| 398 | 12774 | 30875 | 
| 36 | 10 examples/gaussian_process/plot_gpr_noisy.py | 1 | 67| 747 | 13521 | 32004 | 
| 37 | 11 examples/mixture/plot_gmm_covariances.py | 68 | 135| 599 | 14120 | 33221 | 
| 38 | **11 sklearn/mixture/gaussian_mixture.py** | 98 | 137| 318 | 14438 | 33221 | 
| 39 | 12 sklearn/utils/estimator_checks.py | 1082 | 1150| 601 | 15039 | 53266 | 
| 40 | 13 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 1 | 52| 391 | 15430 | 54325 | 
| 41 | 13 examples/mixture/plot_gmm.py | 1 | 36| 289 | 15719 | 54325 | 
| 42 | **13 sklearn/mixture/gaussian_mixture.py** | 54 | 74| 133 | 15852 | 54325 | 
| 43 | **13 sklearn/mixture/gaussian_mixture.py** | 658 | 675| 174 | 16026 | 54325 | 
| 44 | 13 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 16532 | 54325 | 
| 45 | 13 sklearn/mixture/bayesian_mixture.py | 470 | 486| 166 | 16698 | 54325 | 
| 46 | 13 examples/mixture/plot_gmm_sin.py | 1 | 54| 464 | 17162 | 54325 | 
| 47 | 14 sklearn/gaussian_process/kernels.py | 93 | 116| 291 | 17453 | 69614 | 
| 48 | 14 sklearn/naive_bayes.py | 311 | 399| 802 | 18255 | 69614 | 
| 49 | 15 sklearn/ensemble/gradient_boosting.py | 810 | 893| 806 | 19061 | 87892 | 
| 50 | 16 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 123| 542 | 19603 | 88885 | 
| 51 | 17 benchmarks/bench_glmnet.py | 47 | 129| 796 | 20399 | 89972 | 
| 52 | 18 sklearn/linear_model/stochastic_gradient.py | 90 | 149| 708 | 21107 | 103946 | 
| 53 | 18 sklearn/naive_bayes.py | 401 | 427| 246 | 21353 | 103946 | 
| 54 | 19 benchmarks/bench_plot_nmf.py | 231 | 278| 530 | 21883 | 107859 | 
| 55 | 20 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 23197 | 109200 | 
| 56 | 20 sklearn/naive_bayes.py | 107 | 166| 542 | 23739 | 109200 | 
| 57 | **20 sklearn/mixture/gaussian_mixture.py** | 77 | 95| 180 | 23919 | 109200 | 
| 58 | 20 sklearn/utils/estimator_checks.py | 1226 | 1284| 600 | 24519 | 109200 | 
| 59 | 21 benchmarks/bench_mnist.py | 85 | 106| 314 | 24833 | 110931 | 
| 60 | 21 examples/gaussian_process/plot_gpr_noisy.py | 68 | 98| 353 | 25186 | 110931 | 
| 61 | 22 examples/gaussian_process/plot_gpr_noisy_targets.py | 1 | 100| 757 | 25943 | 111907 | 
| 62 | 22 examples/mixture/plot_gmm_covariances.py | 1 | 44| 282 | 26225 | 111907 | 
| 63 | **22 sklearn/mixture/base.py** | 67 | 86| 170 | 26395 | 111907 | 
| 64 | 22 sklearn/utils/estimator_checks.py | 1354 | 1451| 1021 | 27416 | 111907 | 
| 65 | 22 sklearn/mixture/bayesian_mixture.py | 502 | 525| 239 | 27655 | 111907 | 
| 66 | 23 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 28166 | 112440 | 
| 67 | 24 sklearn/gaussian_process/gpr.py | 247 | 259| 170 | 28336 | 116802 | 
| 68 | 24 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 28850 | 116802 | 
| 69 | 24 sklearn/mixture/bayesian_mixture.py | 561 | 589| 282 | 29132 | 116802 | 
| 70 | 24 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 327 | 29459 | 116802 | 
| 71 | 25 benchmarks/bench_plot_incremental_pca.py | 1 | 32| 176 | 29635 | 118158 | 
| 72 | 26 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 950 | 30585 | 122541 | 
| 73 | **26 sklearn/mixture/gaussian_mixture.py** | 381 | 432| 553 | 31138 | 122541 | 
| 74 | 26 benchmarks/bench_plot_nmf.py | 152 | 193| 476 | 31614 | 122541 | 
| 75 | 27 sklearn/mixture/__init__.py | 1 | 11| 0 | 31614 | 122601 | 
| 76 | 28 examples/gaussian_process/plot_gpc.py | 1 | 77| 751 | 32365 | 123637 | 
| 77 | 28 sklearn/mixture/bayesian_mixture.py | 620 | 647| 260 | 32625 | 123637 | 
| 78 | 28 sklearn/utils/estimator_checks.py | 1454 | 1516| 511 | 33136 | 123637 | 
| 79 | **28 sklearn/mixture/base.py** | 41 | 64| 208 | 33344 | 123637 | 
| 80 | 28 sklearn/utils/estimator_checks.py | 1785 | 1827| 449 | 33793 | 123637 | 
| 81 | 29 sklearn/utils/fixes.py | 1 | 96| 646 | 34439 | 126179 | 
| 82 | 29 sklearn/mixture/bayesian_mixture.py | 527 | 559| 316 | 34755 | 126179 | 
| 83 | 30 sklearn/cluster/k_means_.py | 107 | 142| 330 | 35085 | 140875 | 
| 84 | 30 benchmarks/bench_plot_nmf.py | 106 | 149| 401 | 35486 | 140875 | 
| 85 | **30 sklearn/mixture/base.py** | 290 | 314| 150 | 35636 | 140875 | 
| 86 | 30 sklearn/linear_model/stochastic_gradient.py | 190 | 249| 552 | 36188 | 140875 | 
| 87 | 31 benchmarks/bench_sample_without_replacement.py | 38 | 208| 1233 | 37421 | 142296 | 
| 88 | 31 sklearn/utils/estimator_checks.py | 731 | 766| 343 | 37764 | 142296 | 
| 89 | 31 sklearn/mixture/bayesian_mixture.py | 591 | 618| 270 | 38034 | 142296 | 
| 90 | 31 examples/mixture/plot_gmm.py | 39 | 65| 309 | 38343 | 142296 | 
| 91 | 31 sklearn/gaussian_process/gpr.py | 136 | 157| 207 | 38550 | 142296 | 
| 92 | 32 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 39335 | 143968 | 
| 93 | 32 sklearn/utils/estimator_checks.py | 2014 | 2085| 578 | 39913 | 143968 | 
| 94 | **32 sklearn/mixture/gaussian_mixture.py** | 339 | 378| 320 | 40233 | 143968 | 
| 95 | 32 sklearn/ensemble/gradient_boosting.py | 1017 | 1076| 579 | 40812 | 143968 | 
| 96 | 32 sklearn/cluster/k_means_.py | 1507 | 1584| 730 | 41542 | 143968 | 
| **-> 97 <-** | **32 sklearn/mixture/base.py** | 194 | 269| 622 | 42164 | 143968 | 
| 98 | 33 examples/cluster/plot_cluster_iris.py | 1 | 93| 751 | 42915 | 144727 | 
| 99 | 33 sklearn/utils/estimator_checks.py | 831 | 846| 139 | 43054 | 144727 | 
| 100 | 33 sklearn/gaussian_process/gpr.py | 459 | 474| 168 | 43222 | 144727 | 
| 101 | 33 examples/mixture/plot_gmm_selection.py | 78 | 99| 234 | 43456 | 144727 | 
| 102 | 34 sklearn/linear_model/least_angle.py | 207 | 365| 1584 | 45040 | 158387 | 
| 103 | **34 sklearn/mixture/base.py** | 477 | 510| 294 | 45334 | 158387 | 
| 104 | 35 benchmarks/bench_covertype.py | 100 | 110| 151 | 45485 | 160290 | 
| 105 | 36 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 91| 772 | 46257 | 161538 | 
| 106 | 37 examples/gaussian_process/plot_gpr_prior_posterior.py | 36 | 79| 500 | 46757 | 162369 | 
| 107 | 38 sklearn/manifold/mds.py | 234 | 276| 433 | 47190 | 166196 | 
| 108 | 38 sklearn/naive_bayes.py | 484 | 564| 765 | 47955 | 166196 | 
| 109 | 38 sklearn/linear_model/stochastic_gradient.py | 1259 | 1557| 661 | 48616 | 166196 | 
| 110 | 39 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 49388 | 167360 | 
| 111 | 39 benchmarks/bench_plot_nmf.py | 1 | 47| 298 | 49686 | 167360 | 
| 112 | 40 sklearn/neighbors/lof.py | 138 | 152| 174 | 49860 | 170680 | 
| 113 | 40 sklearn/linear_model/stochastic_gradient.py | 492 | 537| 416 | 50276 | 170680 | 
| **-> 114 <-** | **40 sklearn/mixture/base.py** | 160 | 192| 246 | 50522 | 170680 | 
| 115 | 40 sklearn/linear_model/stochastic_gradient.py | 539 | 580| 389 | 50911 | 170680 | 
| 116 | 40 sklearn/linear_model/stochastic_gradient.py | 1104 | 1132| 298 | 51209 | 170680 | 
| 117 | 41 sklearn/gaussian_process/gpc.py | 563 | 575| 148 | 51357 | 177752 | 
| 118 | 41 sklearn/naive_bayes.py | 269 | 309| 324 | 51681 | 177752 | 
| 119 | 41 sklearn/gaussian_process/gpc.py | 148 | 244| 851 | 52532 | 177752 | 
| 120 | **41 sklearn/mixture/gaussian_mixture.py** | 140 | 169| 236 | 52768 | 177752 | 
| 121 | 42 sklearn/covariance/robust_covariance.py | 95 | 175| 755 | 53523 | 184611 | 


### Hint

```


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


## Code snippets

### 1 - sklearn/mixture/bayesian_mixture.py:

Start line: 702, End line: 749

```python
class BayesianGaussianMixture(BaseMixture):

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to decrease at
        each iteration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        n_features, = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = (_compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        if self.covariance_type == 'tied':
            log_wishart = self.n_components * np.float64(_log_wishart_norm(
                self.degrees_of_freedom_, log_det_precisions_chol, n_features))
        else:
            log_wishart = np.sum(_log_wishart_norm(
                self.degrees_of_freedom_, log_det_precisions_chol, n_features))

        if self.weight_concentration_prior_type == 'dirichlet_process':
            log_norm_weight = -np.sum(betaln(self.weight_concentration_[0],
                                             self.weight_concentration_[1]))
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (-np.sum(np.exp(log_resp) * log_resp) -
                log_wishart - log_norm_weight -
                0.5 * n_features * np.sum(np.log(self.mean_precision_)))
```
### 2 - sklearn/mixture/gaussian_mixture.py:

Start line: 586, End line: 600

```python
class GaussianMixture(BaseMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
```
### 3 - sklearn/mixture/gaussian_mixture.py:

Start line: 602, End line: 623

```python
class GaussianMixture(BaseMixture):

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)
```
### 4 - sklearn/mixture/bayesian_mixture.py:

Start line: 369, End line: 393

```python
class BayesianGaussianMixture(BaseMixture):

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.
        elif self.mean_precision_prior > 0.:
            self.mean_precision_prior_ = self.mean_precision_prior
        else:
            raise ValueError("The parameter 'mean_precision_prior' should be "
                             "greater than 0., but got %.3f."
                             % self.mean_precision_prior)

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(self.mean_prior,
                                           dtype=[np.float64, np.float32],
                                           ensure_2d=False)
            _check_shape(self.mean_prior_, (n_features, ), 'means')
```
### 5 - sklearn/mixture/bayesian_mixture.py:

Start line: 413, End line: 452

```python
class BayesianGaussianMixture(BaseMixture):

    def _checkcovariance_prior_parameter(self, X):
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                'full': np.atleast_2d(np.cov(X.T)),
                'tied': np.atleast_2d(np.cov(X.T)),
                'diag': np.var(X, axis=0, ddof=1),
                'spherical': np.var(X, axis=0, ddof=1).mean()
            }[self.covariance_type]

        elif self.covariance_type in ['full', 'tied']:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features, n_features),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_matrix(self.covariance_prior_,
                                    self.covariance_type)
        elif self.covariance_type == 'diag':
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features,),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_positivity(self.covariance_prior_,
                                        self.covariance_type)
        # spherical case
        elif self.covariance_prior > 0.:
            self.covariance_prior_ = self.covariance_prior
        else:
            raise ValueError("The parameter 'spherical covariance_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.covariance_prior)
```
### 6 - sklearn/mixture/gaussian_mixture.py:

Start line: 677, End line: 692

```python
class GaussianMixture(BaseMixture):

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)
```
### 7 - sklearn/mixture/gaussian_mixture.py:

Start line: 625, End line: 656

```python
class GaussianMixture(BaseMixture):

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init
```
### 8 - sklearn/mixture/bayesian_mixture.py:

Start line: 66, End line: 307

```python
class BayesianGaussianMixture(BaseMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of::

            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).

    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float | None, optional.
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed. Smaller
        values concentrate the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it's set to 1.

    mean_prior : array-like, shape (n_features,), optional
        The prior on the mean distribution (Gaussian).
        If it is None, it's set to the mean of X.

    degrees_of_freedom_prior : float | None, optional.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, optional
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like, shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed.
        Smaller values concentrate the means of each clusters around
        `mean_prior`.

    mean_precision_ : array-like, shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    means_prior_ : array-like, shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like, shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <http://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
    """
```
### 9 - sklearn/mixture/gaussian_mixture.py:

Start line: 435, End line: 584

```python
class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
```
### 10 - sklearn/mixture/bayesian_mixture.py:

Start line: 309, End line: 329

```python
class BayesianGaussianMixture(BaseMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        super(BayesianGaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
```
### 13 - sklearn/mixture/gaussian_mixture.py:

Start line: 694, End line: 710

```python
class GaussianMixture(BaseMixture):

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2
```
### 15 - sklearn/mixture/gaussian_mixture.py:

Start line: 1, End line: 51

```python
"""Gaussian Mixture Model."""

import numpy as np

from scipy import linalg

from .base import BaseMixture, _check_shape
from ..externals.six.moves import zip
from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..utils.extmath import row_norms


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights
```
### 18 - sklearn/mixture/base.py:

Start line: 124, End line: 158

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)
```
### 20 - sklearn/mixture/base.py:

Start line: 88, End line: 122

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        # Check all the parameters values of the derived class
        self._check_parameters(X)
```
### 23 - sklearn/mixture/gaussian_mixture.py:

Start line: 712, End line: 724

```python
class GaussianMixture(BaseMixture):

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)
```
### 28 - sklearn/mixture/base.py:

Start line: 389, End line: 436

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components))

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == 'full':
            X = np.vstack([
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp)])
        elif self.covariance_type == "tied":
            X = np.vstack([
                rng.multivariate_normal(mean, self.covariances_, int(sample))
                for (mean, sample) in zip(
                    self.means_, n_samples_comp)])
        else:
            X = np.vstack([
                mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp)])

        y = np.concatenate([j * np.ones(sample, dtype=int)
                           for j, sample in enumerate(n_samples_comp)])

        return (X, y)
```
### 29 - sklearn/mixture/gaussian_mixture.py:

Start line: 726, End line: 754

```python
class GaussianMixture(BaseMixture):

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
```
### 30 - sklearn/mixture/base.py:

Start line: 512, End line: 530

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t ll change %.5f" % (
                    n_iter, cur_time - self._iter_prev_time, diff_ll))
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" %
                  (self.converged_, time() - self._init_prev_time, ll))
```
### 38 - sklearn/mixture/gaussian_mixture.py:

Start line: 98, End line: 137

```python
def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions
```
### 42 - sklearn/mixture/gaussian_mixture.py:

Start line: 54, End line: 74

```python
def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means
```
### 43 - sklearn/mixture/gaussian_mixture.py:

Start line: 658, End line: 675

```python
class GaussianMixture(BaseMixture):

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)
```
### 57 - sklearn/mixture/gaussian_mixture.py:

Start line: 77, End line: 95

```python
def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)
```
### 63 - sklearn/mixture/base.py:

Start line: 67, End line: 86

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components, tol, reg_covar,
                 max_iter, n_init, init_params, random_state, warm_start,
                 verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
```
### 73 - sklearn/mixture/gaussian_mixture.py:

Start line: 381, End line: 432

```python
def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
```
### 79 - sklearn/mixture/base.py:

Start line: 41, End line: 64

```python
def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X
```
### 85 - sklearn/mixture/base.py:

Start line: 290, End line: 314

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    @abstractmethod
    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass
```
### 94 - sklearn/mixture/gaussian_mixture.py:

Start line: 339, End line: 378

```python
###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol
```
### 97 - sklearn/mixture/base.py:

Start line: 194, End line: 269

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
        predicts the most probable label for the input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = self.lower_bound_

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                self.lower_bound_ = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = self.lower_bound_ - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(self.lower_bound_)

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return log_resp.argmax(axis=1)
```
### 103 - sklearn/mixture/base.py:

Start line: 477, End line: 510

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time
```
### 114 - sklearn/mixture/base.py:

Start line: 160, End line: 192

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    @abstractmethod
    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        pass

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fits the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self.fit_predict(X, y)
        return self
```
### 120 - sklearn/mixture/gaussian_mixture.py:

Start line: 140, End line: 169

```python
###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances
```
