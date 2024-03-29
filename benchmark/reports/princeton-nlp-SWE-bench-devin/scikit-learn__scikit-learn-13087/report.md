# scikit-learn__scikit-learn-13087

| **scikit-learn/scikit-learn** | `a73260db9c0b63d582ef4a7f3c696b68058c1c43` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3204 |
| **Any found context length** | 3204 |
| **Avg pos** | 15.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 4 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -519,7 +519,8 @@ def predict(self, T):
         return expit(-(self.a_ * T + self.b_))
 
 
-def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
+def calibration_curve(y_true, y_prob, normalize=False, n_bins=5,
+                      strategy='uniform'):
     """Compute true and predicted probabilities for a calibration curve.
 
      The method assumes the inputs come from a binary classifier.
@@ -546,6 +547,14 @@ def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
         points (i.e. without corresponding values in y_prob) will not be
         returned, thus there may be fewer than n_bins in the return value.
 
+    strategy : {'uniform', 'quantile'}, (default='uniform')
+        Strategy used to define the widths of the bins.
+
+        uniform
+            All bins have identical widths.
+        quantile
+            All bins have the same number of points.
+
     Returns
     -------
     prob_true : array, shape (n_bins,) or smaller
@@ -572,7 +581,16 @@ def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
 
     y_true = _check_binary_probabilistic_predictions(y_true, y_prob)
 
-    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
+    if strategy == 'quantile':  # Determine bin edges by distribution of data
+        quantiles = np.linspace(0, 1, n_bins + 1)
+        bins = np.percentile(y_prob, quantiles * 100)
+        bins[-1] = bins[-1] + 1e-8
+    elif strategy == 'uniform':
+        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
+    else:
+        raise ValueError("Invalid entry to 'strategy' input. Strategy "
+                         "must be either 'quantile' or 'uniform'.")
+
     binids = np.digitize(y_prob, bins) - 1
 
     bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/calibration.py | 522 | 522 | 5 | 4 | 3204
| sklearn/calibration.py | 549 | 549 | 5 | 4 | 3204
| sklearn/calibration.py | 575 | 575 | 5 | 4 | 3204


## Problem Statement

```
Feature request: support for arbitrary bin spacing in calibration.calibration_curve
#### Description
I was using [`sklearn.calibration.calibration_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html), and it currently accepts an `n_bins` parameter to specify the number of bins to evenly partition the probability space between 0 and 1.

However, I am using this in combination with a gradient boosting model in which the probabilities are very uncalibrated, and most of the predictions are close to 0. When I use the calibrated classifier, the result is very noisy because there are many data points in some bins and few, if any, in others (see example below).

In the code below, I made a work-around to do what I want and show a plot of my output (in semilog space because of the skewed distribution). I haven't contributed to a large open-source project before, but if there's agreement this would be a useful feature, I would be happy to try to draft up a PR.

#### My work-around
\`\`\`python
import numpy as np

def my_calibration_curve(y_true, y_prob, my_bins):
    prob_true = []
    prob_pred = []
    for i in range(len(my_bins) - 1):
        idx_use = np.logical_and(y_prob < my_bins[i+1], y_prob >= my_bins[i])
        prob_true.append(y_true[idx_use].mean())
        prob_pred.append(y_pred[idx_use].mean())
    return prob_true, prob_pred

# example bins:
# my_bins = np.concatenate([[0], np.logspace(-3, 0, 10)])
\`\`\`

#### Results comparison
Notice the large disparity in results between the different bins chosen. For this reason, I think the user should be able to choose the bin edges, as in numpy's or matplotlib's [histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) functions.

![image](https://user-images.githubusercontent.com/7298871/52183657-d1e18c80-27be-11e9-9c84-011c043e0978.png)


#### Versions

\`\`\`
Darwin-18.0.0-x86_64-i386-64bit
Python 3.6.4 |Anaconda custom (x86_64)| (default, Jan 16 2018, 12:04:33) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]
NumPy 1.15.1
SciPy 1.1.0
Scikit-Learn 0.19.1
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/calibration/plot_calibration.py | 1 | 82| 764 | 764 | 1241 | 
| 2 | 2 examples/calibration/plot_calibration_curve.py | 1 | 69| 664 | 1428 | 2618 | 
| 3 | 3 examples/calibration/plot_compare_calibration.py | 1 | 77| 755 | 2183 | 3807 | 
| 4 | 3 examples/calibration/plot_calibration.py | 84 | 120| 401 | 2584 | 3807 | 
| **-> 5 <-** | **4 sklearn/calibration.py** | 522 | 587| 620 | 3204 | 8521 | 
| 6 | 5 examples/calibration/plot_calibration_multiclass.py | 122 | 169| 588 | 3792 | 10652 | 
| 7 | 5 examples/calibration/plot_compare_calibration.py | 78 | 123| 407 | 4199 | 10652 | 
| 8 | 5 examples/calibration/plot_calibration_multiclass.py | 1 | 80| 764 | 4963 | 10652 | 
| 9 | 5 examples/calibration/plot_calibration_multiclass.py | 81 | 121| 752 | 5715 | 10652 | 
| 10 | 6 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 6508 | 12498 | 
| 11 | 7 sklearn/preprocessing/_discretization.py | 191 | 223| 345 | 6853 | 15074 | 
| 12 | 7 sklearn/preprocessing/_discretization.py | 1 | 109| 936 | 7789 | 15074 | 
| 13 | 8 examples/preprocessing/plot_discretization.py | 1 | 87| 779 | 8568 | 15883 | 
| 14 | 9 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 9352 | 18991 | 
| 15 | 10 examples/preprocessing/plot_discretization_strategies.py | 1 | 96| 787 | 10139 | 19795 | 
| 16 | 11 examples/ensemble/plot_partial_dependence.py | 61 | 115| 496 | 10635 | 20838 | 
| 17 | 11 examples/calibration/plot_calibration_curve.py | 72 | 135| 668 | 11303 | 20838 | 
| 18 | 11 examples/preprocessing/plot_all_scaling.py | 149 | 180| 320 | 11623 | 20838 | 
| 19 | 12 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 12377 | 22057 | 
| 20 | 12 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 13192 | 22057 | 
| 21 | 13 examples/plot_anomaly_comparison.py | 81 | 153| 763 | 13955 | 23581 | 
| 22 | 13 sklearn/preprocessing/_discretization.py | 111 | 189| 708 | 14663 | 23581 | 
| 23 | **13 sklearn/calibration.py** | 404 | 450| 363 | 15026 | 23581 | 
| 24 | 14 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 15819 | 24826 | 
| 25 | **14 sklearn/calibration.py** | 31 | 108| 762 | 16581 | 24826 | 
| 26 | 15 examples/model_selection/plot_learning_curve.py | 110 | 129| 221 | 16802 | 26080 | 
| 27 | 16 examples/plot_johnson_lindenstrauss_bound.py | 94 | 168| 709 | 17511 | 27923 | 
| 28 | 17 benchmarks/bench_plot_neighbors.py | 111 | 186| 663 | 18174 | 29360 | 
| 29 | 18 examples/classification/plot_classifier_comparison.py | 1 | 78| 633 | 18807 | 30712 | 
| 30 | 19 examples/classification/plot_classification_probability.py | 1 | 99| 795 | 19602 | 31532 | 
| 31 | 20 examples/datasets/plot_random_dataset.py | 1 | 68| 645 | 20247 | 32177 | 
| 32 | 21 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 141| 529 | 20776 | 33307 | 
| 33 | 22 examples/ensemble/plot_voting_probas.py | 1 | 84| 788 | 21564 | 34095 | 
| 34 | **22 sklearn/calibration.py** | 452 | 465| 198 | 21762 | 34095 | 
| 35 | 23 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 772 | 22534 | 35016 | 
| 36 | 23 examples/plot_johnson_lindenstrauss_bound.py | 170 | 208| 376 | 22910 | 35016 | 
| 37 | 24 sklearn/naive_bayes.py | 907 | 927| 217 | 23127 | 43405 | 
| 38 | **24 sklearn/calibration.py** | 1 | 28| 164 | 23291 | 43405 | 
| 39 | 24 examples/preprocessing/plot_all_scaling.py | 311 | 356| 353 | 23644 | 43405 | 
| 40 | 25 examples/ensemble/plot_gradient_boosting_quantile.py | 1 | 80| 548 | 24192 | 43953 | 
| 41 | **25 sklearn/calibration.py** | 298 | 315| 144 | 24336 | 43953 | 
| 42 | 26 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 25032 | 44677 | 
| 43 | 27 examples/gaussian_process/plot_gpc.py | 1 | 77| 751 | 25783 | 45713 | 
| 44 | 28 sklearn/utils/estimator_checks.py | 1 | 78| 703 | 26486 | 67206 | 
| 45 | 29 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 26997 | 67739 | 
| 46 | 30 sklearn/ensemble/gradient_boosting.py | 1687 | 1934| 2499 | 29496 | 87986 | 
| 47 | 30 examples/preprocessing/plot_all_scaling.py | 182 | 216| 381 | 29877 | 87986 | 
| 48 | 30 benchmarks/bench_plot_neighbors.py | 26 | 109| 710 | 30587 | 87986 | 
| 49 | 31 examples/plot_kernel_approximation.py | 1 | 87| 735 | 31322 | 89927 | 
| 50 | 31 examples/preprocessing/plot_discretization_classification.py | 109 | 193| 874 | 32196 | 89927 | 
| 51 | 32 examples/gaussian_process/plot_gpc_isoprobability.py | 1 | 88| 759 | 32955 | 90857 | 
| 52 | 33 benchmarks/bench_plot_incremental_pca.py | 104 | 151| 468 | 33423 | 92213 | 
| 53 | 34 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 34195 | 93377 | 
| 54 | 35 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 34741 | 93951 | 
| 55 | 36 examples/applications/plot_species_distribution_modeling.py | 128 | 206| 796 | 35537 | 95722 | 
| 56 | **36 sklearn/calibration.py** | 362 | 401| 325 | 35862 | 95722 | 
| 57 | 37 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 94| 785 | 36647 | 96706 | 
| 58 | 38 benchmarks/bench_plot_nmf.py | 151 | 192| 476 | 37123 | 100608 | 
| 59 | 39 examples/gaussian_process/plot_gpr_noisy.py | 1 | 67| 747 | 37870 | 101737 | 
| 60 | 39 examples/gaussian_process/plot_gpc_isoprobability.py | 90 | 103| 121 | 37991 | 101737 | 
| 61 | 40 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 38567 | 102872 | 
| 62 | 41 examples/ensemble/plot_random_forest_embedding.py | 1 | 84| 757 | 39324 | 103834 | 
| 63 | 42 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 39821 | 104863 | 
| 64 | 43 examples/neighbors/plot_species_kde.py | 1 | 78| 690 | 40511 | 105904 | 
| 65 | 44 sklearn/datasets/samples_generator.py | 154 | 221| 782 | 41293 | 119908 | 
| 66 | 45 examples/ensemble/plot_adaboost_twoclass.py | 1 | 85| 705 | 41998 | 120778 | 
| 67 | 46 examples/model_selection/plot_grid_search_refit_callable.py | 78 | 117| 323 | 42321 | 121610 | 
| 68 | 47 sklearn/ensemble/partial_dependence.py | 239 | 321| 768 | 43089 | 125360 | 
| 69 | 47 sklearn/ensemble/partial_dependence.py | 322 | 342| 276 | 43365 | 125360 | 
| 70 | 48 examples/cluster/plot_cluster_comparison.py | 1 | 86| 698 | 44063 | 126945 | 
| 71 | 49 examples/classification/plot_lda.py | 38 | 69| 325 | 44388 | 127529 | 
| 72 | 49 sklearn/ensemble/partial_dependence.py | 343 | 395| 668 | 45056 | 127529 | 
| 73 | 50 examples/ensemble/plot_bias_variance.py | 116 | 192| 684 | 45740 | 129343 | 
| 74 | 51 examples/model_selection/plot_precision_recall.py | 101 | 205| 770 | 46510 | 131685 | 
| 75 | 52 benchmarks/bench_glmnet.py | 47 | 129| 796 | 47306 | 132772 | 
| 76 | 52 sklearn/preprocessing/_discretization.py | 225 | 261| 341 | 47647 | 132772 | 
| 77 | 53 examples/text/plot_document_classification_20newsgroups.py | 247 | 323| 662 | 48309 | 135311 | 
| 78 | 53 examples/ensemble/plot_gradient_boosting_oob.py | 96 | 138| 423 | 48732 | 135311 | 
| 79 | 53 benchmarks/bench_plot_nmf.py | 230 | 277| 530 | 49262 | 135311 | 
| 80 | 54 examples/datasets/plot_random_multilabel_dataset.py | 1 | 56| 432 | 49694 | 136194 | 
| 81 | 55 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 51008 | 137535 | 
| 82 | 56 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 51548 | 139059 | 
| 83 | 57 examples/neighbors/plot_kde_1d.py | 73 | 152| 790 | 52338 | 140627 | 
| 84 | **57 sklearn/calibration.py** | 109 | 200| 760 | 53098 | 140627 | 
| 85 | 57 examples/cluster/plot_cluster_comparison.py | 88 | 189| 887 | 53985 | 140627 | 
| 86 | 58 examples/preprocessing/plot_map_data_to_normal.py | 99 | 138| 477 | 54462 | 141862 | 
| 87 | 59 sklearn/linear_model/stochastic_gradient.py | 7 | 43| 346 | 54808 | 156237 | 
| 88 | 60 benchmarks/bench_tree.py | 64 | 125| 523 | 55331 | 157107 | 
| 89 | 61 examples/covariance/plot_mahalanobis_distances.py | 79 | 134| 726 | 56057 | 158770 | 
| 90 | 62 benchmarks/bench_20newsgroups.py | 1 | 97| 778 | 56835 | 159548 | 
| 91 | 63 examples/mixture/plot_gmm_sin.py | 103 | 155| 644 | 57479 | 161122 | 
| 92 | 64 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 80| 666 | 58145 | 161819 | 
| 93 | 65 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 947 | 59092 | 166199 | 
| 94 | 66 sklearn/preprocessing/label.py | 604 | 669| 531 | 59623 | 173755 | 
| 95 | 67 examples/plot_multilabel.py | 96 | 114| 185 | 59808 | 174850 | 
| 96 | 68 benchmarks/bench_isolation_forest.py | 54 | 161| 1032 | 60840 | 176318 | 
| 97 | 69 examples/svm/plot_rbf_parameters.py | 74 | 160| 732 | 61572 | 178300 | 
| 98 | 69 sklearn/ensemble/gradient_boosting.py | 2146 | 2387| 2474 | 64046 | 178300 | 
| 99 | 70 sklearn/linear_model/bayes.py | 1 | 136| 1114 | 65160 | 183904 | 
| 100 | 71 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 77| 753 | 65913 | 184948 | 
| 101 | 72 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 66698 | 186628 | 
| 102 | 73 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 67230 | 188573 | 
| 103 | 74 examples/cluster/plot_inductive_clustering.py | 59 | 121| 493 | 67723 | 189482 | 
| 104 | 75 examples/linear_model/plot_ard.py | 1 | 101| 769 | 68492 | 190406 | 
| 105 | 75 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 69051 | 190406 | 
| 106 | 75 examples/preprocessing/plot_all_scaling.py | 106 | 146| 405 | 69456 | 190406 | 
| 107 | 75 examples/classification/plot_classifier_comparison.py | 79 | 145| 710 | 70166 | 190406 | 
| 108 | 76 benchmarks/bench_mnist.py | 84 | 105| 314 | 70480 | 192124 | 
| 109 | 77 examples/ensemble/plot_adaboost_multiclass.py | 1 | 89| 748 | 71228 | 193138 | 
| 110 | 77 benchmarks/bench_plot_nmf.py | 331 | 367| 377 | 71605 | 193138 | 
| 111 | 78 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 71986 | 194453 | 
| 112 | 79 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 44 | 91| 609 | 72595 | 195381 | 
| 113 | 80 examples/bicluster/plot_spectral_biclustering.py | 1 | 63| 426 | 73021 | 195833 | 
| 114 | 81 benchmarks/bench_covertype.py | 99 | 109| 151 | 73172 | 197724 | 


### Hint

```
It actually sounds like the problem is not the number of bins, but that
bins should be constructed to reflect the distribution, rather than the
range, of the input. I think we should still use n_bins as the primary
parameter, but allow those bins to be quantile based, providing a strategy
option for discretisation (
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html
).

My only question is whether this is still true to the meaning of
"calibration curve" / "reliability curve"

Quantile bins seem a good default here...

Yup, quantile bins would have my desired effect. I just thought it would be nice to allow more flexibility by allowing a `bins` parameter, but I suppose it's not necessary.

I think this still satisfies "calibration curve." I don't see any reason that a "calibration" needs to have evenly-spaced bins. Often it's natural to do things in a log-spaced manner.
I'm happy to see a pr for quantiles here
Sweet. I'll plan to do work on it next weekend
```

## Patch

```diff
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -519,7 +519,8 @@ def predict(self, T):
         return expit(-(self.a_ * T + self.b_))
 
 
-def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
+def calibration_curve(y_true, y_prob, normalize=False, n_bins=5,
+                      strategy='uniform'):
     """Compute true and predicted probabilities for a calibration curve.
 
      The method assumes the inputs come from a binary classifier.
@@ -546,6 +547,14 @@ def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
         points (i.e. without corresponding values in y_prob) will not be
         returned, thus there may be fewer than n_bins in the return value.
 
+    strategy : {'uniform', 'quantile'}, (default='uniform')
+        Strategy used to define the widths of the bins.
+
+        uniform
+            All bins have identical widths.
+        quantile
+            All bins have the same number of points.
+
     Returns
     -------
     prob_true : array, shape (n_bins,) or smaller
@@ -572,7 +581,16 @@ def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
 
     y_true = _check_binary_probabilistic_predictions(y_true, y_prob)
 
-    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
+    if strategy == 'quantile':  # Determine bin edges by distribution of data
+        quantiles = np.linspace(0, 1, n_bins + 1)
+        bins = np.percentile(y_prob, quantiles * 100)
+        bins[-1] = bins[-1] + 1e-8
+    elif strategy == 'uniform':
+        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
+    else:
+        raise ValueError("Invalid entry to 'strategy' input. Strategy "
+                         "must be either 'quantile' or 'uniform'.")
+
     binids = np.digitize(y_prob, bins) - 1
 
     bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_calibration.py b/sklearn/tests/test_calibration.py
--- a/sklearn/tests/test_calibration.py
+++ b/sklearn/tests/test_calibration.py
@@ -259,6 +259,21 @@ def test_calibration_curve():
     assert_raises(ValueError, calibration_curve, [1.1], [-0.1],
                   normalize=False)
 
+    # test that quantiles work as expected
+    y_true2 = np.array([0, 0, 0, 0, 1, 1])
+    y_pred2 = np.array([0., 0.1, 0.2, 0.5, 0.9, 1.])
+    prob_true_quantile, prob_pred_quantile = calibration_curve(
+        y_true2, y_pred2, n_bins=2, strategy='quantile')
+
+    assert len(prob_true_quantile) == len(prob_pred_quantile)
+    assert len(prob_true_quantile) == 2
+    assert_almost_equal(prob_true_quantile, [0, 2 / 3])
+    assert_almost_equal(prob_pred_quantile, [0.1, 0.8])
+
+    # Check that error is raised when invalid strategy is selected
+    assert_raises(ValueError, calibration_curve, y_true2, y_pred2,
+                  strategy='percentile')
+
 
 def test_calibration_nan_imputer():
     """Test that calibration can accept nan"""

```


## Code snippets

### 1 - examples/calibration/plot_calibration.py:

Start line: 1, End line: 82

```python
"""
======================================
Probability calibration of classifiers
======================================

When performing classification you often want to predict not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. However, not all
classifiers provide well-calibrated probabilities, some being over-confident
while others being under-confident. Thus, a separate calibration of predicted
probabilities is often desirable as a postprocessing. This example illustrates
two different methods for this calibration and evaluates the quality of the
returned probabilities using Brier's score
(see https://en.wikipedia.org/wiki/Brier_score).

Compared are the estimated probability using a Gaussian naive Bayes classifier
without calibration, with a sigmoid calibration, and with a non-parametric
isotonic calibration. One can observe that only the non-parametric model is
able to provide a probability calibration that returns probabilities close
to the expected 0.5 for most of the samples belonging to the middle
cluster with heterogeneous labels. This results in a significantly improved
Brier score.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = \
    train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)

# Gaussian Naive-Bayes with no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(X_train, y_train, sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, y_train, sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier scores: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)
```
### 2 - examples/calibration/plot_calibration_curve.py:

Start line: 1, End line: 69

```python
"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to display
how well calibrated the predicted probabilities are and how to calibrate an
uncalibrated classifier.

The experiment is performed on an artificial dataset for binary classification
with 100,000 samples (1,000 of them are used for model fitting) with 20
features. Of the 20 features, only 2 are informative and 10 are redundant. The
first figure shows the estimated probabilities obtained with logistic
regression, Gaussian naive Bayes, and Gaussian naive Bayes with both isotonic
calibration and sigmoid calibration. The calibration performance is evaluated
with Brier score, reported in the legend (the smaller the better). One can
observe here that logistic regression is well calibrated while raw Gaussian
naive Bayes performs very badly. This is because of the redundant features
which violate the assumption of feature-independence and result in an overly
confident classifier, which is indicated by the typical transposed-sigmoid
curve.

Calibration of the probabilities of Gaussian naive Bayes with isotonic
regression can fix this issue as can be seen from the nearly diagonal
calibration curve. Sigmoid calibration also improves the brier score slightly,
albeit not as strongly as the non-parametric isotonic regression. This can be
attributed to the fact that we have plenty of calibration data such that the
greater flexibility of the non-parametric model can be exploited.

The second figure shows the calibration curve of a linear support-vector
classifier (LinearSVC). LinearSVC shows the opposite behavior as Gaussian
naive Bayes: the calibration curve has a sigmoid curve, which is typical for
an under-confident classifier. In the case of LinearSVC, this is caused by the
margin property of the hinge loss, which lets the model focus on hard samples
that are close to the decision boundary (the support vectors).

Both kinds of calibration can fix this issue and yield nearly identical
results. This shows that sigmoid calibration can deal with situations where
the calibration curve of the base classifier is sigmoid (e.g., for LinearSVC)
but not where it is transposed-sigmoid (e.g., Gaussian naive Bayes).
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split


# Create dataset of classification task with many redundant and few
# informative features
X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                    random_state=42)
```
### 3 - examples/calibration/plot_compare_calibration.py:

Start line: 1, End line: 77

```python
"""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of the predict_proba method can be directly interpreted as a confidence level.
For instance a well calibrated (binary) classifier should classify the samples
such that among the samples to which it gave a predict_proba value close to
0.8, approx. 80% actually belong to the positive class.

LogisticRegression returns well calibrated predictions as it directly
optimizes log-loss. In contrast, the other methods return biased probabilities,
with different biases per method:

* GaussianNaiveBayes tends to push probabilities to 0 or 1 (note the counts in
  the histograms). This is mainly because it makes the assumption that features
  are conditionally independent given the class, which is not the case in this
  dataset which contains 2 redundant features.

* RandomForestClassifier shows the opposite behavior: the histograms show
  peaks at approx. 0.2 and 0.9 probability, while probabilities close to 0 or 1
  are very rare. An explanation for this is given by Niculescu-Mizil and Caruana
  [1]_: "Methods such as bagging and random forests that average predictions
  from a base set of models can have difficulty making predictions near 0 and 1
  because variance in the underlying base models will bias predictions that
  should be near zero or one away from these values. Because predictions are
  restricted to the interval [0,1], errors caused by variance tend to be one-
  sided near zero and one. For example, if a model should predict p = 0 for a
  case, the only way bagging can achieve this is if all bagged trees predict
  zero. If we add noise to the trees that bagging is averaging over, this noise
  will cause some trees to predict values larger than 0 for this case, thus
  moving the average prediction of the bagged ensemble away from 0. We observe
  this effect most strongly with random forests because the base-level trees
  trained with random forests have relatively high variance due to feature
  subsetting." As a result, the calibration curve shows a characteristic
  sigmoid shape, indicating that the classifier could trust its "intuition"
  more and return probabilities closer to 0 or 1 typically.

* Support Vector Classification (SVC) shows an even more sigmoid curve as
  the  RandomForestClassifier, which is typical for maximum-margin methods
  (compare Niculescu-Mizil and Caruana [1]_), which focus on hard samples
  that are close to the decision boundary (the support vectors).

.. topic:: References:

    .. [1] Predicting Good Probabilities with Supervised Learning,
          A. Niculescu-Mizil & R. Caruana, ICML 2005
"""
print(__doc__)

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train = X[:train_samples]
X_test = X[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

# Create classifiers
```
### 4 - examples/calibration/plot_calibration.py:

Start line: 84, End line: 120

```python
clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

# #############################################################################
# Plot the data and the predicted probabilities
plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50,
                c=color[np.newaxis, :],
                alpha=0.5, edgecolor='k',
                label="Class %s" % this_y)
plt.legend(loc="best")
plt.title("Data")

plt.figure()
order = np.lexsort((prob_pos_clf, ))
plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
plt.plot(prob_pos_isotonic[order], 'g', linewidth=3,
         label='Isotonic calibration (%1.3f)' % clf_isotonic_score)
plt.plot(prob_pos_sigmoid[order], 'b', linewidth=3,
         label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)
plt.plot(np.linspace(0, y_test.size, 51)[1::2],
         y_test[order].reshape(25, -1).mean(1),
         'k', linewidth=3, label=r'Empirical')
plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability "
           "(uncalibrated GNB)")
plt.ylabel("P(y=1)")
plt.legend(loc="upper left")
plt.title("Gaussian naive Bayes probabilities")

plt.show()
```
### 5 - sklearn/calibration.py:

Start line: 522, End line: 587

```python
def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
    """Compute true and predicted probabilities for a calibration curve.

     The method assumes the inputs come from a binary classifier.

     Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.

    n_bins : int
        Number of bins. A bigger number requires more data. Bins with no data
        points (i.e. without corresponding values in y_prob) will not be
        returned, thus there may be fewer than n_bins in the return value.

    Returns
    -------
    prob_true : array, shape (n_bins,) or smaller
        The true probability in each bin (fraction of positives).

    prob_pred : array, shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred
```
### 6 - examples/calibration/plot_calibration_multiclass.py:

Start line: 122, End line: 169

```python
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(loc="best")

print("Log-loss of")
print(" * uncalibrated classifier trained on 800 datapoints: %.3f "
      % score)
print(" * classifier trained on 600 datapoints and calibrated on "
      "200 datapoint: %.3f" % sig_score)

# Illustrate calibrator
plt.figure(1)
# generate grid over 2-simplex
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

calibrated_classifier = sig_clf.calibrated_classifiers_[0]
prediction = np.vstack([calibrator.predict(this_p)
                        for calibrator, this_p in
                        zip(calibrated_classifier.calibrators_, p.T)]).T
prediction /= prediction.sum(axis=1)[:, None]

# Plot modifications of calibrator
for i in range(prediction.shape[0]):
    plt.arrow(p[i, 0], p[i, 1],
              prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
              head_width=1e-2, color=colors[np.argmax(p[i])])
# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Illustration of sigmoid calibrator")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.show()
```
### 7 - examples/calibration/plot_compare_calibration.py:

Start line: 78, End line: 123

```python
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()
```
### 8 - examples/calibration/plot_calibration_multiclass.py:

Start line: 1, End line: 80

```python
"""
==================================================
Probability Calibration for 3-class classification
==================================================

This example illustrates how sigmoid calibration changes predicted
probabilities for a 3-class classification problem. Illustrated is the
standard 2-simplex, where the three corners correspond to the three classes.
Arrows point from the probability vectors predicted by an uncalibrated
classifier to the probability vectors predicted by the same classifier after
sigmoid calibration on a hold-out validation set. Colors indicate the true
class of an instance (red: class 1, green: class 2, blue: class 3).

The base classifier is a random forest classifier with 25 base estimators
(trees). If this classifier is trained on all 800 training datapoints, it is
overly confident in its predictions and thus incurs a large log-loss.
Calibrating an identical classifier, which was trained on 600 datapoints, with
method='sigmoid' on the remaining 200 datapoints reduces the confidence of the
predictions, i.e., moves the probability vectors from the edges of the simplex
towards the center. This calibration results in a lower log-loss. Note that an
alternative would have been to increase the number of base estimators which
would have resulted in a similar decrease in log-loss.
"""
print(__doc__)


import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=1000, n_features=2, random_state=42,
                  cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)

# Train random forest classifier, calibrate on validation data and evaluate
# on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
sig_clf.fit(X_valid, y_valid)
sig_clf_probs = sig_clf.predict_proba(X_test)
sig_score = log_loss(y_test, sig_clf_probs)

# Plot changes in predicted probabilities via arrows
plt.figure(0)
colors = ["r", "g", "b"]
for i in range(clf_probs.shape[0]):
    plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
              sig_clf_probs[i, 0] - clf_probs[i, 0],
              sig_clf_probs[i, 1] - clf_probs[i, 1],
              color=colors[y_test[i]], head_width=1e-2)

# Plot perfect predictions
plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")

# Plot boundaries of unit simplex
```
### 9 - examples/calibration/plot_calibration_multiclass.py:

Start line: 81, End line: 121

```python
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

# Annotate points on the simplex
plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
             xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
             xy=(.5, .0), xytext=(.5, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
             xy=(.0, .5), xytext=(.1, .5), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
             xy=(.5, .5), xytext=(.6, .6), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $0$, $1$)',
             xy=(0, 0), xytext=(.1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($1$, $0$, $0$)',
             xy=(1, 0), xytext=(1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $1$, $0$)',
             xy=(0, 1), xytext=(.1, 1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
# Add grid
plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Change of predicted probabilities after sigmoid calibration")
plt.xlabel("Probability class 1")
```
### 10 - examples/preprocessing/plot_discretization_classification.py:

Start line: 1, End line: 89

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
Feature discretization
======================

A demonstration of feature discretization on synthetic classification datasets.
Feature discretization decomposes each feature into a set of bins, here equally
distributed in width. The discrete values are then one-hot encoded, and given
to a linear classifier. This preprocessing enables a non-linear behavior even
though the classifier is linear.

On this example, the first two rows represent linearly non-separable datasets
(moons and concentric circles) while the third is approximately linearly
separable. On the two linearly non-separable datasets, feature discretization
largely increases the performance of linear classifiers. On the linearly
separable dataset, feature discretization decreases the performance of linear
classifiers. Two non-linear classifiers are also shown for comparison.

This example should be taken with a grain of salt, as the intuition conveyed
does not necessarily carry over to real datasets. Particularly in
high-dimensional spaces, data can more easily be separated linearly. Moreover,
using feature discretization and one-hot encoding increases the number of
features, which easily lead to overfitting when the number of samples is small.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

print(__doc__)

h = .02  # step size in the mesh


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == 'Pipeline':
        name = [get_name(est[1]) for est in estimator.steps]
        name = ' + '.join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
classifiers = [
    (LogisticRegression(solver='lbfgs', random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (LinearSVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'),
        LogisticRegression(solver='lbfgs', random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'logisticregression__C': np.logspace(-2, 7, 10),
        }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'), LinearSVC(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'linearsvc__C': np.logspace(-2, 7, 10),
        }),
    (GradientBoostingClassifier(n_estimators=50, random_state=0), {
        'learning_rate': np.logspace(-4, 0, 10)
    }),
    (SVC(random_state=0, gamma='scale'), {
        'C': np.logspace(-2, 7, 10)
    }),
]
```
### 23 - sklearn/calibration.py:

Start line: 404, End line: 450

```python
def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray, shape (n_samples,)
        The targets.

    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        if sample_weight is not None:
            return (sample_weight * loss).sum()
        else:
            return loss.sum()
    # ... other code
```
### 25 - sklearn/calibration.py:

Start line: 31, End line: 108

```python
class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration with isotonic regression or sigmoid.

    See glossary entry for :term:`cross-validation estimator`.

    With this class, the base_estimator is fit on the train set of the
    cross-validation generator and the test set is used for calibration.
    The probabilities for each of the folds are then averaged
    for prediction. In case that cv="prefit" is passed to __init__,
    it is assumed that base_estimator has been fitted already and all
    data is used for calibration. Note that data for fitting the
    classifier and for calibrating it must be disjoint.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv=prefit, the
        classifier must have been fit already on data.

    method : 'sigmoid' or 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach. It is not advised to use isotonic calibration
        with too few calibration samples ``(<<1000)`` since it tends to
        overfit.
        Use sigmoids (Platt's calibration) in this case.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The class labels.

    calibrated_classifiers_ : list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each crossvalidation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
```
### 34 - sklearn/calibration.py:

Start line: 452, End line: 465

```python
def _sigmoid_calibration(df, y, sample_weight=None):
    # ... other code

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]
```
### 38 - sklearn/calibration.py:

Start line: 1, End line: 28

```python
"""Calibration of predicted probabilities."""

import warnings
from inspect import signature

from math import log
import numpy as np

from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import LabelEncoder

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from .preprocessing import label_binarize, LabelBinarizer
from .utils import check_X_y, check_array, indexable, column_or_1d
from .utils.validation import check_is_fitted, check_consistent_length
from .isotonic import IsotonicRegression
from .svm import LinearSVC
from .model_selection import check_cv
from .metrics.classification import _check_binary_probabilistic_predictions
```
### 41 - sklearn/calibration.py:

Start line: 298, End line: 315

```python
class _CalibratedClassifier(object):

    def _preproc(self, X):
        n_classes = len(self.classes_)
        if hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X)
            if df.ndim == 1:
                df = df[:, np.newaxis]
        elif hasattr(self.base_estimator, "predict_proba"):
            df = self.base_estimator.predict_proba(X)
            if n_classes == 2:
                df = df[:, 1:]
        else:
            raise RuntimeError('classifier has no decision_function or '
                               'predict_proba method.')

        idx_pos_class = self.label_encoder_.\
            transform(self.base_estimator.classes_)

        return df, idx_pos_class
```
### 56 - sklearn/calibration.py:

Start line: 362, End line: 401

```python
class _CalibratedClassifier(object):

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        df, idx_pos_class = self._preproc(X)

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, self.calibrators_):
            if n_classes == 2:
                k += 1
            proba[:, k] = calibrator.predict(this_df)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba
```
### 84 - sklearn/calibration.py:

Start line: 109, End line: 200

```python
class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, method='sigmoid', cv='warn'):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False)
        X, y = indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
                np.any([np.sum(y == class_) < n_folds for class_ in
                        self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        self.calibrated_classifiers_ = []
        if self.base_estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            base_estimator = LinearSVC(random_state=0)
        else:
            base_estimator = self.base_estimator

        if self.cv == "prefit":
            calibrated_classifier = _CalibratedClassifier(
                base_estimator, method=self.method)
            if sample_weight is not None:
                calibrated_classifier.fit(X, y, sample_weight)
            else:
                calibrated_classifier.fit(X, y)
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            cv = check_cv(self.cv, y, classifier=True)
            fit_parameters = signature(base_estimator.fit).parameters
            estimator_name = type(base_estimator).__name__
            if (sample_weight is not None
                    and "sample_weight" not in fit_parameters):
                warnings.warn("%s does not support sample_weight. Samples"
                              " weights are only used for the calibration"
                              " itself." % estimator_name)
                base_estimator_sample_weight = None
            else:
                if sample_weight is not None:
                    sample_weight = check_array(sample_weight, ensure_2d=False)
                    check_consistent_length(y, sample_weight)
                base_estimator_sample_weight = sample_weight
            for train, test in cv.split(X, y):
                this_estimator = clone(base_estimator)
                if base_estimator_sample_weight is not None:
                    this_estimator.fit(
                        X[train], y[train],
                        sample_weight=base_estimator_sample_weight[train])
                else:
                    this_estimator.fit(X[train], y[train])

                calibrated_classifier = _CalibratedClassifier(
                    this_estimator, method=self.method,
                    classes=self.classes_)
                if sample_weight is not None:
                    calibrated_classifier.fit(X[test], y[test],
                                              sample_weight[test])
                else:
                    calibrated_classifier.fit(X[test], y[test])
                self.calibrated_classifiers_.append(calibrated_classifier)

        return self
```
