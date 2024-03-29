# scikit-learn__scikit-learn-3840

| **scikit-learn/scikit-learn** | `6b5440a9964480ccb0fe1b59ab516d9228186571` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 939 |
| **Any found context length** | 939 |
| **Avg pos** | 8.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/metrics/ranking.py b/sklearn/metrics/ranking.py
--- a/sklearn/metrics/ranking.py
+++ b/sklearn/metrics/ranking.py
@@ -217,7 +217,8 @@ def _binary_uninterpolated_average_precision(
                                  sample_weight=sample_weight)
 
 
-def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
+def roc_auc_score(y_true, y_score, average="macro", sample_weight=None,
+                  max_fpr=None):
     """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
     from prediction scores.
 
@@ -257,6 +258,10 @@ def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
     sample_weight : array-like of shape = [n_samples], optional
         Sample weights.
 
+    max_fpr : float > 0 and <= 1, optional
+        If not ``None``, the standardized partial AUC [3]_ over the range
+        [0, max_fpr] is returned.
+
     Returns
     -------
     auc : float
@@ -269,6 +274,9 @@ def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
     .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
            Letters, 2006, 27(8):861-874.
 
+    .. [3] `Analyzing a portion of the ROC curve. McClish, 1989
+            <http://www.ncbi.nlm.nih.gov/pubmed/2668680>`_
+
     See also
     --------
     average_precision_score : Area under the precision-recall curve
@@ -292,7 +300,25 @@ def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
 
         fpr, tpr, tresholds = roc_curve(y_true, y_score,
                                         sample_weight=sample_weight)
-        return auc(fpr, tpr)
+        if max_fpr is None or max_fpr == 1:
+            return auc(fpr, tpr)
+        if max_fpr <= 0 or max_fpr > 1:
+            raise ValueError("Expected max_frp in range ]0, 1], got: %r"
+                             % max_fpr)
+
+        # Add a single point at max_fpr by linear interpolation
+        stop = np.searchsorted(fpr, max_fpr, 'right')
+        x_interp = [fpr[stop - 1], fpr[stop]]
+        y_interp = [tpr[stop - 1], tpr[stop]]
+        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
+        fpr = np.append(fpr[:stop], max_fpr)
+        partial_auc = auc(fpr, tpr)
+
+        # McClish correction: standardize result to be 0.5 if non-discriminant
+        # and 1 if maximal
+        min_area = 0.5 * max_fpr**2
+        max_area = max_fpr
+        return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))
 
     y_type = type_of_target(y_true)
     if y_type == "binary":

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/metrics/ranking.py | 220 | 220 | 2 | 2 | 939
| sklearn/metrics/ranking.py | 260 | 260 | 2 | 2 | 939
| sklearn/metrics/ranking.py | 272 | 272 | 2 | 2 | 939
| sklearn/metrics/ranking.py | 295 | 295 | 2 | 2 | 939


## Problem Statement

```
partial AUC
I suggest adding partial AUC to the metrics. this would compute the area under the curve up to a specified FPR (in the case of the ROC curve). this measure is important for comparing classifiers in cases where FPR is much more important than TPR. The partial AUC should also allow applying the McClish correction. see here: http://cran.r-project.org/web/packages/pROC/pROC.pdf


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/model_selection/plot_roc_crossval.py | 89 | 106| 194 | 194 | 939 | 
| **-> 2 <-** | **2 sklearn/metrics/ranking.py** | 220 | 304| 745 | 939 | 8379 | 
| 3 | 3 examples/model_selection/plot_roc.py | 94 | 149| 482 | 1421 | 9631 | 
| 4 | **3 sklearn/metrics/ranking.py** | 39 | 125| 807 | 2228 | 9631 | 
| 5 | 3 examples/model_selection/plot_roc.py | 1 | 93| 770 | 2998 | 9631 | 
| 6 | **3 sklearn/metrics/ranking.py** | 564 | 608| 526 | 3524 | 9631 | 
| 7 | **3 sklearn/metrics/ranking.py** | 482 | 563| 745 | 4269 | 9631 | 
| 8 | 3 examples/model_selection/plot_roc_crossval.py | 1 | 88| 745 | 5014 | 9631 | 
| 9 | 4 examples/model_selection/plot_precision_recall.py | 1 | 99| 970 | 5984 | 11986 | 
| 10 | 5 benchmarks/bench_lof.py | 36 | 107| 650 | 6634 | 12902 | 
| 11 | 6 examples/ensemble/plot_feature_transformation.py | 92 | 122| 336 | 6970 | 14025 | 
| 12 | 7 sklearn/metrics/scorer.py | 469 | 533| 744 | 7714 | 18899 | 
| 13 | 7 sklearn/metrics/scorer.py | 534 | 570| 413 | 8127 | 18899 | 
| 14 | 8 sklearn/metrics/__init__.py | 67 | 124| 367 | 8494 | 19753 | 
| 15 | 9 sklearn/feature_selection/univariate_selection.py | 511 | 559| 397 | 8891 | 25856 | 
| 16 | 9 sklearn/metrics/__init__.py | 1 | 65| 486 | 9377 | 25856 | 
| 17 | 10 benchmarks/bench_multilabel_metrics.py | 1 | 37| 239 | 9616 | 27507 | 
| 18 | 10 benchmarks/bench_lof.py | 1 | 35| 266 | 9882 | 27507 | 
| 19 | 11 sklearn/metrics/classification.py | 1021 | 1059| 426 | 10308 | 45885 | 
| 20 | 11 examples/model_selection/plot_precision_recall.py | 206 | 272| 612 | 10920 | 45885 | 
| 21 | 11 examples/model_selection/plot_precision_recall.py | 100 | 205| 772 | 11692 | 45885 | 
| 22 | 12 sklearn/metrics/pairwise.py | 1310 | 1320| 107 | 11799 | 57417 | 
| 23 | 12 sklearn/metrics/classification.py | 1061 | 1133| 680 | 12479 | 57417 | 
| 24 | **12 sklearn/metrics/ranking.py** | 397 | 479| 785 | 13264 | 57417 | 
| 25 | 13 benchmarks/bench_isolation_forest.py | 54 | 160| 1025 | 14289 | 58878 | 
| 26 | **13 sklearn/metrics/ranking.py** | 128 | 205| 691 | 14980 | 58878 | 
| 27 | 13 sklearn/metrics/classification.py | 1134 | 1167| 389 | 15369 | 58878 | 
| 28 | 14 examples/applications/plot_out_of_core_classification.py | 268 | 360| 807 | 16176 | 62187 | 
| 29 | 15 sklearn/utils/multiclass.py | 403 | 449| 441 | 16617 | 66053 | 
| 30 | 15 sklearn/metrics/classification.py | 884 | 1020| 1550 | 18167 | 66053 | 
| 31 | 16 examples/model_selection/plot_multi_metric_evaluation.py | 1 | 71| 541 | 18708 | 66932 | 
| 32 | 17 sklearn/metrics/cluster/supervised.py | 196 | 214| 257 | 18965 | 74842 | 
| 33 | 18 sklearn/multiclass.py | 219 | 273| 509 | 19474 | 81330 | 
| 34 | 19 examples/linear_model/plot_lasso_model_selection.py | 92 | 156| 503 | 19977 | 82659 | 
| 35 | 19 sklearn/metrics/cluster/supervised.py | 687 | 704| 232 | 20209 | 82659 | 
| 36 | 20 benchmarks/bench_plot_incremental_pca.py | 109 | 157| 485 | 20694 | 84117 | 
| 37 | 20 benchmarks/bench_multilabel_metrics.py | 136 | 191| 543 | 21237 | 84117 | 
| 38 | 21 examples/cluster/plot_adjusted_for_chance_measures.py | 1 | 31| 193 | 21430 | 85110 | 
| 39 | 21 sklearn/metrics/cluster/supervised.py | 110 | 195| 754 | 22184 | 85110 | 
| 40 | 21 examples/model_selection/plot_multi_metric_evaluation.py | 73 | 98| 317 | 22501 | 85110 | 
| 41 | 22 examples/preprocessing/plot_scaling_importance.py | 83 | 132| 436 | 22937 | 86316 | 
| 42 | 22 sklearn/metrics/pairwise.py | 1133 | 1138| 131 | 23068 | 86316 | 
| 43 | 23 examples/decomposition/plot_pca_vs_fa_model_selection.py | 88 | 126| 437 | 23505 | 87417 | 
| 44 | 23 sklearn/feature_selection/univariate_selection.py | 562 | 621| 495 | 24000 | 87417 | 
| 45 | 24 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 24759 | 88900 | 
| 46 | 25 examples/ensemble/plot_partial_dependence.py | 62 | 116| 496 | 25255 | 89951 | 
| 47 | 25 sklearn/metrics/classification.py | 720 | 832| 1143 | 26398 | 89951 | 
| 48 | **25 sklearn/metrics/ranking.py** | 206 | 217| 151 | 26549 | 89951 | 
| 49 | 25 benchmarks/bench_plot_incremental_pca.py | 86 | 106| 229 | 26778 | 89951 | 
| 50 | 26 examples/feature_selection/plot_f_test_vs_mi.py | 1 | 50| 440 | 27218 | 90391 | 
| 51 | 26 sklearn/metrics/cluster/supervised.py | 787 | 859| 751 | 27969 | 90391 | 
| 52 | 26 sklearn/metrics/scorer.py | 111 | 150| 297 | 28266 | 90391 | 
| 53 | 27 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 93| 777 | 29043 | 91380 | 
| 54 | 28 sklearn/ensemble/partial_dependence.py | 242 | 322| 767 | 29810 | 95129 | 
| 55 | 28 sklearn/metrics/scorer.py | 1 | 43| 318 | 30128 | 95129 | 
| 56 | 28 sklearn/utils/multiclass.py | 294 | 326| 297 | 30425 | 95129 | 
| 57 | 28 examples/decomposition/plot_pca_vs_fa_model_selection.py | 1 | 85| 641 | 31066 | 95129 | 
| 58 | 28 sklearn/metrics/classification.py | 1170 | 1267| 994 | 32060 | 95129 | 
| 59 | 28 sklearn/ensemble/partial_dependence.py | 344 | 396| 669 | 32729 | 95129 | 
| 60 | 28 examples/ensemble/plot_partial_dependence.py | 1 | 59| 554 | 33283 | 95129 | 
| 61 | 29 examples/gaussian_process/plot_gpc.py | 1 | 77| 751 | 34034 | 96165 | 
| 62 | 29 benchmarks/bench_isolation_forest.py | 1 | 29| 213 | 34247 | 96165 | 
| 63 | 29 sklearn/metrics/scorer.py | 343 | 400| 538 | 34785 | 96165 | 
| 64 | 29 sklearn/metrics/scorer.py | 153 | 212| 445 | 35230 | 96165 | 
| 65 | 29 sklearn/metrics/classification.py | 835 | 1167| 413 | 35643 | 96165 | 
| 66 | 29 sklearn/multiclass.py | 84 | 107| 184 | 35827 | 96165 | 
| 67 | 29 sklearn/metrics/classification.py | 618 | 717| 1047 | 36874 | 96165 | 
| 68 | 29 sklearn/multiclass.py | 515 | 565| 425 | 37299 | 96165 | 
| 69 | 30 examples/calibration/plot_calibration.py | 1 | 82| 764 | 38063 | 97401 | 
| 70 | 31 benchmarks/bench_covertype.py | 100 | 110| 151 | 38214 | 99304 | 
| 71 | 32 examples/svm/plot_rbf_parameters.py | 157 | 198| 453 | 38667 | 101253 | 
| 72 | **32 sklearn/metrics/ranking.py** | 611 | 684| 652 | 39319 | 101253 | 
| 73 | 33 examples/feature_selection/plot_permutation_test_for_classification.py | 1 | 70| 498 | 39817 | 101777 | 
| 74 | 34 examples/decomposition/plot_incremental_pca.py | 1 | 61| 485 | 40302 | 102278 | 
| 75 | 35 examples/text/plot_document_classification_20newsgroups.py | 250 | 328| 673 | 40975 | 104835 | 
| 76 | **35 sklearn/metrics/ranking.py** | 307 | 394| 816 | 41791 | 104835 | 
| 77 | 35 examples/linear_model/plot_lasso_model_selection.py | 1 | 78| 691 | 42482 | 104835 | 
| 78 | 35 benchmarks/bench_isolation_forest.py | 44 | 53| 108 | 42590 | 104835 | 
| 79 | 36 benchmarks/bench_rcv1_logreg_convergence.py | 95 | 121| 218 | 42808 | 106783 | 
| 80 | 37 examples/classification/plot_lda.py | 41 | 72| 325 | 43133 | 107374 | 
| 81 | 37 sklearn/metrics/classification.py | 1270 | 1365| 971 | 44104 | 107374 | 
| 82 | 38 sklearn/multioutput.py | 44 | 60| 123 | 44227 | 112856 | 
| 83 | 38 sklearn/ensemble/partial_dependence.py | 323 | 343| 277 | 44504 | 112856 | 
| 84 | 38 benchmarks/bench_plot_incremental_pca.py | 1 | 32| 180 | 44684 | 112856 | 
| 85 | 39 benchmarks/bench_plot_randomized_svd.py | 297 | 340| 517 | 45201 | 117240 | 
| 86 | 40 sklearn/utils/estimator_checks.py | 1138 | 1161| 185 | 45386 | 136570 | 
| 87 | 41 sklearn/learning_curve.py | 240 | 258| 238 | 45624 | 139978 | 
| 88 | 42 examples/ensemble/plot_random_forest_regression_multioutput.py | 1 | 77| 633 | 46257 | 140632 | 
| 89 | 42 sklearn/ensemble/partial_dependence.py | 136 | 163| 311 | 46568 | 140632 | 
| 90 | 43 benchmarks/bench_plot_nmf.py | 332 | 368| 377 | 46945 | 144545 | 
| 91 | 44 examples/exercises/plot_cv_diabetes.py | 1 | 81| 668 | 47613 | 145213 | 
| 92 | 45 examples/linear_model/plot_ard.py | 1 | 101| 768 | 48381 | 146136 | 
| 93 | 46 examples/model_selection/plot_nested_cross_validation_iris.py | 74 | 118| 433 | 48814 | 147210 | 
| 94 | 47 sklearn/covariance/graph_lasso_.py | 29 | 42| 152 | 48966 | 155394 | 
| 95 | 48 examples/calibration/plot_compare_calibration.py | 1 | 78| 759 | 49725 | 156575 | 
| 96 | 48 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 123| 542 | 50267 | 156575 | 
| 97 | **48 sklearn/metrics/ranking.py** | 687 | 743| 512 | 50779 | 156575 | 
| 98 | 48 examples/applications/plot_out_of_core_classification.py | 188 | 215| 234 | 51013 | 156575 | 
| 99 | 49 examples/classification/plot_classifier_comparison.py | 1 | 78| 627 | 51640 | 157921 | 
| 100 | 49 sklearn/metrics/cluster/supervised.py | 530 | 608| 800 | 52440 | 157921 | 
| 101 | 49 sklearn/metrics/classification.py | 375 | 459| 726 | 53166 | 157921 | 
| 102 | 50 examples/neural_networks/plot_mlp_alpha.py | 1 | 55| 414 | 53580 | 159026 | 
| 103 | 51 sklearn/metrics/regression.py | 1 | 40| 143 | 53723 | 164012 | 
| 104 | 51 benchmarks/bench_plot_incremental_pca.py | 48 | 58| 147 | 53870 | 164012 | 
| 105 | 51 sklearn/utils/estimator_checks.py | 1272 | 1379| 1122 | 54992 | 164012 | 
| 106 | 51 benchmarks/bench_multilabel_metrics.py | 40 | 95| 497 | 55489 | 164012 | 
| 107 | **51 sklearn/metrics/ranking.py** | 1 | 36| 172 | 55661 | 164012 | 
| 108 | 51 benchmarks/bench_covertype.py | 113 | 191| 757 | 56418 | 164012 | 
| 109 | 51 sklearn/metrics/classification.py | 1368 | 1426| 447 | 56865 | 164012 | 
| 110 | 52 examples/plot_kernel_ridge_regression.py | 155 | 172| 194 | 57059 | 165739 | 
| 111 | 53 benchmarks/bench_mnist.py | 85 | 106| 314 | 57373 | 167470 | 
| 112 | 54 sklearn/model_selection/_validation.py | 1219 | 1237| 238 | 57611 | 179344 | 
| 113 | 54 sklearn/metrics/regression.py | 533 | 576| 432 | 58043 | 179344 | 
| 114 | 54 examples/calibration/plot_compare_calibration.py | 79 | 123| 396 | 58439 | 179344 | 
| 115 | 54 sklearn/metrics/cluster/supervised.py | 365 | 432| 561 | 59000 | 179344 | 
| 116 | 54 benchmarks/bench_multilabel_metrics.py | 114 | 133| 190 | 59190 | 179344 | 
| 117 | 55 examples/gaussian_process/plot_compare_gpr_krr.py | 76 | 122| 502 | 59692 | 180637 | 
| 118 | 55 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 60206 | 180637 | 
| 119 | 56 examples/calibration/plot_calibration_multiclass.py | 122 | 169| 589 | 60795 | 182770 | 
| 120 | 57 benchmarks/bench_plot_omp_lars.py | 1 | 101| 858 | 61653 | 183936 | 
| 121 | 57 benchmarks/bench_rcv1_logreg_convergence.py | 124 | 139| 148 | 61801 | 183936 | 
| 122 | 58 examples/plot_anomaly_comparison.py | 1 | 82| 716 | 62517 | 185050 | 
| 123 | 58 examples/neural_networks/plot_mlp_alpha.py | 56 | 115| 671 | 63188 | 185050 | 
| 124 | 58 benchmarks/bench_rcv1_logreg_convergence.py | 67 | 92| 201 | 63389 | 185050 | 
| 125 | 58 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 951 | 64340 | 185050 | 
| 126 | 58 sklearn/metrics/classification.py | 295 | 356| 666 | 65006 | 185050 | 
| 127 | 59 examples/plot_missing_values.py | 1 | 74| 626 | 65632 | 185676 | 
| 128 | 59 sklearn/metrics/classification.py | 1 | 41| 190 | 65822 | 185676 | 
| 129 | 59 examples/applications/plot_out_of_core_classification.py | 361 | 416| 458 | 66280 | 185676 | 
| 130 | 59 sklearn/metrics/classification.py | 462 | 535| 764 | 67044 | 185676 | 
| 131 | 60 examples/feature_selection/plot_feature_selection.py | 1 | 87| 616 | 67660 | 186292 | 
| 132 | 61 sklearn/covariance/robust_covariance.py | 382 | 408| 365 | 68025 | 193151 | 
| 133 | 61 sklearn/metrics/pairwise.py | 1283 | 1307| 230 | 68255 | 193151 | 
| 134 | 61 sklearn/ensemble/partial_dependence.py | 72 | 134| 717 | 68972 | 193151 | 
| 135 | 61 sklearn/feature_selection/univariate_selection.py | 121 | 149| 223 | 69195 | 193151 | 
| 136 | 61 sklearn/metrics/classification.py | 358 | 372| 182 | 69377 | 193151 | 
| 137 | 61 benchmarks/bench_plot_incremental_pca.py | 35 | 45| 146 | 69523 | 193151 | 
| 138 | 62 examples/plot_multilabel.py | 97 | 115| 185 | 69708 | 194255 | 
| 139 | 62 sklearn/metrics/classification.py | 1518 | 1538| 206 | 69914 | 194255 | 
| 140 | 63 examples/gaussian_process/plot_gpc_iris.py | 1 | 64| 636 | 70550 | 194891 | 
| 141 | 64 sklearn/calibration.py | 355 | 394| 325 | 70875 | 199549 | 
| 142 | 64 sklearn/metrics/pairwise.py | 1039 | 1050| 110 | 70985 | 199549 | 


### Hint

```
+1

+1

@arjoly @mblondel is anyone working on this right now?

Hi,

I'd like to help.

@arjoly @mblondel @MechCoder , may I take this one?

Only if it is ok for you @MechCoder .

You can take this one. :-)

sure :)

Ok! I'll try my best. :)

@eyaler 
In this pROC package, it is possible to choose between "specificity" (fpr) and "sensitivity" (tpr).
And both are determined by an interval, e.g. [0.8, 1.0]

Should I do the same?

@karane
thank you for taking this.
I am only familiar with the use case of looking at the pAUC over an interval of low fpr (high specificity)

@eyaler 
you're welcome

That's Ok. I think I got it. :)

@eyaler The partial AUC is indeed usually defined in the low FPR interval. This is also the way McClish described it in her seminal paper (in [McClish, 1989](http://www.ncbi.nlm.nih.gov/pubmed/2668680?dopt=Abstract&holding=f1000,f1000m,isrctn)). However it makes no sense to me why one wouldn't want it on the high TPR region too: at times TPR can be much more important than the FPR. See [Jiang, 1996](http://www.ncbi.nlm.nih.gov/pubmed/8939225?dopt=Abstract&holding=f1000,f1000m,isrctn) who first proposed to do this (afaik).

@karane I would advise against defining it on an interval where both bounds can be changed: it makes the implementation much more tricky, and I haven't seen any case where it is actually useful. Just define it on the TPR interval [0.0, X](and on the FPR interval [X, 1.0]) where only X can be changed. In that case the trapezoid computation is pretty straightforward, most code in the auc function in pROC is there to deal with the multiple edge cases raised by releasing the second bound too.

+1

```

## Patch

```diff
diff --git a/sklearn/metrics/ranking.py b/sklearn/metrics/ranking.py
--- a/sklearn/metrics/ranking.py
+++ b/sklearn/metrics/ranking.py
@@ -217,7 +217,8 @@ def _binary_uninterpolated_average_precision(
                                  sample_weight=sample_weight)
 
 
-def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
+def roc_auc_score(y_true, y_score, average="macro", sample_weight=None,
+                  max_fpr=None):
     """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
     from prediction scores.
 
@@ -257,6 +258,10 @@ def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
     sample_weight : array-like of shape = [n_samples], optional
         Sample weights.
 
+    max_fpr : float > 0 and <= 1, optional
+        If not ``None``, the standardized partial AUC [3]_ over the range
+        [0, max_fpr] is returned.
+
     Returns
     -------
     auc : float
@@ -269,6 +274,9 @@ def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
     .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
            Letters, 2006, 27(8):861-874.
 
+    .. [3] `Analyzing a portion of the ROC curve. McClish, 1989
+            <http://www.ncbi.nlm.nih.gov/pubmed/2668680>`_
+
     See also
     --------
     average_precision_score : Area under the precision-recall curve
@@ -292,7 +300,25 @@ def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
 
         fpr, tpr, tresholds = roc_curve(y_true, y_score,
                                         sample_weight=sample_weight)
-        return auc(fpr, tpr)
+        if max_fpr is None or max_fpr == 1:
+            return auc(fpr, tpr)
+        if max_fpr <= 0 or max_fpr > 1:
+            raise ValueError("Expected max_frp in range ]0, 1], got: %r"
+                             % max_fpr)
+
+        # Add a single point at max_fpr by linear interpolation
+        stop = np.searchsorted(fpr, max_fpr, 'right')
+        x_interp = [fpr[stop - 1], fpr[stop]]
+        y_interp = [tpr[stop - 1], tpr[stop]]
+        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
+        fpr = np.append(fpr[:stop], max_fpr)
+        partial_auc = auc(fpr, tpr)
+
+        # McClish correction: standardize result to be 0.5 if non-discriminant
+        # and 1 if maximal
+        min_area = 0.5 * max_fpr**2
+        max_area = max_fpr
+        return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))
 
     y_type = type_of_target(y_true)
     if y_type == "binary":

```

## Test Patch

```diff
diff --git a/sklearn/metrics/tests/test_common.py b/sklearn/metrics/tests/test_common.py
--- a/sklearn/metrics/tests/test_common.py
+++ b/sklearn/metrics/tests/test_common.py
@@ -163,6 +163,7 @@
     "samples_roc_auc": partial(roc_auc_score, average="samples"),
     "micro_roc_auc": partial(roc_auc_score, average="micro"),
     "macro_roc_auc": partial(roc_auc_score, average="macro"),
+    "partial_roc_auc": partial(roc_auc_score, max_fpr=0.5),
 
     "average_precision_score": average_precision_score,
     "weighted_average_precision_score":
@@ -220,6 +221,7 @@
     "weighted_roc_auc",
     "macro_roc_auc",
     "samples_roc_auc",
+    "partial_roc_auc",
 
     # with default average='binary', multiclass is prohibited
     "precision_score",
@@ -240,7 +242,7 @@
 
 # Threshold-based metrics with an "average" argument
 THRESHOLDED_METRICS_WITH_AVERAGING = [
-    "roc_auc_score", "average_precision_score",
+    "roc_auc_score", "average_precision_score", "partial_roc_auc",
 ]
 
 # Metrics with a "pos_label" argument
@@ -297,7 +299,7 @@
     "unnormalized_log_loss",
 
     "roc_auc_score", "weighted_roc_auc", "samples_roc_auc",
-    "micro_roc_auc", "macro_roc_auc",
+    "micro_roc_auc", "macro_roc_auc", "partial_roc_auc",
 
     "average_precision_score", "weighted_average_precision_score",
     "samples_average_precision_score", "micro_average_precision_score",
diff --git a/sklearn/metrics/tests/test_ranking.py b/sklearn/metrics/tests/test_ranking.py
--- a/sklearn/metrics/tests/test_ranking.py
+++ b/sklearn/metrics/tests/test_ranking.py
@@ -1,5 +1,6 @@
 from __future__ import division, print_function
 
+import pytest
 import numpy as np
 from itertools import product
 import warnings
@@ -148,6 +149,34 @@ def _average_precision_slow(y_true, y_score):
     return average_precision
 
 
+def _partial_roc_auc_score(y_true, y_predict, max_fpr):
+    """Alternative implementation to check for correctness of `roc_auc_score`
+    with `max_fpr` set.
+    """
+
+    def _partial_roc(y_true, y_predict, max_fpr):
+        fpr, tpr, _ = roc_curve(y_true, y_predict)
+        new_fpr = fpr[fpr <= max_fpr]
+        new_fpr = np.append(new_fpr, max_fpr)
+        new_tpr = tpr[fpr <= max_fpr]
+        idx_out = np.argmax(fpr > max_fpr)
+        idx_in = idx_out - 1
+        x_interp = [fpr[idx_in], fpr[idx_out]]
+        y_interp = [tpr[idx_in], tpr[idx_out]]
+        new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
+        return (new_fpr, new_tpr)
+
+    new_fpr, new_tpr = _partial_roc(y_true, y_predict, max_fpr)
+    partial_auc = auc(new_fpr, new_tpr)
+
+    # Formula (5) from McClish 1989
+    fpr1 = 0
+    fpr2 = max_fpr
+    min_area = 0.5 * (fpr2 - fpr1) * (fpr2 + fpr1)
+    max_area = fpr2 - fpr1
+    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))
+
+
 def test_roc_curve():
     # Test Area under Receiver Operating Characteristic (ROC) curve
     y_true, _, probas_pred = make_prediction(binary=True)
@@ -1052,3 +1081,28 @@ def test_ranking_loss_ties_handling():
     assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1)
     assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 1)
     assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 1)
+
+
+def test_partial_roc_auc_score():
+    # Check `roc_auc_score` for max_fpr != `None`
+    y_true = np.array([0, 0, 1, 1])
+    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1
+    assert roc_auc_score(y_true, y_true, max_fpr=0.001) == 1
+    with pytest.raises(ValueError):
+        assert roc_auc_score(y_true, y_true, max_fpr=-0.1)
+    with pytest.raises(ValueError):
+        assert roc_auc_score(y_true, y_true, max_fpr=1.1)
+    with pytest.raises(ValueError):
+        assert roc_auc_score(y_true, y_true, max_fpr=0)
+
+    y_scores = np.array([0.1,  0,  0.1, 0.01])
+    roc_auc_with_max_fpr_one = roc_auc_score(y_true, y_scores, max_fpr=1)
+    unconstrained_roc_auc = roc_auc_score(y_true, y_scores)
+    assert roc_auc_with_max_fpr_one == unconstrained_roc_auc
+    assert roc_auc_score(y_true, y_scores, max_fpr=0.3) == 0.5
+
+    y_true, y_pred, _ = make_prediction(binary=True)
+    for max_fpr in np.linspace(1e-4, 1, 5):
+        assert_almost_equal(
+            roc_auc_score(y_true, y_pred, max_fpr=max_fpr),
+            _partial_roc_auc_score(y_true, y_pred, max_fpr))

```


## Code snippets

### 1 - examples/model_selection/plot_roc_crossval.py:

Start line: 89, End line: 106

```python
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```
### 2 - sklearn/metrics/ranking.py:

Start line: 220, End line: 304

```python
def roc_auc_score(y_true, y_score, average="macro", sample_weight=None):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task in label indicator format.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    auc : float

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    See also
    --------
    average_precision_score : Area under the precision-recall curve

    roc_curve : Compute Receiver operating characteristic (ROC) curve

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import roc_auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> roc_auc_score(y_true, y_scores)
    0.75

    """
    def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
        if len(np.unique(y_true)) != 2:
            raise ValueError("Only one class present in y_true. ROC AUC score "
                             "is not defined in that case.")

        fpr, tpr, tresholds = roc_curve(y_true, y_score,
                                        sample_weight=sample_weight)
        return auc(fpr, tpr)

    y_type = type_of_target(y_true)
    if y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, labels)[:, 0]

    return _average_binary_score(
        _binary_roc_auc_score, y_true, y_score, average,
        sample_weight=sample_weight)
```
### 3 - examples/model_selection/plot_roc.py:

Start line: 94, End line: 149

```python
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
```
### 4 - sklearn/metrics/ranking.py:

Start line: 39, End line: 125

```python
def auc(x, y, reorder='deprecated'):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default='deprecated')
        Whether to sort x before computing. If False, assume that x must be
        either monotonic increasing or monotonic decreasing. If True, y is
        used to break ties when sorting x. Make sure that y has a monotonic
        relation to x when setting reorder to True.

        .. deprecated:: 0.20
           Parameter ``reorder`` has been deprecated in version 0.20 and will
           be removed in 0.22. It's introduced for roc_auc_score (not for
           general use) and is no longer used there. What's more, the result
           from auc will be significantly influenced if x is sorted
           unexpectedly due to slight floating point error (See issue #9786).
           Future (and default) behavior is equivalent to ``reorder=False``.

    Returns
    -------
    auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve
    average_precision_score : Compute average precision from prediction scores
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    check_consistent_length(x, y)
    x = column_or_1d(x)
    y = column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    if reorder != 'deprecated':
        warnings.warn("The 'reorder' parameter has been deprecated in "
                      "version 0.20 and will be removed in 0.22. It is "
                      "recommended not to set 'reorder' and ensure that x "
                      "is monotonic increasing or monotonic decreasing.",
                      DeprecationWarning)

    direction = 1
    if reorder is True:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("x is neither increasing nor decreasing "
                                 ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area
```
### 5 - examples/model_selection/plot_roc.py:

Start line: 1, End line: 93

```python
"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

Multiclass settings
-------------------

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-class classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
```
### 6 - sklearn/metrics/ranking.py:

Start line: 564, End line: 608

```python
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
```
### 7 - sklearn/metrics/ranking.py:

Start line: 482, End line: 563

```python
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0. ,  0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 1.8 ,  0.8 ,  0.4 ,  0.35,  0.1 ])

    """
    # ... other code
```
### 8 - examples/model_selection/plot_roc_crossval.py:

Start line: 1, End line: 88

```python
"""
=============================================================
Receiver Operating Characteristic (ROC) with cross validation
=============================================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality using cross-validation.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

This example shows the ROC response of different datasets, created from K-fold
cross-validation. Taking all of these curves, it is possible to calculate the
mean area under curve, and see the variance of the curve when the
training set is split into different subsets. This roughly shows how the
classifier output is affected by changes in the training data, and how
different the splits generated by K-fold cross-validation are from one another.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :func:`sklearn.model_selection.cross_val_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc.py`,

"""
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Data IO and generation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
```
### 9 - examples/model_selection/plot_precision_recall.py:

Start line: 1, End line: 99

```python
"""
================
Precision-Recall
================

Example of Precision-Recall metric to evaluate classifier output quality.

Precision-Recall is a useful measure of success of prediction when the
classes are very imbalanced. In information retrieval, precision is a
measure of result relevancy, while recall is a measure of how many truly
relevant results are returned.

The precision-recall curve shows the tradeoff between precision and
recall for different threshold. A high area under the curve represents
both high recall and high precision, where high precision relates to a
low false positive rate, and high recall relates to a low false negative
rate. High scores for both show that the classifier is returning accurate
results (high precision), as well as returning a majority of all positive
results (high recall).

A system with high recall but low precision returns many results, but most of
its predicted labels are incorrect when compared to the training labels. A
system with high precision but low recall is just the opposite, returning very
few results, but most of its predicted labels are correct when compared to the
training labels. An ideal system with high precision and high recall will
return many results, with all results labeled correctly.

Precision (:math:`P`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false positives
(:math:`F_p`).

:math:`P = \\frac{T_p}{T_p+F_p}`

Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false negatives
(:math:`F_n`).

:math:`R = \\frac{T_p}{T_p + F_n}`

These quantities are also related to the (:math:`F_1`) score, which is defined
as the harmonic mean of precision and recall.

:math:`F1 = 2\\frac{P \\times R}{P+R}`

Note that the precision may not decrease with recall. The
definition of precision (:math:`\\frac{T_p}{T_p + F_p}`) shows that lowering
the threshold of a classifier may increase the denominator, by increasing the
number of results returned. If the threshold was previously set too high, the
new results may all be true positives, which will increase precision. If the
previous threshold was about right or too low, further lowering the threshold
will introduce false positives, decreasing precision.

Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, where :math:`T_p+F_n` does
not depend on the classifier threshold. This means that lowering the classifier
threshold may increase recall, by increasing the number of true positive
results. It is also possible that lowering the threshold may leave recall
unchanged, while the precision fluctuates.

The relationship between recall and precision can be observed in the
stairstep area of the plot - at the edges of these steps a small change
in the threshold considerably reduces precision, with only a minor gain in
recall.

**Average precision** (AP) summarizes such a plot as the weighted mean of
precisions achieved at each threshold, with the increase in recall from the
previous threshold used as the weight:

:math:`\\text{AP} = \\sum_n (R_n - R_{n-1}) P_n`

where :math:`P_n` and :math:`R_n` are the precision and recall at the
nth threshold. A pair :math:`(R_k, P_k)` is referred to as an
*operating point*.

AP and the trapezoidal area under the operating points
(:func:`sklearn.metrics.auc`) are common ways to summarize a precision-recall
curve that lead to different results. Read more in the
:ref:`User Guide <precision_recall_f_measure_metrics>`.

Precision-recall curves are typically used in binary classification to study
the output of a classifier. In order to extend the precision-recall curve and
average precision to multi-class or multi-label classification, it is necessary
to binarize the output. One curve can be drawn per label, but one can also draw
a precision-recall curve by considering each element of the label indicator
matrix as a binary prediction (micro-averaging).

.. note::

    See also :func:`sklearn.metrics.average_precision_score`,
             :func:`sklearn.metrics.recall_score`,
             :func:`sklearn.metrics.precision_score`,
             :func:`sklearn.metrics.f1_score`
"""
from __future__ import print_function

###############################################################################
# In binary classification settings
# --------------------------------------------------------
#
# Create simple data
```
### 10 - benchmarks/bench_lof.py:

Start line: 36, End line: 107

```python
for dataset_name in datasets:
    # loading and vectorization
    print('loading data')
    if dataset_name in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dataset_name, percent10=True,
                                 random_state=random_state)
        X = dataset.data
        y = dataset.target

    if dataset_name == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dataset_name == 'forestcover':
        dataset = fetch_covtype()
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dataset_name == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'http' or dataset_name == 'smtp':
        y = (y != b'normal.').astype(int)

    X = X.astype(float)

    print('LocalOutlierFactor processing...')
    model = LocalOutlierFactor(n_neighbors=20)
    tstart = time()
    model.fit(X)
    fit_time = time() - tstart
    scoring = -model.negative_outlier_factor_  # the lower, the more normal
    fpr, tpr, thresholds = roc_curve(y, scoring)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1,
             label=('ROC for %s (area = %0.3f, train-time: %0.2fs)'
                    % (dataset_name, AUC, fit_time)))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
```
### 24 - sklearn/metrics/ranking.py:

Start line: 397, End line: 479

```python
def precision_recall_curve(y_true, probas_pred, pos_label=None,
                           sample_weight=None):
    """Compute precision-recall pairs for different probability thresholds

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    y axis.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification in range {-1, 1} or {0, 1}.

    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.

    See also
    --------
    average_precision_score : Compute average precision from prediction scores

    roc_curve : Compute Receiver operating characteristic (ROC) curve

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  # doctest: +ELLIPSIS
    array([ 0.66...,  0.5       ,  1.        ,  1.        ])
    >>> recall
    array([ 1. ,  0.5,  0.5,  0. ])
    >>> thresholds
    array([ 0.35,  0.4 ,  0.8 ])

    """
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
```
### 26 - sklearn/metrics/ranking.py:

Start line: 128, End line: 205

```python
def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None):
    """Compute average precision (AP) from prediction scores

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels (either {0, 1} or {-1, 1}).

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    average_precision : float

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <http://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)  # doctest: +ELLIPSIS
    0.83...

    """
    # ... other code
```
### 48 - sklearn/metrics/ranking.py:

Start line: 206, End line: 217

```python
def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None):
    def _binary_uninterpolated_average_precision(
            y_true, y_score, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    return _average_binary_score(_binary_uninterpolated_average_precision,
                                 y_true, y_score, average,
                                 sample_weight=sample_weight)
```
### 72 - sklearn/metrics/ranking.py:

Start line: 611, End line: 684

```python
def label_ranking_average_precision_score(y_true, y_score):
    """Compute ranking-based average precision

    Label ranking average precision (LRAP) is the average over each ground
    truth label assigned to each sample, of the ratio of true vs. total
    labels with lower score.

    This metric is used in multilabel ranking problem, where the goal
    is to give better rank to the labels associated to each sample.

    The obtained score is always strictly greater than 0 and
    the best value is 1.

    Read more in the :ref:`User Guide <label_ranking_average_precision>`.

    Parameters
    ----------
    y_true : array or sparse matrix, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.

    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import label_ranking_average_precision_score
    >>> y_true = np.array([[1, 0, 0], [0, 0, 1]])
    >>> y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    >>> label_ranking_average_precision_score(y_true, y_score) \
        # doctest: +ELLIPSIS
    0.416...

    """
    check_consistent_length(y_true, y_score)
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # Handle badly formatted array and the degenerate case with one label
    y_type = type_of_target(y_true)
    if (y_type != "multilabel-indicator" and
            not (y_type == "binary" and y_true.ndim == 2)):
        raise ValueError("{0} format is not supported".format(y_type))

    y_true = csr_matrix(y_true)
    y_score = -y_score

    n_samples, n_labels = y_true.shape

    out = 0.
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        relevant = y_true.indices[start:stop]

        if (relevant.size == 0 or relevant.size == n_labels):
            # If all labels are relevant or unrelevant, the score is also
            # equal to 1. The label ranking has no meaning.
            out += 1.
            continue

        scores_i = y_score[i]
        rank = rankdata(scores_i, 'max')[relevant]
        L = rankdata(scores_i[relevant], 'max')
        out += (L / rank).mean()

    return out / n_samples
```
### 76 - sklearn/metrics/ranking.py:

Start line: 307, End line: 394

```python
def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presense of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]
```
### 97 - sklearn/metrics/ranking.py:

Start line: 687, End line: 743

```python
def coverage_error(y_true, y_score, sample_weight=None):
    """Coverage error measure

    Compute how far we need to go through the ranked scores to cover all
    true labels. The best value is equal to the average number
    of labels in ``y_true`` per sample.

    Ties in ``y_scores`` are broken by giving maximal rank that would have
    been assigned to all tied values.

    Note: Our implementation's score is 1 greater than the one given in
    Tsoumakas et al., 2010. This extends it to handle the degenerate case
    in which an instance has 0 true labels.

    Read more in the :ref:`User Guide <coverage_error>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.

    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    coverage_error : float

    References
    ----------
    .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
           Mining multi-label data. In Data mining and knowledge discovery
           handbook (pp. 667-685). Springer US.

    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)

    y_type = type_of_target(y_true)
    if y_type != "multilabel-indicator":
        raise ValueError("{0} format is not supported".format(y_type))

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
    coverage = (y_score >= y_min_relevant).sum(axis=1)
    coverage = coverage.filled(0)

    return np.average(coverage, weights=sample_weight)
```
### 107 - sklearn/metrics/ranking.py:

Start line: 1, End line: 36

```python
"""Metrics to assess performance on classification task given scores

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from __future__ import division

import warnings
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from ..utils import assert_all_finite
from ..utils import check_consistent_length
from ..utils import column_or_1d, check_array
from ..utils.multiclass import type_of_target
from ..utils.extmath import stable_cumsum
from ..utils.sparsefuncs import count_nonzero
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize

from .base import _average_binary_score
```
