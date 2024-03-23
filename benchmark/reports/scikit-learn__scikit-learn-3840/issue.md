# scikit-learn__scikit-learn-3840

* repo: scikit-learn/scikit-learn
* base_commit: `6b5440a9964480ccb0fe1b59ab516d9228186571`

## Problem statement

partial AUC
I suggest adding partial AUC to the metrics. this would compute the area under the curve up to a specified FPR (in the case of the ROC curve). this measure is important for comparing classifiers in cases where FPR is much more important than TPR. The partial AUC should also allow applying the McClish correction. see here: http://cran.r-project.org/web/packages/pROC/pROC.pdf



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
