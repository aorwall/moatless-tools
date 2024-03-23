# scikit-learn__scikit-learn-13087

* repo: scikit-learn/scikit-learn
* base_commit: `a73260db9c0b63d582ef4a7f3c696b68058c1c43`

## Problem statement

Feature request: support for arbitrary bin spacing in calibration.calibration_curve
#### Description
I was using [`sklearn.calibration.calibration_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html), and it currently accepts an `n_bins` parameter to specify the number of bins to evenly partition the probability space between 0 and 1.

However, I am using this in combination with a gradient boosting model in which the probabilities are very uncalibrated, and most of the predictions are close to 0. When I use the calibrated classifier, the result is very noisy because there are many data points in some bins and few, if any, in others (see example below).

In the code below, I made a work-around to do what I want and show a plot of my output (in semilog space because of the skewed distribution). I haven't contributed to a large open-source project before, but if there's agreement this would be a useful feature, I would be happy to try to draft up a PR.

#### My work-around
```python
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
```

#### Results comparison
Notice the large disparity in results between the different bins chosen. For this reason, I think the user should be able to choose the bin edges, as in numpy's or matplotlib's [histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) functions.

![image](https://user-images.githubusercontent.com/7298871/52183657-d1e18c80-27be-11e9-9c84-011c043e0978.png)


#### Versions

```
Darwin-18.0.0-x86_64-i386-64bit
Python 3.6.4 |Anaconda custom (x86_64)| (default, Jan 16 2018, 12:04:33) 
[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]
NumPy 1.15.1
SciPy 1.1.0
Scikit-Learn 0.19.1
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
