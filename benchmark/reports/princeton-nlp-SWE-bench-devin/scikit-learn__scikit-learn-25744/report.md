# scikit-learn__scikit-learn-25744

| **scikit-learn/scikit-learn** | `2c867b8f822eb7a684f0d5c4359e4426e1c9cfe0` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | - |
| **Any found context length** | 25882 |
| **Avg pos** | 23.0 |
| **Min pos** | 23 |
| **Max pos** | 23 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/tree/_classes.py b/sklearn/tree/_classes.py
--- a/sklearn/tree/_classes.py
+++ b/sklearn/tree/_classes.py
@@ -99,16 +99,16 @@ class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
         "max_depth": [Interval(Integral, 1, None, closed="left"), None],
         "min_samples_split": [
             Interval(Integral, 2, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="right"),
+            Interval("real_not_int", 0.0, 1.0, closed="right"),
         ],
         "min_samples_leaf": [
             Interval(Integral, 1, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="neither"),
+            Interval("real_not_int", 0.0, 1.0, closed="neither"),
         ],
         "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
         "max_features": [
             Interval(Integral, 1, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="right"),
+            Interval("real_not_int", 0.0, 1.0, closed="right"),
             StrOptions({"auto", "sqrt", "log2"}, deprecated={"auto"}),
             None,
         ],
diff --git a/sklearn/utils/_param_validation.py b/sklearn/utils/_param_validation.py
--- a/sklearn/utils/_param_validation.py
+++ b/sklearn/utils/_param_validation.py
@@ -364,9 +364,12 @@ class Interval(_Constraint):
 
     Parameters
     ----------
-    type : {numbers.Integral, numbers.Real}
+    type : {numbers.Integral, numbers.Real, "real_not_int"}
         The set of numbers in which to set the interval.
 
+        If "real_not_int", only reals that don't have the integer type
+        are allowed. For example 1.0 is allowed but 1 is not.
+
     left : float or int or None
         The left bound of the interval. None means left bound is -∞.
 
@@ -392,14 +395,6 @@ class Interval(_Constraint):
     `[0, +∞) U {+∞}`.
     """
 
-    @validate_params(
-        {
-            "type": [type],
-            "left": [Integral, Real, None],
-            "right": [Integral, Real, None],
-            "closed": [StrOptions({"left", "right", "both", "neither"})],
-        }
-    )
     def __init__(self, type, left, right, *, closed):
         super().__init__()
         self.type = type
@@ -410,6 +405,18 @@ def __init__(self, type, left, right, *, closed):
         self._check_params()
 
     def _check_params(self):
+        if self.type not in (Integral, Real, "real_not_int"):
+            raise ValueError(
+                "type must be either numbers.Integral, numbers.Real or 'real_not_int'."
+                f" Got {self.type} instead."
+            )
+
+        if self.closed not in ("left", "right", "both", "neither"):
+            raise ValueError(
+                "closed must be either 'left', 'right', 'both' or 'neither'. "
+                f"Got {self.closed} instead."
+            )
+
         if self.type is Integral:
             suffix = "for an interval over the integers."
             if self.left is not None and not isinstance(self.left, Integral):
@@ -424,6 +431,11 @@ def _check_params(self):
                 raise ValueError(
                     f"right can't be None when closed == {self.closed} {suffix}"
                 )
+        else:
+            if self.left is not None and not isinstance(self.left, Real):
+                raise TypeError("Expecting left to be a real number.")
+            if self.right is not None and not isinstance(self.right, Real):
+                raise TypeError("Expecting right to be a real number.")
 
         if self.right is not None and self.left is not None and self.right <= self.left:
             raise ValueError(
@@ -447,8 +459,13 @@ def __contains__(self, val):
             return False
         return True
 
+    def _has_valid_type(self, val):
+        if self.type == "real_not_int":
+            return isinstance(val, Real) and not isinstance(val, Integral)
+        return isinstance(val, self.type)
+
     def is_satisfied_by(self, val):
-        if not isinstance(val, self.type):
+        if not self._has_valid_type(val):
             return False
 
         return val in self

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/tree/_classes.py | 102 | 111 | - | 1 | -
| sklearn/utils/_param_validation.py | 367 | 367 | - | 13 | -
| sklearn/utils/_param_validation.py | 395 | 402 | - | 13 | -
| sklearn/utils/_param_validation.py | 413 | 413 | 23 | 13 | 25882
| sklearn/utils/_param_validation.py | 427 | 427 | 23 | 13 | 25882
| sklearn/utils/_param_validation.py | 450 | 450 | - | 13 | -


## Problem Statement

```
Setting min_samples_split=1 in DecisionTreeClassifier does not raise exception
### Describe the bug

If `min_samples_split` is set to 1, an exception should be raised according to the paramter's constraints:

https://github.com/scikit-learn/scikit-learn/blob/e2e705021eb6c9f23f0972f119b56e37cd7567ef/sklearn/tree/_classes.py#L100-L103

However, `DecisionTreeClassifier` accepts `min_samples_split=1` without complaining.

With scikit-survival 1.0, this raises an exception as expected:
\`\`\`
ValueError: min_samples_split == 1, must be >= 2.
\`\`\`

I suspect that this has to do with the Intervals of the constraints overlapping. `min_samples_split=1` satisfies the `Real` constraint, whereas the `Integral` constraint should have precedence.

### Steps/Code to Reproduce

\`\`\`python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
t = DecisionTreeClassifier(min_samples_split=1)
t.fit(X, y)
\`\`\`

### Expected Results

\`\`\`
sklearn.utils._param_validation.InvalidParameterError: The 'min_samples_split' parameter of DecisionTreeClassifier must be an int in the range [2, inf) or a float in the range (0.0, 1.0]. Got 1 instead.
\`\`\`

### Actual Results

No exception is raised.

### Versions

\`\`\`shell
System:
    python: 3.10.8 | packaged by conda-forge | (main, Nov 22 2022, 08:26:04) [GCC 10.4.0]
executable: /…/bin/python
   machine: Linux-6.1.6-100.fc36.x86_64-x86_64-with-glibc2.35

Python dependencies:
      sklearn: 1.3.dev0
          pip: 22.2.2
   setuptools: 63.2.0
        numpy: 1.24.1
        scipy: 1.10.0
       Cython: None
       pandas: None
   matplotlib: None
       joblib: 1.2.0
threadpoolctl: 3.1.0

Built with OpenMP: True

threadpoolctl info:
       user_api: openmp
   internal_api: openmp
         prefix: libgomp
       filepath: /…/lib/libgomp.so.1.0.0
        version: None
    num_threads: 16

       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: /…/lib/python3.10/site-packages/numpy.libs/libopenblas64_p-r0-15028c96.3.21.so
        version: 0.3.21
threading_layer: pthreads
   architecture: Zen
    num_threads: 16

       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: /…/lib/python3.10/site-packages/scipy.libs/libopenblasp-r0-41284840.3.18.so
        version: 0.3.18
threading_layer: pthreads
   architecture: Zen
    num_threads: 16
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/tree/_classes.py** | 822 | 857| 262 | 262 | 15340 | 
| 2 | **1 sklearn/tree/_classes.py** | 596 | 820| 2353 | 2615 | 15340 | 
| 3 | **1 sklearn/tree/_classes.py** | 1509 | 1769| 208 | 2823 | 15340 | 
| 4 | **1 sklearn/tree/_classes.py** | 1284 | 1507| 2270 | 5093 | 15340 | 
| 5 | **1 sklearn/tree/_classes.py** | 1 | 83| 471 | 5564 | 15340 | 
| 6 | **1 sklearn/tree/_classes.py** | 176 | 263| 726 | 6290 | 15340 | 
| 7 | **1 sklearn/tree/_classes.py** | 1181 | 1216| 244 | 6534 | 15340 | 
| 8 | 2 benchmarks/bench_tree.py | 62 | 123| 522 | 7056 | 16202 | 
| 9 | **2 sklearn/tree/_classes.py** | 265 | 358| 812 | 7868 | 16202 | 
| 10 | 3 sklearn/ensemble/_hist_gradient_boosting/grower.py | 335 | 359| 185 | 8053 | 22858 | 
| 11 | 3 benchmarks/bench_tree.py | 1 | 59| 340 | 8393 | 22858 | 
| 12 | 4 sklearn/ensemble/_gb.py | 867 | 1195| 3416 | 11809 | 39083 | 
| 13 | **4 sklearn/tree/_classes.py** | 1542 | 1740| 1949 | 13758 | 39083 | 
| 14 | **4 sklearn/tree/_classes.py** | 978 | 1179| 2046 | 15804 | 39083 | 
| 15 | 5 sklearn/ensemble/_forest.py | 2090 | 2483| 427 | 16231 | 63104 | 
| 16 | 6 examples/release_highlights/plot_release_highlights_0_22_0.py | 195 | 281| 764 | 16995 | 65514 | 
| 17 | 7 sklearn/ensemble/_iforest.py | 5 | 520| 138 | 17133 | 69954 | 
| 18 | 8 benchmarks/bench_hist_gradient_boosting.py | 92 | 171| 754 | 17887 | 72381 | 
| 19 | 9 sklearn/utils/_testing.py | 890 | 932| 374 | 18261 | 80172 | 
| 20 | 10 sklearn/utils/estimator_checks.py | 1956 | 2704| 6424 | 24685 | 116528 | 
| 21 | 11 benchmarks/bench_mnist.py | 83 | 117| 323 | 25008 | 118329 | 
| 22 | 12 benchmarks/bench_hist_gradient_boosting_categorical_only.py | 1 | 81| 651 | 25659 | 118980 | 
| **-> 23 <-** | **13 sklearn/utils/_param_validation.py** | 412 | 432| 223 | 25882 | 125259 | 
| 24 | 14 examples/release_highlights/plot_release_highlights_0_24_0.py | 1 | 120| 1150 | 27032 | 127697 | 
| 25 | 15 benchmarks/bench_hist_gradient_boosting_threading.py | 87 | 139| 382 | 27414 | 130435 | 
| 26 | 15 sklearn/ensemble/_forest.py | 1395 | 1801| 425 | 27839 | 130435 | 
| 27 | **15 sklearn/tree/_classes.py** | 359 | 387| 199 | 28038 | 130435 | 
| 28 | **15 sklearn/tree/_classes.py** | 389 | 400| 146 | 28184 | 130435 | 
| 29 | 16 setup.py | 70 | 132| 676 | 28860 | 136406 | 
| 30 | 16 sklearn/ensemble/_forest.py | 1104 | 1393| 2938 | 31798 | 136406 | 
| 31 | **16 sklearn/tree/_classes.py** | 1742 | 1770| 190 | 31988 | 136406 | 
| 32 | 16 sklearn/utils/estimator_checks.py | 3433 | 3543| 957 | 32945 | 136406 | 
| 33 | 17 sklearn/model_selection/_validation.py | 346 | 378| 350 | 33295 | 152358 | 
| 34 | 17 sklearn/ensemble/_forest.py | 1805 | 2088| 2846 | 36141 | 152358 | 
| 35 | 18 examples/ensemble/plot_adaboost_multiclass.py | 1 | 91| 749 | 36890 | 153383 | 
| 36 | 18 benchmarks/bench_hist_gradient_boosting_threading.py | 1 | 66| 516 | 37406 | 153383 | 
| 37 | 18 benchmarks/bench_hist_gradient_boosting.py | 1 | 50| 397 | 37803 | 153383 | 
| 38 | 19 examples/ensemble/plot_adaboost_regression.py | 1 | 77| 619 | 38422 | 154025 | 
| 39 | 20 benchmarks/bench_covertype.py | 101 | 110| 137 | 38559 | 155943 | 
| 40 | 21 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 1 | 52| 363 | 38922 | 173146 | 
| 41 | 22 benchmarks/bench_isolation_forest.py | 54 | 165| 1035 | 39957 | 174617 | 
| 42 | 23 examples/ensemble/plot_forest_iris.py | 1 | 75| 585 | 40542 | 176103 | 
| 43 | 23 examples/release_highlights/plot_release_highlights_0_24_0.py | 211 | 265| 510 | 41052 | 176103 | 
| 44 | 24 sklearn/datasets/_samples_generator.py | 42 | 309| 392 | 41444 | 192015 | 
| 45 | 25 examples/tree/plot_tree_regression_multioutput.py | 1 | 65| 570 | 42014 | 192585 | 
| 46 | 25 examples/release_highlights/plot_release_highlights_0_22_0.py | 1 | 89| 749 | 42763 | 192585 | 


### Hint

```
I think that this is on purpose. Otherwise, we would have used `closed="neither"` for the `Real` case and `1` is qualified as a `Real`.

At least this is not a regression since the code in the past would have failed and now we allow it to be considered as 100% of the train set.

If we exclude `1` it means that we don't accept both 100% and 1. I don't know if this is something that we want.
Note that with sklearn 1.0, `min_samples_split=1.0` does not raise an exception, only `min_samples_split=1`.
Reading the docstring, I agree it is strange to interpret the integer `1` as 100%:

https://github.com/scikit-learn/scikit-learn/blob/baefe83933df9abecc2c16769d42e52b2694a9c8/sklearn/tree/_classes.py#L635-L638

From the docstring, `min_samples_split=1` is interpreted as 1 sample, which does not make any sense. 

I think we should be able to specify "1.0" but not "1" in our parameter validation framework. @jeremiedbb What do you think of having a way to reject `Integral`, such as:

\`\`\`python
Interval(Real, 0.0, 1.0, closed="right", invalid_type=Integral),
\`\`\`

If we have a way to specify a `invalid_type`, then I prefer to reject `min_samples_split=1` as we did in previous versions. 
Also note that `min_samples_split=1.0` and `min_samples_split=1` do not result in the same behavior:

https://github.com/scikit-learn/scikit-learn/blob/baefe83933df9abecc2c16769d42e52b2694a9c8/sklearn/tree/_classes.py#L257-L263

If `min_samples_split=1`, the actual `min_samples_split` is determine by `min_samples_leaf`:
\`\`\`python
min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
\`\`\`

If `min_samples_split=1.0` and assuming there are more than 2 samples in the data, `min_samples_split = n_samples`:
\`\`\`python
min_samples_split = int(ceil(self.min_samples_split * n_samples))
\`\`\`
```

## Patch

```diff
diff --git a/sklearn/tree/_classes.py b/sklearn/tree/_classes.py
--- a/sklearn/tree/_classes.py
+++ b/sklearn/tree/_classes.py
@@ -99,16 +99,16 @@ class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
         "max_depth": [Interval(Integral, 1, None, closed="left"), None],
         "min_samples_split": [
             Interval(Integral, 2, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="right"),
+            Interval("real_not_int", 0.0, 1.0, closed="right"),
         ],
         "min_samples_leaf": [
             Interval(Integral, 1, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="neither"),
+            Interval("real_not_int", 0.0, 1.0, closed="neither"),
         ],
         "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
         "max_features": [
             Interval(Integral, 1, None, closed="left"),
-            Interval(Real, 0.0, 1.0, closed="right"),
+            Interval("real_not_int", 0.0, 1.0, closed="right"),
             StrOptions({"auto", "sqrt", "log2"}, deprecated={"auto"}),
             None,
         ],
diff --git a/sklearn/utils/_param_validation.py b/sklearn/utils/_param_validation.py
--- a/sklearn/utils/_param_validation.py
+++ b/sklearn/utils/_param_validation.py
@@ -364,9 +364,12 @@ class Interval(_Constraint):
 
     Parameters
     ----------
-    type : {numbers.Integral, numbers.Real}
+    type : {numbers.Integral, numbers.Real, "real_not_int"}
         The set of numbers in which to set the interval.
 
+        If "real_not_int", only reals that don't have the integer type
+        are allowed. For example 1.0 is allowed but 1 is not.
+
     left : float or int or None
         The left bound of the interval. None means left bound is -∞.
 
@@ -392,14 +395,6 @@ class Interval(_Constraint):
     `[0, +∞) U {+∞}`.
     """
 
-    @validate_params(
-        {
-            "type": [type],
-            "left": [Integral, Real, None],
-            "right": [Integral, Real, None],
-            "closed": [StrOptions({"left", "right", "both", "neither"})],
-        }
-    )
     def __init__(self, type, left, right, *, closed):
         super().__init__()
         self.type = type
@@ -410,6 +405,18 @@ def __init__(self, type, left, right, *, closed):
         self._check_params()
 
     def _check_params(self):
+        if self.type not in (Integral, Real, "real_not_int"):
+            raise ValueError(
+                "type must be either numbers.Integral, numbers.Real or 'real_not_int'."
+                f" Got {self.type} instead."
+            )
+
+        if self.closed not in ("left", "right", "both", "neither"):
+            raise ValueError(
+                "closed must be either 'left', 'right', 'both' or 'neither'. "
+                f"Got {self.closed} instead."
+            )
+
         if self.type is Integral:
             suffix = "for an interval over the integers."
             if self.left is not None and not isinstance(self.left, Integral):
@@ -424,6 +431,11 @@ def _check_params(self):
                 raise ValueError(
                     f"right can't be None when closed == {self.closed} {suffix}"
                 )
+        else:
+            if self.left is not None and not isinstance(self.left, Real):
+                raise TypeError("Expecting left to be a real number.")
+            if self.right is not None and not isinstance(self.right, Real):
+                raise TypeError("Expecting right to be a real number.")
 
         if self.right is not None and self.left is not None and self.right <= self.left:
             raise ValueError(
@@ -447,8 +459,13 @@ def __contains__(self, val):
             return False
         return True
 
+    def _has_valid_type(self, val):
+        if self.type == "real_not_int":
+            return isinstance(val, Real) and not isinstance(val, Integral)
+        return isinstance(val, self.type)
+
     def is_satisfied_by(self, val):
-        if not isinstance(val, self.type):
+        if not self._has_valid_type(val):
             return False
 
         return val in self

```

## Test Patch

```diff
diff --git a/sklearn/tree/tests/test_tree.py b/sklearn/tree/tests/test_tree.py
--- a/sklearn/tree/tests/test_tree.py
+++ b/sklearn/tree/tests/test_tree.py
@@ -2425,3 +2425,25 @@ def test_tree_deserialization_from_read_only_buffer(tmpdir):
         clf.tree_,
         "The trees of the original and loaded classifiers are not equal.",
     )
+
+
+@pytest.mark.parametrize("Tree", ALL_TREES.values())
+def test_min_sample_split_1_error(Tree):
+    """Check that an error is raised when min_sample_split=1.
+
+    non-regression test for issue gh-25481.
+    """
+    X = np.array([[0, 0], [1, 1]])
+    y = np.array([0, 1])
+
+    # min_samples_split=1.0 is valid
+    Tree(min_samples_split=1.0).fit(X, y)
+
+    # min_samples_split=1 is invalid
+    tree = Tree(min_samples_split=1)
+    msg = (
+        r"'min_samples_split' .* must be an int in the range \[2, inf\) "
+        r"or a float in the range \(0.0, 1.0\]"
+    )
+    with pytest.raises(ValueError, match=msg):
+        tree.fit(X, y)
diff --git a/sklearn/utils/tests/test_param_validation.py b/sklearn/utils/tests/test_param_validation.py
--- a/sklearn/utils/tests/test_param_validation.py
+++ b/sklearn/utils/tests/test_param_validation.py
@@ -662,3 +662,10 @@ def fit(self, X=None, y=None):
     # does not raise, even though "b" is not in the constraints dict and "a" is not
     # a parameter of the estimator.
     ThirdPartyEstimator(b=0).fit()
+
+
+def test_interval_real_not_int():
+    """Check for the type "real_not_int" in the Interval constraint."""
+    constraint = Interval("real_not_int", 0, 1, closed="both")
+    assert constraint.is_satisfied_by(1.0)
+    assert not constraint.is_satisfied_by(1)

```


## Code snippets

### 1 - sklearn/tree/_classes.py:

Start line: 822, End line: 857

```python
class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [StrOptions({"gini", "entropy", "log_loss"}), Hidden(Criterion)],
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
    }

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
```
### 2 - sklearn/tree/_classes.py:

Start line: 596, End line: 820

```python
class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeRegressor : A decision tree regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """
```
### 3 - sklearn/tree/_classes.py:

Start line: 1509, End line: 1769

```python
class ExtraTreeClassifier(DecisionTreeClassifier):

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
        )


class ExtraTreeRegressor(DecisionTreeRegressor):
```
### 4 - sklearn/tree/_classes.py:

Start line: 1284, End line: 1507

```python
class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"auto", "sqrt", "log2"} or None, default="sqrt"
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            .. versionchanged:: 1.1
                The default of `max_features` changed from `"auto"` to `"sqrt"`.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeRegressor : An extremely randomized tree regressor.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.
    sklearn.ensemble.RandomForestClassifier : A random forest classifier.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.
    sklearn.ensemble.RandomTreesEmbedding : An ensemble of
        totally random trees.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.tree import ExtraTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, random_state=0)
    >>> extra_tree = ExtraTreeClassifier(random_state=0)
    >>> cls = BaggingClassifier(extra_tree, random_state=0).fit(
    ...    X_train, y_train)
    >>> cls.score(X_test, y_test)
    0.8947...
    """
```
### 5 - sklearn/tree/_classes.py:

Start line: 1, End line: 83

```python
"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

import numbers
import warnings
import copy
from abc import ABCMeta
from abc import abstractmethod
from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse

from ..base import BaseEstimator
from ..base import ClassifierMixin
from ..base import clone
from ..base import RegressorMixin
from ..base import is_classifier
from ..base import MultiOutputMixin
from ..utils import Bunch
from ..utils import check_random_state
from ..utils.validation import _check_sample_weight
from ..utils import compute_sample_weight
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ..utils._param_validation import Hidden, Interval, StrOptions

from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import DepthFirstTreeBuilder
from ._tree import BestFirstTreeBuilder
from ._tree import Tree
from ._tree import _build_pruned_tree_ccp
from ._tree import ccp_pruning_path
from . import _tree, _splitter, _criterion

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}
```
### 6 - sklearn/tree/_classes.py:

Start line: 176, End line: 263

```python
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._validate_params()
        random_state = check_random_state(self.random_state)

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based sparse matrices"
                    )

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        # ... other code
```
### 7 - sklearn/tree/_classes.py:

Start line: 1181, End line: 1216

```python
class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [
            StrOptions({"squared_error", "friedman_mse", "absolute_error", "poisson"}),
            Hidden(Criterion),
        ],
    }

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
```
### 8 - benchmarks/bench_tree.py:

Start line: 62, End line: 123

```python
if __name__ == "__main__":

    print("============================================")
    print("Warning: this is going to take a looong time")
    print("============================================")

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure("scikit-learn tree benchmark results")
    plt.subplot(211)
    plt.title("Learning with varying number of samples")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print("============================================")
        print("Entering iteration %s of %s" % (i, n))
        print("============================================")
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title("Learning in high dimensional spaces")
    plt.plot(xx, scikit_classifier_results, "g-", label="classification")
    plt.plot(xx, scikit_regressor_results, "r-", label="regression")
    plt.legend(loc="upper left")
    plt.xlabel("number of dimensions")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
```
### 9 - sklearn/tree/_classes.py:

Start line: 265, End line: 358

```python
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, check_input=True):
        # ... other code

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                    warnings.warn(
                        "`max_features='auto'` has been deprecated in 1.1 "
                        "and will be removed in 1.3. To keep the past behaviour, "
                        "explicitly set `max_features='sqrt'`.",
                        FutureWarning,
                    )
                else:
                    max_features = self.n_features_in_
                    warnings.warn(
                        "`max_features='auto'` has been deprecated in 1.1 "
                        "and will be removed in 1.3. To keep the past behaviour, "
                        "explicitly set `max_features=1.0'`.",
                        FutureWarning,
                    )
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )

        if is_classifier(self):
            self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        # ... other code
```
### 10 - sklearn/ensemble/_hist_gradient_boosting/grower.py:

Start line: 335, End line: 359

```python
class TreeGrower:

    def _validate_parameters(
        self,
        X_binned,
        min_gain_to_split,
        min_hessian_to_split,
    ):
        """Validate parameters passed to __init__.

        Also validate parameters passed to splitter.
        """
        if X_binned.dtype != np.uint8:
            raise NotImplementedError("X_binned must be of type uint8.")
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency."
            )
        if min_gain_to_split < 0:
            raise ValueError(
                "min_gain_to_split={} must be positive.".format(min_gain_to_split)
            )
        if min_hessian_to_split < 0:
            raise ValueError(
                "min_hessian_to_split={} must be positive.".format(min_hessian_to_split)
            )
```
### 13 - sklearn/tree/_classes.py:

Start line: 1542, End line: 1740

```python
class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"auto", "sqrt", "log2"} or None, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `1.0`.

        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    feature_importances_ : ndarray of shape (n_features,)
        Return impurity-based feature importances (the higher, the more
        important the feature).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeClassifier : An extremely randomized tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.tree import ExtraTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.33...
    """
```
### 14 - sklearn/tree/_classes.py:

Start line: 978, End line: 1179

```python
class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """
```
### 23 - sklearn/utils/_param_validation.py:

Start line: 412, End line: 432

```python
class Interval(_Constraint):

    def _check_params(self):
        if self.type is Integral:
            suffix = "for an interval over the integers."
            if self.left is not None and not isinstance(self.left, Integral):
                raise TypeError(f"Expecting left to be an int {suffix}")
            if self.right is not None and not isinstance(self.right, Integral):
                raise TypeError(f"Expecting right to be an int {suffix}")
            if self.left is None and self.closed in ("left", "both"):
                raise ValueError(
                    f"left can't be None when closed == {self.closed} {suffix}"
                )
            if self.right is None and self.closed in ("right", "both"):
                raise ValueError(
                    f"right can't be None when closed == {self.closed} {suffix}"
                )

        if self.right is not None and self.left is not None and self.right <= self.left:
            raise ValueError(
                f"right can't be less than left. Got left={self.left} and "
                f"right={self.right}"
            )
```
### 27 - sklearn/tree/_classes.py:

Start line: 359, End line: 387

```python
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, check_input=True):
        # ... other code
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self
```
### 28 - sklearn/tree/_classes.py:

Start line: 389, End line: 400

```python
class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
            if issparse(X) and (
                X.indices.dtype != np.intc or X.indptr.dtype != np.intc
            ):
                raise ValueError("No support for np.int64 index based sparse matrices")
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)
        return X
```
### 31 - sklearn/tree/_classes.py:

Start line: 1742, End line: 1770

```python
class ExtraTreeRegressor(DecisionTreeRegressor):

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        random_state=None,
        min_impurity_decrease=0.0,
        max_leaf_nodes=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
        )
```
