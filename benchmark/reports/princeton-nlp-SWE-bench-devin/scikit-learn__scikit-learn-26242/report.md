# scikit-learn__scikit-learn-26242

| **scikit-learn/scikit-learn** | `b747bacfa1d706bf3c52680566590bfaf0d74363` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 820 |
| **Any found context length** | 820 |
| **Avg pos** | 7.0 |
| **Min pos** | 4 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/ensemble/_base.py b/sklearn/ensemble/_base.py
--- a/sklearn/ensemble/_base.py
+++ b/sklearn/ensemble/_base.py
@@ -157,7 +157,7 @@ def _validate_estimator(self, default=None):
 
         if self.estimator is not None:
             self.estimator_ = self.estimator
-        elif self.base_estimator not in [None, "deprecated"]:
+        elif self.base_estimator != "deprecated":
             warnings.warn(
                 (
                     "`base_estimator` was renamed to `estimator` in version 1.2 and "
@@ -165,7 +165,10 @@ def _validate_estimator(self, default=None):
                 ),
                 FutureWarning,
             )
-            self.estimator_ = self.base_estimator
+            if self.base_estimator is not None:
+                self.estimator_ = self.base_estimator
+            else:
+                self.estimator_ = default
         else:
             self.estimator_ = default
 
diff --git a/sklearn/ensemble/_weight_boosting.py b/sklearn/ensemble/_weight_boosting.py
--- a/sklearn/ensemble/_weight_boosting.py
+++ b/sklearn/ensemble/_weight_boosting.py
@@ -64,7 +64,11 @@ class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
         "n_estimators": [Interval(Integral, 1, None, closed="left")],
         "learning_rate": [Interval(Real, 0, None, closed="neither")],
         "random_state": ["random_state"],
-        "base_estimator": [HasMethods(["fit", "predict"]), StrOptions({"deprecated"})],
+        "base_estimator": [
+            HasMethods(["fit", "predict"]),
+            StrOptions({"deprecated"}),
+            None,
+        ],
     }
 
     @abstractmethod

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/ensemble/_base.py | 160 | 160 | 5 | 2 | 1118
| sklearn/ensemble/_base.py | 168 | 168 | 5 | 2 | 1118
| sklearn/ensemble/_weight_boosting.py | 67 | 67 | 4 | 1 | 820


## Problem Statement

```
AdaBoost: deprecation of "base_estimator" does not handle "base_estimator=None" setting properly
### Describe the bug

Scikit-learn 1.2 deprecated `AdaBoostClassifier` 's `base_estimator` in favour of `estimator` (see #23819). Because there are also validators in place, old code that explicitly defined `base_estimator=None` stopped working.

A solution that fixes the deprecation is to add a possible `None` to a list allowed values in `_parameter_constraints`; I will do that in a PR.

### Steps/Code to Reproduce

\`\`\`
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=None)
clf.fit([[1]], [0])
\`\`\`

### Expected Results

No error is thrown.

### Actual Results

\`\`\`
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/marko/opt/miniconda3/envs/orange310/lib/python3.10/site-packages/sklearn/ensemble/_weight_boosting.py", line 124, in fit
    self._validate_params()
  File "/Users/marko/opt/miniconda3/envs/orange310/lib/python3.10/site-packages/sklearn/base.py", line 600, in _validate_params
    validate_parameter_constraints(
  File "/Users/marko/opt/miniconda3/envs/orange310/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 97, in validate_parameter_constraints
    raise InvalidParameterError(
sklearn.utils._param_validation.InvalidParameterError: The 'base_estimator' parameter of AdaBoostClassifier must be an object implementing 'fit' and 'predict' or a str among {'deprecated'}. Got None instead.
\`\`\`

### Versions

\`\`\`shell
sklearn: 1.2.2; others are not important
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/ensemble/_weight_boosting.py** | 485 | 508| 142 | 142 | 9785 | 
| 2 | **1 sklearn/ensemble/_weight_boosting.py** | 1065 | 1093| 184 | 326 | 9785 | 
| 3 | **1 sklearn/ensemble/_weight_boosting.py** | 510 | 527| 184 | 510 | 9785 | 
| **-> 4 <-** | **1 sklearn/ensemble/_weight_boosting.py** | 55 | 100| 310 | 820 | 9785 | 
| **-> 5 <-** | **2 sklearn/ensemble/_base.py** | 146 | 181| 298 | 1118 | 12115 | 
| 6 | **2 sklearn/ensemble/_weight_boosting.py** | 331 | 483| 1378 | 2496 | 12115 | 
| 7 | **2 sklearn/ensemble/_weight_boosting.py** | 934 | 1063| 1141 | 3637 | 12115 | 
| 8 | 3 sklearn/ensemble/_gb.py | 465 | 556| 752 | 4389 | 27895 | 
| 9 | 3 sklearn/ensemble/_gb.py | 1160 | 1211| 370 | 4759 | 27895 | 
| 10 | 3 sklearn/ensemble/_gb.py | 842 | 1158| 3277 | 8036 | 27895 | 
| 11 | 3 sklearn/ensemble/_gb.py | 275 | 309| 316 | 8352 | 27895 | 
| 12 | 4 sklearn/utils/estimator_checks.py | 3429 | 3539| 957 | 9309 | 64458 | 
| 13 | 4 sklearn/utils/estimator_checks.py | 1953 | 2701| 6424 | 15733 | 64458 | 
| 14 | 5 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 628 | 757| 1021 | 16754 | 81262 | 
| 15 | 6 sklearn/base.py | 576 | 616| 465 | 17219 | 89717 | 
| 16 | 7 sklearn/experimental/enable_hist_gradient_boosting.py | 1 | 22| 171 | 17390 | 89888 | 
| 17 | 7 sklearn/ensemble/_gb.py | 134 | 201| 554 | 17944 | 89888 | 
| 18 | 8 benchmarks/bench_hist_gradient_boosting_categorical_only.py | 1 | 81| 651 | 18595 | 90539 | 
| 19 | 8 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 500 | 626| 1100 | 19695 | 90539 | 
| 20 | 8 sklearn/utils/estimator_checks.py | 1696 | 1950| 2262 | 21957 | 90539 | 
| 21 | 9 sklearn/ensemble/_bagging.py | 728 | 761| 203 | 22160 | 100015 | 
| 22 | 9 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 265 | 300| 295 | 22455 | 100015 | 
| 23 | **9 sklearn/ensemble/_weight_boosting.py** | 1 | 52| 314 | 22769 | 100015 | 
| 24 | 10 sklearn/linear_model/_base.py | 48 | 128| 622 | 23391 | 107333 | 
| 25 | 10 sklearn/ensemble/_gb.py | 599 | 664| 599 | 23990 | 107333 | 
| 26 | **10 sklearn/ensemble/_weight_boosting.py** | 1193 | 1208| 178 | 24168 | 107333 | 
| 27 | 10 sklearn/ensemble/_gb.py | 1711 | 1770| 450 | 24618 | 107333 | 
| 28 | 11 examples/ensemble/plot_adaboost_hastie_10_2.py | 104 | 172| 487 | 25105 | 108627 | 
| 29 | 11 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 425 | 499| 652 | 25757 | 108627 | 
| 30 | 11 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 759 | 797| 322 | 26079 | 108627 | 
| 31 | 12 benchmarks/bench_hist_gradient_boosting_threading.py | 87 | 139| 382 | 26461 | 111365 | 
| 32 | 13 sklearn/inspection/_partial_dependence.py | 440 | 509| 756 | 27217 | 116717 | 
| 33 | 13 sklearn/base.py | 332 | 346| 118 | 27335 | 116717 | 
| 34 | **13 sklearn/ensemble/_base.py** | 83 | 144| 425 | 27760 | 116717 | 
| 35 | 13 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 103| 767 | 28527 | 116717 | 
| 36 | 14 sklearn/ensemble/_stacking.py | 597 | 618| 188 | 28715 | 124957 | 
| 37 | 14 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 882 | 916| 208 | 28923 | 124957 | 
| 38 | 15 sklearn/model_selection/__init__.py | 1 | 90| 632 | 29555 | 125589 | 
| 39 | 15 sklearn/ensemble/_gb.py | 343 | 377| 359 | 29914 | 125589 | 
| 40 | **15 sklearn/ensemble/_base.py** | 263 | 290| 237 | 30151 | 125589 | 
| 41 | 15 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 83 | 177| 777 | 30928 | 125589 | 
| 42 | 15 sklearn/base.py | 125 | 164| 321 | 31249 | 125589 | 
| 43 | 15 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 837 | 880| 304 | 31553 | 125589 | 
| 44 | 16 sklearn/linear_model/_stochastic_gradient.py | 9 | 56| 405 | 31958 | 145274 | 
| 45 | 16 sklearn/ensemble/_bagging.py | 1242 | 1280| 297 | 32255 | 145274 | 
| 46 | 16 sklearn/ensemble/_gb.py | 326 | 341| 163 | 32418 | 145274 | 
| 47 | 16 sklearn/ensemble/_gb.py | 1405 | 1709| 3181 | 35599 | 145274 | 
| 48 | 17 asv_benchmarks/benchmarks/ensemble.py | 56 | 92| 203 | 35802 | 145923 | 
| 49 | **17 sklearn/ensemble/_weight_boosting.py** | 1095 | 1191| 725 | 36527 | 145923 | 
| 50 | **17 sklearn/ensemble/_base.py** | 1 | 42| 261 | 36788 | 145923 | 
| 51 | 17 sklearn/utils/estimator_checks.py | 2704 | 3426| 6199 | 42987 | 145923 | 
| 52 | 18 examples/ensemble/plot_adaboost_regression.py | 1 | 77| 619 | 43606 | 146565 | 
| 53 | 19 sklearn/exceptions.py | 155 | 166| 121 | 43727 | 147716 | 
| 54 | 20 sklearn/calibration.py | 250 | 283| 227 | 43954 | 158410 | 
| 55 | 21 benchmarks/bench_hist_gradient_boosting_adult.py | 1 | 38| 315 | 44269 | 159268 | 
| 56 | 21 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 339 | 424| 802 | 45071 | 159268 | 
| 57 | 21 sklearn/base.py | 1 | 32| 191 | 45262 | 159268 | 
| 58 | 21 sklearn/ensemble/_gb.py | 203 | 273| 463 | 45725 | 159268 | 
| 59 | 22 examples/ensemble/plot_gradient_boosting_categorical.py | 218 | 278| 687 | 46412 | 161555 | 
| 60 | 23 doc/sphinxext/allow_nan_estimators.py | 1 | 55| 402 | 46814 | 161957 | 
| 61 | 23 sklearn/ensemble/_gb.py | 379 | 463| 714 | 47528 | 161957 | 
| 62 | 24 examples/release_highlights/plot_release_highlights_0_22_0.py | 195 | 281| 764 | 48292 | 164367 | 
| 63 | 24 sklearn/ensemble/_gb.py | 1213 | 1226| 147 | 48439 | 164367 | 
| 64 | 25 examples/release_highlights/plot_release_highlights_0_24_0.py | 1 | 120| 1150 | 49589 | 166805 | 
| 65 | 26 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 106| 757 | 50346 | 167997 | 
| 66 | 26 sklearn/exceptions.py | 133 | 153| 130 | 50476 | 167997 | 
| 67 | 26 sklearn/linear_model/_stochastic_gradient.py | 141 | 159| 194 | 50670 | 167997 | 
| 68 | 26 sklearn/ensemble/_gb.py | 666 | 688| 216 | 50886 | 167997 | 
| 69 | 26 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 1424 | 1486| 407 | 51293 | 167997 | 
| 70 | 27 examples/ensemble/plot_adaboost_multiclass.py | 1 | 91| 749 | 52042 | 169022 | 
| 71 | 27 sklearn/ensemble/_stacking.py | 147 | 228| 655 | 52697 | 169022 | 
| 72 | 28 sklearn/multiclass.py | 114 | 162| 303 | 53000 | 177489 | 


## Patch

```diff
diff --git a/sklearn/ensemble/_base.py b/sklearn/ensemble/_base.py
--- a/sklearn/ensemble/_base.py
+++ b/sklearn/ensemble/_base.py
@@ -157,7 +157,7 @@ def _validate_estimator(self, default=None):
 
         if self.estimator is not None:
             self.estimator_ = self.estimator
-        elif self.base_estimator not in [None, "deprecated"]:
+        elif self.base_estimator != "deprecated":
             warnings.warn(
                 (
                     "`base_estimator` was renamed to `estimator` in version 1.2 and "
@@ -165,7 +165,10 @@ def _validate_estimator(self, default=None):
                 ),
                 FutureWarning,
             )
-            self.estimator_ = self.base_estimator
+            if self.base_estimator is not None:
+                self.estimator_ = self.base_estimator
+            else:
+                self.estimator_ = default
         else:
             self.estimator_ = default
 
diff --git a/sklearn/ensemble/_weight_boosting.py b/sklearn/ensemble/_weight_boosting.py
--- a/sklearn/ensemble/_weight_boosting.py
+++ b/sklearn/ensemble/_weight_boosting.py
@@ -64,7 +64,11 @@ class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
         "n_estimators": [Interval(Integral, 1, None, closed="left")],
         "learning_rate": [Interval(Real, 0, None, closed="neither")],
         "random_state": ["random_state"],
-        "base_estimator": [HasMethods(["fit", "predict"]), StrOptions({"deprecated"})],
+        "base_estimator": [
+            HasMethods(["fit", "predict"]),
+            StrOptions({"deprecated"}),
+            None,
+        ],
     }
 
     @abstractmethod

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_weight_boosting.py b/sklearn/ensemble/tests/test_weight_boosting.py
--- a/sklearn/ensemble/tests/test_weight_boosting.py
+++ b/sklearn/ensemble/tests/test_weight_boosting.py
@@ -613,6 +613,27 @@ def test_base_estimator_argument_deprecated(AdaBoost, Estimator):
         model.fit(X, y)
 
 
+# TODO(1.4): remove in 1.4
+@pytest.mark.parametrize(
+    "AdaBoost",
+    [
+        AdaBoostClassifier,
+        AdaBoostRegressor,
+    ],
+)
+def test_base_estimator_argument_deprecated_none(AdaBoost):
+    X = np.array([[1, 2], [3, 4]])
+    y = np.array([1, 0])
+    model = AdaBoost(base_estimator=None)
+
+    warn_msg = (
+        "`base_estimator` was renamed to `estimator` in version 1.2 and "
+        "will be removed in 1.4."
+    )
+    with pytest.warns(FutureWarning, match=warn_msg):
+        model.fit(X, y)
+
+
 # TODO(1.4): remove in 1.4
 @pytest.mark.parametrize(
     "AdaBoost",

```


## Code snippets

### 1 - sklearn/ensemble/_weight_boosting.py:

Start line: 485, End line: 508

```python
class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):

    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "algorithm": [StrOptions({"SAMME", "SAMME.R"})],
    }

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            base_estimator=base_estimator,
        )

        self.algorithm = algorithm
```
### 2 - sklearn/ensemble/_weight_boosting.py:

Start line: 1065, End line: 1093

```python
class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):

    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "loss": [StrOptions({"linear", "square", "exponential"})],
    }

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
        random_state=None,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            base_estimator=base_estimator,
        )

        self.loss = loss
        self.random_state = random_state

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor(max_depth=3))
```
### 3 - sklearn/ensemble/_weight_boosting.py:

Start line: 510, End line: 527

```python
class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == "SAMME.R":
            if not hasattr(self.estimator_, "predict_proba"):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                f"{self.estimator.__class__.__name__} doesn't support sample_weight."
            )
```
### 4 - sklearn/ensemble/_weight_boosting.py:

Start line: 55, End line: 100

```python
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
        "base_estimator": [HasMethods(["fit", "predict"]), StrOptions({"deprecated"})],
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        estimator_params=tuple(),
        learning_rate=1.0,
        random_state=None,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            base_estimator=base_estimator,
        )

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        # Only called to validate X in non-fit methods, therefore reset=False
        return self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            reset=False,
        )
```
### 5 - sklearn/ensemble/_base.py:

Start line: 146, End line: 181

```python
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    def _validate_estimator(self, default=None):
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None and (
            self.base_estimator not in [None, "deprecated"]
        ):
            raise ValueError(
                "Both `estimator` and `base_estimator` were set. Only set `estimator`."
            )

        if self.estimator is not None:
            self.estimator_ = self.estimator
        elif self.base_estimator not in [None, "deprecated"]:
            warnings.warn(
                (
                    "`base_estimator` was renamed to `estimator` in version 1.2 and "
                    "will be removed in 1.4."
                ),
                FutureWarning,
            )
            self.estimator_ = self.base_estimator
        else:
            self.estimator_ = default

    # TODO(1.4): remove
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `base_estimator_` was deprecated in version 1.2 and will be removed "
        "in 1.4. Use `estimator_` instead."
    )
    @property
    def base_estimator_(self):
        """Estimator used to grow the ensemble."""
        return self.estimator_
```
### 6 - sklearn/ensemble/_weight_boosting.py:

Start line: 331, End line: 483

```python
class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. deprecated:: 1.2
            `base_estimator` is deprecated and will be removed in 1.4.
            Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

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

    See Also
    --------
    AdaBoostRegressor : An AdaBoost regressor that begins by fitting a
        regressor on the original dataset and then fits additional copies of
        the regressor on the same dataset but where the weights of instances
        are adjusted according to the error of the current prediction.

    GradientBoostingClassifier : GB builds an additive model in a forward
        stage-wise fashion. Regression trees are fit on the negative gradient
        of the binomial or multinomial deviance loss function. Binary
        classification is a special case where only a single regression tree is
        induced.

    sklearn.tree.DecisionTreeClassifier : A non-parametric supervised learning
        method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.983...
    """
```
### 7 - sklearn/ensemble/_weight_boosting.py:

Start line: 934, End line: 1063

```python
class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration. A higher
        learning rate increases the contribution of each regressor. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. deprecated:: 1.2
            `base_estimator` is deprecated and will be removed in 1.4.
            Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

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

    See Also
    --------
    AdaBoostClassifier : An AdaBoost classifier.
    GradientBoostingRegressor : Gradient Boosting Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...
    """
```
### 8 - sklearn/ensemble/_gb.py:

Start line: 465, End line: 556

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, monitor=None):
        # ... other code

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == "zero":
                raw_predictions = np.zeros(
                    shape=(X.shape[0], self._loss.K), dtype=np.float64
                )
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = (
                        "The initial estimator {} does not support sample "
                        "weights.".format(self.init_.__class__.__name__)
                    )
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            # regular estimator without SW support
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise
                    except ValueError as e:
                        if (
                            "pass parameters to specific steps of "
                            "your pipeline using the "
                            "stepname__parameter"
                            in str(e)
                        ):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = self._loss.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True" % (self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _raw_predict
            # are more constrained than fit. It accepts only CSR
            # matrices. Finite values have already been checked in _validate_data.
            X = check_array(
                X,
                dtype=DTYPE,
                order="C",
                accept_sparse="csr",
                force_all_finite=False,
            )
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X,
            y,
            raw_predictions,
            sample_weight,
            self._rng,
            X_val,
            y_val,
            sample_weight_val,
            begin_at_stage,
            monitor,
        )

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, "oob_improvement_"):
                # OOB scores were computed
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]
        self.n_estimators_ = n_stages
        return self
```
### 9 - sklearn/ensemble/_gb.py:

Start line: 1160, End line: 1211

```python
class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):

    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
```
### 10 - sklearn/ensemble/_gb.py:

Start line: 842, End line: 1158

```python
class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.

    This algorithm builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage ``n_classes_`` regression trees are fit on the negative gradient
    of the loss function, e.g. binary or multiclass log loss. Binary
    classification is a special case where only a single regression tree is
    induced.

    :class:`sklearn.ensemble.HistGradientBoostingClassifier` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'log_loss', 'exponential'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to binomial and
        multinomial deviance, the same as used in logistic regression.
        It is a good choice for classification with probabilistic outputs.
        For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        'friedman_mse' for the mean squared error with improvement score by
        Friedman, 'squared_error' for mean squared error. The default value of
        'friedman_mse' is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If `None`, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range `[1, inf)`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor of \
            shape (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_classes_ : int
        The number of classes.

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingClassifier : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    The following example shows how to fit a gradient boosting classifier with
    100 decision stumps as weak learners.

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.913...
    """
```
### 23 - sklearn/ensemble/_weight_boosting.py:

Start line: 1, End line: 52

```python
"""Weight Boosting.

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The `BaseWeightBoosting` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- :class:`~sklearn.ensemble.AdaBoostClassifier` implements adaptive boosting
  (AdaBoost-SAMME) for classification problems.

- :class:`~sklearn.ensemble.AdaBoostRegressor` implements adaptive boosting
  (AdaBoost.R2) for regression problems.
"""

from abc import ABCMeta, abstractmethod

from numbers import Integral, Real
import numpy as np

import warnings

from scipy.special import xlogy

from ._base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_random_state, _safe_indexing
from ..utils.extmath import softmax
from ..utils.extmath import stable_cumsum
from ..metrics import accuracy_score, r2_score
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_sample_weight
from ..utils.validation import has_fit_parameter
from ..utils.validation import _num_samples
from ..utils._param_validation import HasMethods, Interval, StrOptions

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
]
```
### 26 - sklearn/ensemble/_weight_boosting.py:

Start line: 1193, End line: 1208

```python
class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]
```
### 34 - sklearn/ensemble/_base.py:

Start line: 83, End line: 144

```python
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    base_estimator : object, default="deprecated"
        Use `estimator` instead.

        .. deprecated:: 1.2
            `base_estimator` is deprecated and will be removed in 1.4.
            Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        estimator_params=tuple(),
        base_estimator="deprecated",
    ):
        # Set parameters
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.base_estimator = base_estimator

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.
```
### 40 - sklearn/ensemble/_base.py:

Start line: 263, End line: 290

```python
class _BaseHeterogeneousEnsemble(
    MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta
):

    def _validate_estimators(self):
        if len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a "
                "non-empty list of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin
        self._validate_names(names)

        has_estimator = any(est != "drop" for est in estimators)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        is_estimator_type = is_classifier if is_classifier(self) else is_regressor

        for est in estimators:
            if est != "drop" and not is_estimator_type(est):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        est.__class__.__name__, is_estimator_type.__name__[3:]
                    )
                )

        return names, estimators
```
### 49 - sklearn/ensemble/_weight_boosting.py:

Start line: 1095, End line: 1191

```python
class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
            Controls also the bootstrap of the weights used to train the weak
            learner.
            replacement.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)),
            size=_num_samples(X),
            replace=True,
            p=sample_weight,
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == "square":
            masked_error_vector **= 2
        elif self.loss == "exponential":
            masked_error_vector = 1.0 - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error
```
### 50 - sklearn/ensemble/_base.py:

Start line: 1, End line: 42

```python
"""Base class for ensemble-based estimators."""

from abc import ABCMeta, abstractmethod
from typing import List
import warnings

import numpy as np

from joblib import effective_n_jobs

from ..base import clone
from ..base import is_classifier, is_regressor
from ..base import BaseEstimator
from ..base import MetaEstimatorMixin
from ..utils import Bunch, _print_elapsed_time, deprecated
from ..utils import check_random_state
from ..utils.metaestimators import _BaseComposition


def _fit_single_estimator(
    estimator, X, y, sample_weight=None, message_clsname=None, message=None
):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y)
    return estimator
```
