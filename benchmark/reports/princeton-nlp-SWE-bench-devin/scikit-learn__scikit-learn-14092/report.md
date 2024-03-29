# scikit-learn__scikit-learn-14092

| **scikit-learn/scikit-learn** | `df7dd8391148a873d157328a4f0328528a0c4ed9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 16387 |
| **Any found context length** | 563 |
| **Avg pos** | 26.0 |
| **Min pos** | 1 |
| **Max pos** | 24 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/neighbors/nca.py b/sklearn/neighbors/nca.py
--- a/sklearn/neighbors/nca.py
+++ b/sklearn/neighbors/nca.py
@@ -13,6 +13,7 @@
 import numpy as np
 import sys
 import time
+import numbers
 from scipy.optimize import minimize
 from ..utils.extmath import softmax
 from ..metrics import pairwise_distances
@@ -299,7 +300,8 @@ def _validate_params(self, X, y):
 
         # Check the preferred dimensionality of the projected space
         if self.n_components is not None:
-            check_scalar(self.n_components, 'n_components', int, 1)
+            check_scalar(
+                self.n_components, 'n_components', numbers.Integral, 1)
 
             if self.n_components > X.shape[1]:
                 raise ValueError('The preferred dimensionality of the '
@@ -318,9 +320,9 @@ def _validate_params(self, X, y):
                                  .format(X.shape[1],
                                          self.components_.shape[1]))
 
-        check_scalar(self.max_iter, 'max_iter', int, 1)
-        check_scalar(self.tol, 'tol', float, 0.)
-        check_scalar(self.verbose, 'verbose', int, 0)
+        check_scalar(self.max_iter, 'max_iter', numbers.Integral, 1)
+        check_scalar(self.tol, 'tol', numbers.Real, 0.)
+        check_scalar(self.verbose, 'verbose', numbers.Integral, 0)
 
         if self.callback is not None:
             if not callable(self.callback):

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/neighbors/nca.py | 16 | 16 | 24 | 1 | 16387
| sklearn/neighbors/nca.py | 302 | 302 | 1 | 1 | 563
| sklearn/neighbors/nca.py | 321 | 323 | 1 | 1 | 563


## Problem Statement

```
NCA fails in GridSearch due to too strict parameter checks
NCA checks its parameters to have a specific type, which can easily fail in a GridSearch due to how param grid is made.

Here is an example:
\`\`\`python
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier

X = np.random.random_sample((100, 10))
y = np.random.randint(2, size=100)

nca = NeighborhoodComponentsAnalysis()
knn = KNeighborsClassifier()

pipe = Pipeline([('nca', nca),
                 ('knn', knn)])
                
params = {'nca__tol': [0.1, 0.5, 1],
          'nca__n_components': np.arange(1, 10)}
          
gs = GridSearchCV(estimator=pipe, param_grid=params, error_score='raise')
gs.fit(X,y)
\`\`\`

The issue is that for `tol`: 1 is not a float, and for  `n_components`: np.int64 is not int

Before proposing a fix for this specific situation, I'd like to have your general opinion about parameter checking.  
I like this idea of common parameter checking tool introduced with the NCA PR. What do you think about extending it across the code-base (or at least for new or recent estimators) ?

Currently parameter checking is not always done or often partially done, and is quite redundant. For instance, here is the input validation of lda:
\`\`\`python
def _check_params(self):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self.n_components)

        if self.total_samples <= 0:
            raise ValueError("Invalid 'total_samples' parameter: %r"
                             % self.total_samples)

        if self.learning_offset < 0:
            raise ValueError("Invalid 'learning_offset' parameter: %r"
                             % self.learning_offset)

        if self.learning_method not in ("batch", "online"):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)
\`\`\`
most params aren't checked and for those who are there's a lot of duplicated code.

A propose to be upgrade the new tool to be able to check open/closed intervals (currently only closed) and list membership.

The api would be something like that:
\`\`\`
check_param(param, name, valid_options)
\`\`\`
where valid_options would be a dict of `type: constraint`. e.g for the `beta_loss` param of `NMF`, it can be either a float or a string in a list, which would give
\`\`\`
valid_options = {numbers.Real: None,  # None for no constraint
                 str: ['frobenius', 'kullback-leibler', 'itakura-saito']}
\`\`\`
Sometimes a parameter can only be positive or within a given interval, e.g. `l1_ratio` of `LogisticRegression` must be between 0 and 1, which would give
\`\`\`
valid_options = {numbers.Real: Interval(0, 1, closed='both')}
\`\`\`
positivity of e.g. `max_iter` would be `numbers.Integral: Interval(left=1)`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/neighbors/nca.py** | 262 | 330| 563 | 563 | 4186 | 
| 2 | 2 sklearn/utils/estimator_checks.py | 324 | 402| 707 | 1270 | 26402 | 
| 3 | 3 sklearn/ensemble/gradient_boosting.py | 1255 | 1337| 797 | 2067 | 47877 | 
| 4 | 3 sklearn/utils/estimator_checks.py | 2161 | 2237| 630 | 2697 | 47877 | 
| 5 | 4 sklearn/model_selection/_search.py | 358 | 375| 158 | 2855 | 60803 | 
| 6 | 5 sklearn/linear_model/stochastic_gradient.py | 110 | 144| 422 | 3277 | 74463 | 
| 7 | **5 sklearn/neighbors/nca.py** | 332 | 368| 375 | 3652 | 74463 | 
| 8 | 5 sklearn/model_selection/_search.py | 817 | 1104| 3064 | 6716 | 74463 | 
| 9 | 5 sklearn/utils/estimator_checks.py | 2322 | 2374| 436 | 7152 | 74463 | 
| 10 | **5 sklearn/neighbors/nca.py** | 29 | 156| 1234 | 8386 | 74463 | 
| 11 | 5 sklearn/model_selection/_search.py | 1105 | 1435| 188 | 8574 | 74463 | 
| 12 | 5 sklearn/model_selection/_search.py | 1125 | 1414| 2997 | 11571 | 74463 | 
| 13 | 5 sklearn/model_selection/_search.py | 771 | 1121| 443 | 12014 | 74463 | 
| 14 | 5 sklearn/utils/estimator_checks.py | 2026 | 2062| 366 | 12380 | 74463 | 
| 15 | 6 sklearn/model_selection/__init__.py | 1 | 60| 405 | 12785 | 74868 | 
| 16 | 6 sklearn/utils/estimator_checks.py | 732 | 784| 466 | 13251 | 74868 | 
| 17 | 6 sklearn/utils/estimator_checks.py | 1185 | 1253| 601 | 13852 | 74868 | 
| 18 | 6 sklearn/utils/estimator_checks.py | 2284 | 2319| 401 | 14253 | 74868 | 
| 19 | 6 sklearn/model_selection/_search.py | 46 | 111| 562 | 14815 | 74868 | 
| 20 | 7 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 46 | 74| 289 | 15104 | 83113 | 
| 21 | 8 sklearn/decomposition/nmf.py | 197 | 224| 295 | 15399 | 95011 | 
| 22 | 8 sklearn/model_selection/_search.py | 1 | 43| 262 | 15661 | 95011 | 
| 23 | **8 sklearn/neighbors/nca.py** | 158 | 236| 593 | 16254 | 95011 | 
| **-> 24 <-** | **8 sklearn/neighbors/nca.py** | 1 | 26| 133 | 16387 | 95011 | 
| 25 | 9 examples/model_selection/plot_grid_search_refit_callable.py | 78 | 117| 323 | 16710 | 95843 | 
| 26 | 9 sklearn/utils/estimator_checks.py | 678 | 729| 436 | 17146 | 95843 | 
| 27 | 9 sklearn/utils/estimator_checks.py | 835 | 873| 375 | 17521 | 95843 | 
| 28 | 10 sklearn/gaussian_process/kernels.py | 92 | 115| 287 | 17808 | 111133 | 
| 29 | 11 sklearn/decomposition/online_lda.py | 300 | 316| 147 | 17955 | 117643 | 
| 30 | 11 sklearn/utils/estimator_checks.py | 273 | 321| 368 | 18323 | 117643 | 
| 31 | 12 sklearn/linear_model/coordinate_descent.py | 1399 | 1577| 1649 | 19972 | 138207 | 
| 32 | 13 sklearn/neighbors/base.py | 124 | 162| 368 | 20340 | 145838 | 
| 33 | 14 sklearn/ensemble/_hist_gradient_boosting/grower.py | 204 | 235| 335 | 20675 | 150000 | 
| 34 | 14 sklearn/utils/estimator_checks.py | 1915 | 1964| 499 | 21174 | 150000 | 
| 35 | 15 examples/compose/plot_compare_reduction.py | 1 | 105| 823 | 21997 | 151074 | 
| 36 | 15 sklearn/utils/estimator_checks.py | 1461 | 1560| 927 | 22924 | 151074 | 
| 37 | 15 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 361 | 23285 | 151074 | 
| 38 | 15 sklearn/model_selection/_search.py | 140 | 179| 259 | 23544 | 151074 | 
| 39 | 15 examples/model_selection/plot_grid_search_refit_callable.py | 1 | 30| 213 | 23757 | 151074 | 
| 40 | 16 examples/model_selection/plot_randomized_search.py | 54 | 87| 310 | 24067 | 151766 | 
| 41 | 16 sklearn/utils/estimator_checks.py | 1 | 62| 452 | 24519 | 151766 | 
| 42 | 16 sklearn/utils/estimator_checks.py | 79 | 112| 261 | 24780 | 151766 | 
| 43 | 17 examples/model_selection/plot_multi_metric_evaluation.py | 1 | 70| 543 | 25323 | 152646 | 
| 44 | 17 sklearn/model_selection/_search.py | 1415 | 1436| 218 | 25541 | 152646 | 
| 45 | 17 sklearn/utils/estimator_checks.py | 1846 | 1888| 469 | 26010 | 152646 | 
| 46 | 18 examples/model_selection/plot_grid_search_digits.py | 1 | 79| 631 | 26641 | 153277 | 
| 47 | 19 sklearn/model_selection/_validation.py | 1441 | 1459| 263 | 26904 | 166346 | 
| 48 | 20 examples/calibration/plot_calibration_multiclass.py | 1 | 80| 762 | 27666 | 168473 | 
| 49 | 21 benchmarks/bench_hist_gradient_boosting.py | 1 | 37| 349 | 28015 | 170699 | 
| 50 | 22 sklearn/preprocessing/data.py | 11 | 60| 312 | 28327 | 195770 | 
| 51 | 22 sklearn/model_selection/_search.py | 113 | 138| 196 | 28523 | 195770 | 
| 52 | 22 sklearn/model_selection/_search.py | 378 | 400| 185 | 28708 | 195770 | 
| 53 | 23 examples/model_selection/grid_search_text_feature_extraction.py | 95 | 129| 369 | 29077 | 196860 | 


### Hint

```
I have developed a framework, experimenting with parameter verification: https://github.com/thomasjpfan/skconfig (Don't expect the API to be stable)

Your idea of using a simple dict for union types is really nice!

Edit: I am currently trying out another idea. I'll update this issue when it becomes something presentable.
If I understood correctly your package is designed for a sklearn user, who has to implement its validator for each estimator, or did I get it wrong ?
I think we want to keep the param validation inside the estimators.

> Edit: I am currently trying out another idea. I'll update this issue when it becomes something presentable.

maybe you can pitch me and if you want I can give a hand :)
I would have loved to using the typing system to get this to work:

\`\`\`py
def __init__(
    self,
    C: Annotated[float, Range('[0, Inf)')],
    ...)
\`\`\`

but that would have to wait for [PEP 593](https://www.python.org/dev/peps/pep-0593/). In the end, I would want the validator to be a part of sklearn estimators. Using typing (as above) is a natural choice, since it keeps the parameter and its constraint physically close to each other.

If we can't use typing, these constraints can be place in a `_validate_parameters` method. This will be called at the beginning of fit to do parameter validation. Estimators that need more validation will overwrite the method, call `super()._validate_parameters` and do more validation. For example, `LogesticRegression`'s `penalty='l2'` only works for specify solvers. `skconfig` defines a framework for handling these situations, but I think it would be too hard to learn.
>  Using typing (as above) is a natural choice

I agree, and to go further it would be really nice to use them for the coverage to check that every possible type of a parameter is covered by tests

> If we can't use typing, these constraints can be place in a _validate_parameters method. 

This is already the case for a subset of the estimators (`_check_params` or `_validate_input`). But it's often incomplete.

> skconfig defines a framework for handling these situations, but I think it would be too hard to learn.

Your framework does way more than what I proposed. Maybe we can do this in 2 steps:
First, a simple single param check which only checks its type and if its value is acceptable in general (e.g. positive for a number of clusters). This will raise a standard error message
Then a more advanced check, depending on the data (e.g. number of clusters should be < n_samples) or consistency across params (e.g. solver + penalty). These checks require more elaborate error messages.

wdyt ?
```

## Patch

```diff
diff --git a/sklearn/neighbors/nca.py b/sklearn/neighbors/nca.py
--- a/sklearn/neighbors/nca.py
+++ b/sklearn/neighbors/nca.py
@@ -13,6 +13,7 @@
 import numpy as np
 import sys
 import time
+import numbers
 from scipy.optimize import minimize
 from ..utils.extmath import softmax
 from ..metrics import pairwise_distances
@@ -299,7 +300,8 @@ def _validate_params(self, X, y):
 
         # Check the preferred dimensionality of the projected space
         if self.n_components is not None:
-            check_scalar(self.n_components, 'n_components', int, 1)
+            check_scalar(
+                self.n_components, 'n_components', numbers.Integral, 1)
 
             if self.n_components > X.shape[1]:
                 raise ValueError('The preferred dimensionality of the '
@@ -318,9 +320,9 @@ def _validate_params(self, X, y):
                                  .format(X.shape[1],
                                          self.components_.shape[1]))
 
-        check_scalar(self.max_iter, 'max_iter', int, 1)
-        check_scalar(self.tol, 'tol', float, 0.)
-        check_scalar(self.verbose, 'verbose', int, 0)
+        check_scalar(self.max_iter, 'max_iter', numbers.Integral, 1)
+        check_scalar(self.tol, 'tol', numbers.Real, 0.)
+        check_scalar(self.verbose, 'verbose', numbers.Integral, 0)
 
         if self.callback is not None:
             if not callable(self.callback):

```

## Test Patch

```diff
diff --git a/sklearn/neighbors/tests/test_nca.py b/sklearn/neighbors/tests/test_nca.py
--- a/sklearn/neighbors/tests/test_nca.py
+++ b/sklearn/neighbors/tests/test_nca.py
@@ -129,7 +129,7 @@ def test_params_validation():
     # TypeError
     assert_raises(TypeError, NCA(max_iter='21').fit, X, y)
     assert_raises(TypeError, NCA(verbose='true').fit, X, y)
-    assert_raises(TypeError, NCA(tol=1).fit, X, y)
+    assert_raises(TypeError, NCA(tol='1').fit, X, y)
     assert_raises(TypeError, NCA(n_components='invalid').fit, X, y)
     assert_raises(TypeError, NCA(warm_start=1).fit, X, y)
 
@@ -518,3 +518,17 @@ def test_convergence_warning():
     assert_warns_message(ConvergenceWarning,
                          '[{}] NCA did not converge'.format(cls_name),
                          nca.fit, iris_data, iris_target)
+
+
+@pytest.mark.parametrize('param, value', [('n_components', np.int32(3)),
+                                          ('max_iter', np.int32(100)),
+                                          ('tol', np.float32(0.0001))])
+def test_parameters_valid_types(param, value):
+    # check that no error is raised when parameters have numpy integer or
+    # floating types.
+    nca = NeighborhoodComponentsAnalysis(**{param: value})
+
+    X = iris_data
+    y = iris_target
+
+    nca.fit(X, y)

```


## Code snippets

### 1 - sklearn/neighbors/nca.py:

Start line: 262, End line: 330

```python
class NeighborhoodComponentsAnalysis(BaseEstimator, TransformerMixin):

    def _validate_params(self, X, y):
        """Validate parameters as soon as :meth:`fit` is called.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training samples.

        y : array-like, shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            The validated training samples.

        y : array, shape (n_samples,)
            The validated training labels, encoded to be integers in
            the range(0, n_classes).

        init : string or numpy array of shape (n_features_a, n_features_b)
            The validated initialization of the linear transformation.

        Raises
        -------
        TypeError
            If a parameter is not an instance of the desired type.

        ValueError
            If a parameter's value violates its legal value range or if the
            combination of two or more given parameters is incompatible.
        """

        # Validate the inputs X and y, and converts y to numerical classes.
        X, y = check_X_y(X, y, ensure_min_samples=2)
        check_classification_targets(y)
        y = LabelEncoder().fit_transform(y)

        # Check the preferred dimensionality of the projected space
        if self.n_components is not None:
            check_scalar(self.n_components, 'n_components', int, 1)

            if self.n_components > X.shape[1]:
                raise ValueError('The preferred dimensionality of the '
                                 'projected space `n_components` ({}) cannot '
                                 'be greater than the given data '
                                 'dimensionality ({})!'
                                 .format(self.n_components, X.shape[1]))

        # If warm_start is enabled, check that the inputs are consistent
        check_scalar(self.warm_start, 'warm_start', bool)
        if self.warm_start and hasattr(self, 'components_'):
            if self.components_.shape[1] != X.shape[1]:
                raise ValueError('The new inputs dimensionality ({}) does not '
                                 'match the input dimensionality of the '
                                 'previously learned transformation ({}).'
                                 .format(X.shape[1],
                                         self.components_.shape[1]))

        check_scalar(self.max_iter, 'max_iter', int, 1)
        check_scalar(self.tol, 'tol', float, 0.)
        check_scalar(self.verbose, 'verbose', int, 0)

        if self.callback is not None:
            if not callable(self.callback):
                raise ValueError('`callback` is not callable.')

        # Check how the linear transformation should be initialized
        init = self.init
        # ... other code
```
### 2 - sklearn/utils/estimator_checks.py:

Start line: 324, End line: 402

```python
def set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if ("n_iter" in params and name != "TSNE"):
        estimator.set_params(n_iter=5)
    if "max_iter" in params:
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        # LinearSVR, LinearSVC
        if estimator.__class__.__name__ in ['LinearSVR', 'LinearSVC']:
            estimator.set_params(max_iter=20)
        # NMF
        if estimator.__class__.__name__ == 'NMF':
            estimator.set_params(max_iter=100)
        # MLP
        if estimator.__class__.__name__ in ['MLPClassifier', 'MLPRegressor']:
            estimator.set_params(max_iter=100)
    if "n_resampling" in params:
        # randomized lasso
        estimator.set_params(n_resampling=5)
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if "max_trials" in params:
        # RANSAC
        estimator.set_params(max_trials=10)
    if "n_init" in params:
        # K-Means
        estimator.set_params(n_init=2)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 2

    if name == 'TruncatedSVD':
        # TruncatedSVD doesn't run with n_components = n_features
        # This is ugly :-/
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = min(estimator.n_clusters, 2)

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    if name == "SelectFdr":
        # be tolerant of noisy datasets (not actually speed)
        estimator.set_params(alpha=.5)

    if name == "TheilSenRegressor":
        estimator.max_subpopulation = 100

    if isinstance(estimator, BaseRandomProjection):
        # Due to the jl lemma and often very few samples, the number
        # of components of the random matrix projection will be probably
        # greater than the number of features.
        # So we impose a smaller number (avoid "auto" mode)
        estimator.set_params(n_components=2)

    if isinstance(estimator, SelectKBest):
        # SelectKBest has a default of k=10
        # which is more feature than we have in most case.
        estimator.set_params(k=1)

    if name in ('HistGradientBoostingClassifier',
                'HistGradientBoostingRegressor'):
        # The default min_samples_leaf (20) isn't appropriate for small
        # datasets (only very shallow trees are built) that the checks use.
        estimator.set_params(min_samples_leaf=5)

    # Speed-up by reducing the number of CV or splits for CV estimators
    loo_cv = ['RidgeCV']
    if name not in loo_cv and hasattr(estimator, 'cv'):
        estimator.set_params(cv=3)
    if hasattr(estimator, 'n_splits'):
        estimator.set_params(n_splits=3)

    if name == 'OneHotEncoder':
        estimator.set_params(handle_unknown='ignore')
```
### 3 - sklearn/ensemble/gradient_boosting.py:

Start line: 1255, End line: 1337

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in _gb_losses.LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (_gb_losses.MultinomialDeviance
                          if len(self.classes_) > 2
                          else _gb_losses.BinomialDeviance)
        else:
            loss_class = _gb_losses.LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

        allowed_presort = ('auto', True, False)
        if self.presort not in allowed_presort:
            raise ValueError("'presort' should be in {}. Got {!r} instead."
                             .format(allowed_presort, self.presort))
```
### 4 - sklearn/utils/estimator_checks.py:

Start line: 2161, End line: 2237

```python
def check_parameters_default_constructible(name, Estimator):
    # this check works on classes, not instances
    # test default-constructibility
    # get rid of deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        required_parameters = getattr(Estimator, "_required_parameters", [])
        if required_parameters:
            if required_parameters in (["base_estimator"], ["estimator"]):
                if issubclass(Estimator, RegressorMixin):
                    estimator = Estimator(Ridge())
                else:
                    estimator = Estimator(LinearDiscriminantAnalysis())
            else:
                raise SkipTest("Can't instantiate estimator {} which"
                               " requires parameters {}".format(
                                   name, required_parameters))
        else:
            estimator = Estimator()
        # test cloning
        clone(estimator)
        # test __repr__
        repr(estimator)
        # test that set_params returns self
        assert estimator.set_params() is estimator

        # test if init does nothing but set parameters
        # this is important for grid_search etc.
        # We get the default parameters from init and then
        # compare these against the actual values of the attributes.

        # this comes from getattr. Gets rid of deprecation decorator.
        init = getattr(estimator.__init__, 'deprecated_original',
                       estimator.__init__)

        try:
            def param_filter(p):
                """Identify hyper parameters of an estimator"""
                return (p.name != 'self' and
                        p.kind != p.VAR_KEYWORD and
                        p.kind != p.VAR_POSITIONAL)

            init_params = [p for p in signature(init).parameters.values()
                           if param_filter(p)]

        except (TypeError, ValueError):
            # init is not a python function.
            # true for mixins
            return
        params = estimator.get_params()
        if required_parameters == ["estimator"]:
            # they can need a non-default argument
            init_params = init_params[1:]

        for init_param in init_params:
            assert_not_equal(init_param.default, init_param.empty,
                             "parameter %s for %s has no default value"
                             % (init_param.name, type(estimator).__name__))
            if type(init_param.default) is type:
                assert_in(init_param.default, [np.float64, np.int64])
            else:
                assert_in(type(init_param.default),
                          [str, int, float, bool, tuple, type(None),
                           np.float64, types.FunctionType, _joblib.Memory])
            if init_param.name not in params.keys():
                # deprecated parameter, not in get_params
                assert init_param.default is None
                continue

            param_value = params[init_param.name]
            if isinstance(param_value, np.ndarray):
                assert_array_equal(param_value, init_param.default)
            else:
                if is_scalar_nan(param_value):
                    # Allows to set default parameters to np.nan
                    assert param_value is init_param.default, init_param.name
                else:
                    assert param_value == init_param.default, init_param.name
```
### 5 - sklearn/model_selection/_search.py:

Start line: 358, End line: 375

```python
def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, str) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))
```
### 6 - sklearn/linear_model/stochastic_gradient.py:

Start line: 110, End line: 144

```python
class BaseSGD(BaseEstimator, SparseCoefMixin, metaclass=ABCMeta):

    def _validate_params(self, set_max_iter=True, for_partial_fit=False):
        """Validate input params. """
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False")
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be > zero. Got %f" % self.max_iter)
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if self.n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in ]0, 1[")
        if self.learning_rate in ("constant", "invscaling", "adaptive"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError("alpha must be > 0 since "
                             "learning_rate is 'optimal'. alpha is used "
                             "to compute the optimal learning rate.")

        # raises ValueError if not registered
        self._get_penalty_type(self.penalty)
        self._get_learning_rate_type(self.learning_rate)

        if self.loss not in self.loss_functions:
            raise ValueError("The loss %s is not supported. " % self.loss)

        if not set_max_iter:
            return
```
### 7 - sklearn/neighbors/nca.py:

Start line: 332, End line: 368

```python
class NeighborhoodComponentsAnalysis(BaseEstimator, TransformerMixin):

    def _validate_params(self, X, y):
        # ... other code

        if isinstance(init, np.ndarray):
            init = check_array(init)

            # Assert that init.shape[1] = X.shape[1]
            if init.shape[1] != X.shape[1]:
                raise ValueError(
                    'The input dimensionality ({}) of the given '
                    'linear transformation `init` must match the '
                    'dimensionality of the given inputs `X` ({}).'
                    .format(init.shape[1], X.shape[1]))

            # Assert that init.shape[0] <= init.shape[1]
            if init.shape[0] > init.shape[1]:
                raise ValueError(
                    'The output dimensionality ({}) of the given '
                    'linear transformation `init` cannot be '
                    'greater than its input dimensionality ({}).'
                    .format(init.shape[0], init.shape[1]))

            if self.n_components is not None:
                # Assert that self.n_components = init.shape[0]
                if self.n_components != init.shape[0]:
                    raise ValueError('The preferred dimensionality of the '
                                     'projected space `n_components` ({}) does'
                                     ' not match the output dimensionality of '
                                     'the given linear transformation '
                                     '`init` ({})!'
                                     .format(self.n_components,
                                             init.shape[0]))
        elif init in ['auto', 'pca', 'lda', 'identity', 'random']:
            pass
        else:
            raise ValueError(
                "`init` must be 'auto', 'pca', 'lda', 'identity', 'random' "
                "or a numpy array of shape (n_components, n_features).")

        return X, y, init
```
### 8 - sklearn/model_selection/_search.py:

Start line: 817, End line: 1104

```python
class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=False
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds.

        .. deprecated:: 0.22
            Parameter ``iid`` is deprecated in 0.22 and will be removed in 0.24

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. ``best_score_`` is not returned if refit is callable.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 2322, End line: 2374

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_set_params(name, estimator_orig):
    # Check that get_params() returns the same thing
    # before and after set_params() with some fuzz
    estimator = clone(estimator_orig)

    orig_params = estimator.get_params(deep=False)
    msg = ("get_params result does not match what was passed to set_params")

    estimator.set_params(**orig_params)
    curr_params = estimator.get_params(deep=False)
    assert_equal(set(orig_params.keys()), set(curr_params.keys()), msg)
    for k, v in curr_params.items():
        assert orig_params[k] is v, msg

    # some fuzz values
    test_values = [-np.inf, np.inf, None]

    test_params = deepcopy(orig_params)
    for param_name in orig_params.keys():
        default_value = orig_params[param_name]
        for value in test_values:
            test_params[param_name] = value
            try:
                estimator.set_params(**test_params)
            except (TypeError, ValueError) as e:
                e_type = e.__class__.__name__
                # Exception occurred, possibly parameter validation
                warnings.warn("{0} occurred during set_params of param {1} on "
                              "{2}. It is recommended to delay parameter "
                              "validation until fit.".format(e_type,
                                                             param_name,
                                                             name))

                change_warning_msg = "Estimator's parameters changed after " \
                                     "set_params raised {}".format(e_type)
                params_before_exception = curr_params
                curr_params = estimator.get_params(deep=False)
                try:
                    assert_equal(set(params_before_exception.keys()),
                                 set(curr_params.keys()))
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                curr_params = estimator.get_params(deep=False)
                assert_equal(set(test_params.keys()),
                             set(curr_params.keys()),
                             msg)
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg
        test_params[param_name] = default_value
```
### 10 - sklearn/neighbors/nca.py:

Start line: 29, End line: 156

```python
class NeighborhoodComponentsAnalysis(BaseEstimator, TransformerMixin):
    """Neighborhood Components Analysis

    Neighborhood Component Analysis (NCA) is a machine learning algorithm for
    metric learning. It learns a linear transformation in a supervised fashion
    to improve the classification accuracy of a stochastic nearest neighbors
    rule in the transformed space.

    Read more in the :ref:`User Guide <nca>`.

    Parameters
    ----------
    n_components : int, optional (default=None)
        Preferred dimensionality of the projected space.
        If None it will be set to ``n_features``.

    init : string or numpy array, optional (default='auto')
        Initialization of the linear transformation. Possible options are
        'auto', 'pca', 'lda', 'identity', 'random', and a numpy array of shape
        (n_features_a, n_features_b).

        'auto'
            Depending on ``n_components``, the most reasonable initialization
            will be chosen. If ``n_components <= n_classes`` we use 'lda', as
            it uses labels information. If not, but
            ``n_components < min(n_features, n_samples)``, we use 'pca', as
            it projects data in meaningful directions (those of higher
            variance). Otherwise, we just use 'identity'.

        'pca'
            ``n_components`` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See `decomposition.PCA`)

        'lda'
            ``min(n_components, n_classes)`` most discriminative
            components of the inputs passed to :meth:`fit` will be used to
            initialize the transformation. (If ``n_components > n_classes``,
            the rest of the components will be zero.) (See
            `discriminant_analysis.LinearDiscriminantAnalysis`)

        'identity'
            If ``n_components`` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first ``n_components`` rows.

        'random'
            The initial transformation will be a random array of shape
            `(n_components, n_features)`. Each value is sampled from the
            standard normal distribution.

        numpy array
            n_features_b must match the dimensionality of the inputs passed to
            :meth:`fit` and n_features_a must be less than or equal to that.
            If ``n_components`` is not None, n_features_a must match it.

    warm_start : bool, optional, (default=False)
        If True and :meth:`fit` has been called before, the solution of the
        previous call to :meth:`fit` is used as the initial linear
        transformation (``n_components`` and ``init`` will be ignored).

    max_iter : int, optional (default=50)
        Maximum number of iterations in the optimization.

    tol : float, optional (default=1e-5)
        Convergence tolerance for the optimization.

    callback : callable, optional (default=None)
        If not None, this function is called after every iteration of the
        optimizer, taking as arguments the current solution (flattened
        transformation matrix) and the number of iterations. This might be
        useful in case one wants to examine or store the transformation
        found after each iteration.

    verbose : int, optional (default=0)
        If 0, no progress messages will be printed.
        If 1, progress messages will be printed to stdout.
        If > 1, progress messages will be printed and the ``disp``
        parameter of :func:`scipy.optimize.minimize` will be set to
        ``verbose - 2``.

    random_state : int or numpy.RandomState or None, optional (default=None)
        A pseudo random number generator object or a seed for it if int. If
        ``init='random'``, ``random_state`` is used to initialize the random
        transformation. If ``init='pca'``, ``random_state`` is passed as an
        argument to PCA when initializing the transformation.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        The linear transformation learned during fitting.

    n_iter_ : int
        Counts the number of iterations performed by the optimizer.

    Examples
    --------
    >>> from sklearn.neighbors.nca import NeighborhoodComponentsAnalysis
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ... stratify=y, test_size=0.7, random_state=42)
    >>> nca = NeighborhoodComponentsAnalysis(random_state=42)
    >>> nca.fit(X_train, y_train)
    NeighborhoodComponentsAnalysis(...)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(X_test, y_test))
    0.933333...
    >>> knn.fit(nca.transform(X_train), y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(nca.transform(X_test), y_test))
    0.961904...

    References
    ----------
    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
           "Neighbourhood Components Analysis". Advances in Neural Information
           Processing Systems. 17, 513-520, 2005.
           http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
           https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

    """
```
### 23 - sklearn/neighbors/nca.py:

Start line: 158, End line: 236

```python
class NeighborhoodComponentsAnalysis(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None, init='auto', warm_start=False,
                 max_iter=50, tol=1e-5, callback=None, verbose=0,
                 random_state=None):
        self.n_components = n_components
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training samples.

        y : array-like, shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        self : object
            returns a trained NeighborhoodComponentsAnalysis model.
        """

        # Verify inputs X and y and NCA parameters, and transform a copy if
        # needed
        X, y, init = self._validate_params(X, y)

        # Initialize the random generator
        self.random_state_ = check_random_state(self.random_state)

        # Measure the total training time
        t_train = time.time()

        # Compute a mask that stays fixed during optimization:
        same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]
        # (n_samples, n_samples)

        # Initialize the transformation
        transformation = self._initialize(X, y, init)

        # Create a dictionary of parameters to be passed to the optimizer
        disp = self.verbose - 2 if self.verbose > 1 else -1
        optimizer_params = {'method': 'L-BFGS-B',
                            'fun': self._loss_grad_lbfgs,
                            'args': (X, same_class_mask, -1.0),
                            'jac': True,
                            'x0': transformation,
                            'tol': self.tol,
                            'options': dict(maxiter=self.max_iter, disp=disp),
                            'callback': self._callback
                            }

        # Call the optimizer
        self.n_iter_ = 0
        opt_result = minimize(**optimizer_params)

        # Reshape the solution found by the optimizer
        self.components_ = opt_result.x.reshape(-1, X.shape[1])

        # Stop timer
        t_train = time.time() - t_train
        if self.verbose:
            cls_name = self.__class__.__name__

            # Warn the user if the algorithm did not converge
            if not opt_result.success:
                warn('[{}] NCA did not converge: {}'.format(
                    cls_name, opt_result.message),
                     ConvergenceWarning)

            print('[{}] Training took {:8.2f}s.'.format(cls_name, t_train))

        return self
```
### 24 - sklearn/neighbors/nca.py:

Start line: 1, End line: 26

```python
# coding: utf-8
"""
Neighborhood Component Analysis
"""

from __future__ import print_function

from warnings import warn
import numpy as np
import sys
import time
from scipy.optimize import minimize
from ..utils.extmath import softmax
from ..metrics import pairwise_distances
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import LabelEncoder
from ..decomposition import PCA
from ..utils.multiclass import check_classification_targets
from ..utils.random import check_random_state
from ..utils.validation import (check_is_fitted, check_array, check_X_y,
                                check_scalar)
from ..exceptions import ConvergenceWarning
```
