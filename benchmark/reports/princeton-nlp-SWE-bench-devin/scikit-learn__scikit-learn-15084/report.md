# scikit-learn__scikit-learn-15084

| **scikit-learn/scikit-learn** | `5e4b2757d61563889672e395d9e92d9372d357f6` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | - |
| **Any found context length** | 1180 |
| **Avg pos** | 50.0 |
| **Min pos** | 3 |
| **Max pos** | 69 |
| **Top file pos** | 3 |
| **Missing snippets** | 13 |
| **Missing patch files** | 2 |


## Expected patch

```diff
diff --git a/sklearn/ensemble/_stacking.py b/sklearn/ensemble/_stacking.py
--- a/sklearn/ensemble/_stacking.py
+++ b/sklearn/ensemble/_stacking.py
@@ -15,6 +15,7 @@
 from ..base import MetaEstimatorMixin
 
 from .base import _parallel_fit_estimator
+from .base import _BaseHeterogeneousEnsemble
 
 from ..linear_model import LogisticRegression
 from ..linear_model import RidgeCV
@@ -32,80 +33,26 @@
 from ..utils.validation import column_or_1d
 
 
-class _BaseStacking(TransformerMixin, MetaEstimatorMixin, _BaseComposition,
+class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble,
                     metaclass=ABCMeta):
     """Base class for stacking method."""
-    _required_parameters = ['estimators']
 
     @abstractmethod
     def __init__(self, estimators, final_estimator=None, cv=None,
                  stack_method='auto', n_jobs=None, verbose=0):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.final_estimator = final_estimator
         self.cv = cv
         self.stack_method = stack_method
         self.n_jobs = n_jobs
         self.verbose = verbose
 
-    @abstractmethod
-    def _validate_estimators(self):
-        if self.estimators is None or len(self.estimators) == 0:
-            raise ValueError(
-                "Invalid 'estimators' attribute, 'estimators' should be a list"
-                " of (string, estimator) tuples."
-            )
-        names, estimators = zip(*self.estimators)
-        self._validate_names(names)
-        return names, estimators
-
     def _clone_final_estimator(self, default):
         if self.final_estimator is not None:
             self.final_estimator_ = clone(self.final_estimator)
         else:
             self.final_estimator_ = clone(default)
 
-    def set_params(self, **params):
-        """Set the parameters for the stacking estimator.
-
-        Valid parameter keys can be listed with `get_params()`.
-
-        Parameters
-        ----------
-        params : keyword arguments
-            Specific parameters using e.g.
-            `set_params(parameter_name=new_value)`. In addition, to setting the
-            parameters of the stacking estimator, the individual estimator of
-            the stacking estimators can also be set, or can be removed by
-            setting them to 'drop'.
-
-        Examples
-        --------
-        In this example, the RandomForestClassifier is removed.
-
-        >>> from sklearn.linear_model import LogisticRegression
-        >>> from sklearn.ensemble import RandomForestClassifier
-        >>> from sklearn.ensemble import VotingClassifier
-        >>> clf1 = LogisticRegression()
-        >>> clf2 = RandomForestClassifier()
-        >>> eclf = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2)])
-        >>> eclf.set_params(rf='drop')
-        StackingClassifier(estimators=[('lr', LogisticRegression()),
-                                        ('rf', 'drop')])
-        """
-        super()._set_params('estimators', **params)
-        return self
-
-    def get_params(self, deep=True):
-        """Get the parameters of the stacking estimator.
-
-        Parameters
-        ----------
-        deep : bool
-            Setting it to True gets the various classifiers and the parameters
-            of the classifiers as well.
-        """
-        return super()._get_params('estimators', deep=deep)
-
     def _concatenate_predictions(self, predictions):
         """Concatenate the predictions of each first layer learner.
 
@@ -172,13 +119,6 @@ def fit(self, X, y, sample_weight=None):
         names, all_estimators = self._validate_estimators()
         self._validate_final_estimator()
 
-        has_estimator = any(est != 'drop' for est in all_estimators)
-        if not has_estimator:
-            raise ValueError(
-                "All estimators are dropped. At least one is required "
-                "to be an estimator."
-            )
-
         stack_method = [self.stack_method] * len(all_estimators)
 
         # Fit the base estimators on the whole training data. Those
@@ -416,16 +356,6 @@ def __init__(self, estimators, final_estimator=None, cv=None,
             verbose=verbose
         )
 
-    def _validate_estimators(self):
-        names, estimators = super()._validate_estimators()
-        for est in estimators:
-            if est != 'drop' and not is_classifier(est):
-                raise ValueError(
-                    "The estimator {} should be a classifier."
-                    .format(est.__class__.__name__)
-                )
-        return names, estimators
-
     def _validate_final_estimator(self):
         self._clone_final_estimator(default=LogisticRegression())
         if not is_classifier(self.final_estimator_):
@@ -651,16 +581,6 @@ def __init__(self, estimators, final_estimator=None, cv=None, n_jobs=None,
             verbose=verbose
         )
 
-    def _validate_estimators(self):
-        names, estimators = super()._validate_estimators()
-        for est in estimators:
-            if est != 'drop' and not is_regressor(est):
-                raise ValueError(
-                    "The estimator {} should be a regressor."
-                    .format(est.__class__.__name__)
-                )
-        return names, estimators
-
     def _validate_final_estimator(self):
         self._clone_final_estimator(default=RidgeCV())
         if not is_regressor(self.final_estimator_):
diff --git a/sklearn/ensemble/base.py b/sklearn/ensemble/base.py
--- a/sklearn/ensemble/base.py
+++ b/sklearn/ensemble/base.py
@@ -5,16 +5,20 @@
 # Authors: Gilles Louppe
 # License: BSD 3 clause
 
-import numpy as np
+from abc import ABCMeta, abstractmethod
 import numbers
 
+import numpy as np
+
 from joblib import effective_n_jobs
 
 from ..base import clone
+from ..base import is_classifier, is_regressor
 from ..base import BaseEstimator
 from ..base import MetaEstimatorMixin
+from ..utils import Bunch
 from ..utils import check_random_state
-from abc import ABCMeta, abstractmethod
+from ..utils.metaestimators import _BaseComposition
 
 MAX_RAND_SEED = np.iinfo(np.int32).max
 
@@ -178,3 +182,92 @@ def _partition_estimators(n_estimators, n_jobs):
     starts = np.cumsum(n_estimators_per_job)
 
     return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
+
+
+class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition,
+                                 metaclass=ABCMeta):
+    """Base class for heterogeneous ensemble of learners.
+
+    Parameters
+    ----------
+    estimators : list of (str, estimator) tuples
+        The ensemble of estimators to use in the ensemble. Each element of the
+        list is defined as a tuple of string (i.e. name of the estimator) and
+        an estimator instance. An estimator can be set to `'drop'` using
+        `set_params`.
+
+    Attributes
+    ----------
+    estimators_ : list of estimators
+        The elements of the estimators parameter, having been fitted on the
+        training data. If an estimator has been set to `'drop'`, it will not
+        appear in `estimators_`.
+    """
+    _required_parameters = ['estimators']
+
+    @property
+    def named_estimators(self):
+        return Bunch(**dict(self.estimators))
+
+    @abstractmethod
+    def __init__(self, estimators):
+        self.estimators = estimators
+
+    def _validate_estimators(self):
+        if self.estimators is None or len(self.estimators) == 0:
+            raise ValueError(
+                "Invalid 'estimators' attribute, 'estimators' should be a list"
+                " of (string, estimator) tuples."
+            )
+        names, estimators = zip(*self.estimators)
+        # defined by MetaEstimatorMixin
+        self._validate_names(names)
+
+        has_estimator = any(est not in (None, 'drop') for est in estimators)
+        if not has_estimator:
+            raise ValueError(
+                "All estimators are dropped. At least one is required "
+                "to be an estimator."
+            )
+
+        is_estimator_type = (is_classifier if is_classifier(self)
+                             else is_regressor)
+
+        for est in estimators:
+            if est not in (None, 'drop') and not is_estimator_type(est):
+                raise ValueError(
+                    "The estimator {} should be a {}."
+                    .format(
+                        est.__class__.__name__, is_estimator_type.__name__[3:]
+                    )
+                )
+
+        return names, estimators
+
+    def set_params(self, **params):
+        """Set the parameters of an estimator from the ensemble.
+
+        Valid parameter keys can be listed with `get_params()`.
+
+        Parameters
+        ----------
+        **params : keyword arguments
+            Specific parameters using e.g.
+            `set_params(parameter_name=new_value)`. In addition, to setting the
+            parameters of the stacking estimator, the individual estimator of
+            the stacking estimators can also be set, or can be removed by
+            setting them to 'drop'.
+        """
+        super()._set_params('estimators', **params)
+        return self
+
+    def get_params(self, deep=True):
+        """Get the parameters of an estimator from the ensemble.
+
+        Parameters
+        ----------
+        deep : bool
+            Setting it to True gets the various classifiers and the parameters
+            of the classifiers as well.
+        """
+        return super()._get_params('estimators', deep=deep)
diff --git a/sklearn/ensemble/voting.py b/sklearn/ensemble/voting.py
--- a/sklearn/ensemble/voting.py
+++ b/sklearn/ensemble/voting.py
@@ -24,25 +24,20 @@
 from ..base import TransformerMixin
 from ..base import clone
 from .base import _parallel_fit_estimator
+from .base import _BaseHeterogeneousEnsemble
 from ..preprocessing import LabelEncoder
 from ..utils import Bunch
 from ..utils.validation import check_is_fitted
-from ..utils.metaestimators import _BaseComposition
 from ..utils.multiclass import check_classification_targets
 from ..utils.validation import column_or_1d
 
 
-class _BaseVoting(TransformerMixin, _BaseComposition):
+class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
     """Base class for voting.
 
     Warning: This class should not be used directly. Use derived classes
     instead.
     """
-    _required_parameters = ['estimators']
-
-    @property
-    def named_estimators(self):
-        return Bunch(**dict(self.estimators))
 
     @property
     def _weights_not_none(self):
@@ -61,10 +56,7 @@ def fit(self, X, y, sample_weight=None):
         """
         common fit operations.
         """
-        if self.estimators is None or len(self.estimators) == 0:
-            raise AttributeError('Invalid `estimators` attribute, `estimators`'
-                                 ' should be a list of (string, estimator)'
-                                 ' tuples')
+        names, clfs = self._validate_estimators()
 
         if (self.weights is not None and
                 len(self.weights) != len(self.estimators)):
@@ -72,17 +64,6 @@ def fit(self, X, y, sample_weight=None):
                              '; got %d weights, %d estimators'
                              % (len(self.weights), len(self.estimators)))
 
-        names, clfs = zip(*self.estimators)
-        self._validate_names(names)
-
-        n_isnone = np.sum(
-            [clf in (None, 'drop') for _, clf in self.estimators]
-        )
-        if n_isnone == len(self.estimators):
-            raise ValueError(
-                'All estimators are None or "drop". At least one is required!'
-            )
-
         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                 delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                                  sample_weight=sample_weight)
@@ -94,46 +75,6 @@ def fit(self, X, y, sample_weight=None):
             self.named_estimators_[k[0]] = e
         return self
 
-    def set_params(self, **params):
-        """ Setting the parameters for the ensemble estimator
-
-        Valid parameter keys can be listed with get_params().
-
-        Parameters
-        ----------
-        **params : keyword arguments
-            Specific parameters using e.g. set_params(parameter_name=new_value)
-            In addition, to setting the parameters of the ensemble estimator,
-            the individual estimators of the ensemble estimator can also be
-            set or replaced by setting them to None.
-
-        Examples
-        --------
-        In this example, the RandomForestClassifier is removed.
-
-        >>> from sklearn.linear_model import LogisticRegression
-        >>> from sklearn.ensemble import RandomForestClassifier
-        >>> from sklearn.ensemble import VotingClassifier
-        >>> clf1 = LogisticRegression()
-        >>> clf2 = RandomForestClassifier()
-        >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)])
-        >>> eclf.set_params(rf=None)
-        VotingClassifier(estimators=[('lr', LogisticRegression()),
-                                     ('rf', None)])
-        """
-        return self._set_params('estimators', **params)
-
-    def get_params(self, deep=True):
-        """ Get the parameters of the ensemble estimator
-
-        Parameters
-        ----------
-        deep : bool
-            Setting it to True gets the various estimators and the parameters
-            of the estimators as well
-        """
-        return self._get_params('estimators', deep=deep)
-
 
 class VotingClassifier(ClassifierMixin, _BaseVoting):
     """Soft Voting/Majority Rule classifier for unfitted estimators.
@@ -230,7 +171,7 @@ class VotingClassifier(ClassifierMixin, _BaseVoting):
 
     def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                  flatten_transform=True):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.voting = voting
         self.weights = weights
         self.n_jobs = n_jobs
@@ -423,7 +364,7 @@ class VotingRegressor(RegressorMixin, _BaseVoting):
     """
 
     def __init__(self, estimators, weights=None, n_jobs=None):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.weights = weights
         self.n_jobs = n_jobs
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/ensemble/_stacking.py | 18 | 18 | - | - | -
| sklearn/ensemble/_stacking.py | 35 | 108 | - | - | -
| sklearn/ensemble/_stacking.py | 175 | 181 | - | - | -
| sklearn/ensemble/_stacking.py | 419 | 428 | - | - | -
| sklearn/ensemble/_stacking.py | 654 | 663 | - | - | -
| sklearn/ensemble/base.py | 8 | 17 | - | - | -
| sklearn/ensemble/base.py | 181 | 181 | - | - | -
| sklearn/ensemble/voting.py | 27 | 42 | - | 3 | -
| sklearn/ensemble/voting.py | 64 | 67 | 69 | 3 | 37923
| sklearn/ensemble/voting.py | 75 | 85 | 69 | 3 | 37923
| sklearn/ensemble/voting.py | 97 | 136 | - | 3 | -
| sklearn/ensemble/voting.py | 233 | 233 | 3 | 3 | 1180
| sklearn/ensemble/voting.py | 426 | 426 | 9 | 3 | 4802


## Problem Statement

```
VotingClassifier and roc_auc TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' and
#### Description
VotingClassifier
TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'

#### Steps/Code to Reproduce
\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pipePre = Pipeline([
    ('simpleimputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ('standardscaler', StandardScaler()),
    ('normalizer', Normalizer())
     ])

df_train_x = pipePre.fit_transform(df_train_x)

X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size = 0.25, random_state=42)

lrg = LinearRegression().fit(X_train, y_train)

rig = Ridge().fit(X_train, y_train)

lreg = LogisticRegression().fit(X_train, y_train)

voting = VotingClassifier(estimators=[('lrg_v', lrg), ('rig_v', rig), 
                                      ('lreg_v', lreg)], voting='hard')
voting_fit = voting.fit(X_train, y_train)

y_pred = voting_fit.predict(X_test)
roc_auc_score(y_test, y_pred)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-50-506a80086b81> in <module>
----> 1 val_error(voting_fit, X_test, y_test)

<ipython-input-6-0fa46ec754f8> in val_error(model, tested, prediction)
     14         Data, prepaired as tested labels
     15     """
---> 16     y_pred = model.predict(tested)
     17     err = roc_auc_score(prediction, y_pred)
     18     return err

~\Anaconda3\lib\site-packages\sklearn\ensemble\voting.py in predict(self, X)
    302                 lambda x: np.argmax(
    303                     np.bincount(x, weights=self._weights_not_none)),
--> 304                 axis=1, arr=predictions)
    305 
    306         maj = self.le_.inverse_transform(maj)

~\Anaconda3\lib\site-packages\numpy\lib\shape_base.py in apply_along_axis(func1d, axis, arr, *args, **kwargs)
    378     except StopIteration:
    379         raise ValueError('Cannot apply_along_axis when any iteration dimensions are 0')
--> 380     res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
    381 
    382     # build a buffer for storing evaluations of func1d.

~\Anaconda3\lib\site-packages\sklearn\ensemble\voting.py in <lambda>(x)
    301             maj = np.apply_along_axis(
    302                 lambda x: np.argmax(
--> 303                     np.bincount(x, weights=self._weights_not_none)),
    304                 axis=1, arr=predictions)
    305 

TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'

\`\`\`

scikit-learn  0.21.2  anaconda


<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/preprocessing/plot_scaling_importance.py | 82 | 134| 457 | 457 | 1219 | 
| 2 | 2 examples/ensemble/plot_voting_regressor.py | 1 | 56| 374 | 831 | 1593 | 
| **-> 3 <-** | **3 sklearn/ensemble/voting.py** | 231 | 273| 349 | 1180 | 5521 | 
| 4 | 4 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 59 | 124| 700 | 1880 | 6703 | 
| 5 | 5 examples/model_selection/plot_roc_crossval.py | 1 | 89| 759 | 2639 | 7590 | 
| 6 | 6 sklearn/metrics/ranking.py | 347 | 375| 366 | 3005 | 20505 | 
| 7 | 7 sklearn/utils/estimator_checks.py | 1988 | 2029| 417 | 3422 | 45038 | 
| 8 | 8 examples/model_selection/plot_roc.py | 1 | 91| 769 | 4191 | 46607 | 
| **-> 9 <-** | **8 sklearn/ensemble/voting.py** | 368 | 428| 611 | 4802 | 46607 | 
| 10 | 9 examples/ensemble/plot_voting_probas.py | 1 | 87| 809 | 5611 | 47416 | 
| 11 | 10 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 775 | 6386 | 49236 | 
| 12 | 10 examples/model_selection/plot_roc.py | 92 | 173| 800 | 7186 | 49236 | 
| 13 | 10 sklearn/utils/estimator_checks.py | 1 | 61| 447 | 7633 | 49236 | 
| 14 | 11 examples/ensemble/plot_feature_transformation.py | 88 | 121| 366 | 7999 | 50384 | 
| 15 | **11 sklearn/ensemble/voting.py** | 430 | 452| 176 | 8175 | 50384 | 
| 16 | 11 sklearn/utils/estimator_checks.py | 1712 | 1811| 918 | 9093 | 50384 | 
| 17 | 12 examples/model_selection/plot_precision_recall.py | 101 | 205| 770 | 9863 | 52726 | 
| 18 | **12 sklearn/ensemble/voting.py** | 138 | 229| 980 | 10843 | 52726 | 
| 19 | **12 sklearn/ensemble/voting.py** | 304 | 332| 226 | 11069 | 52726 | 
| 20 | 13 examples/ensemble/plot_voting_decision_regions.py | 1 | 78| 677 | 11746 | 53403 | 
| 21 | 14 sklearn/metrics/scorer.py | 613 | 680| 735 | 12481 | 59478 | 
| 22 | 15 benchmarks/bench_saga.py | 107 | 188| 632 | 13113 | 61937 | 
| 23 | **15 sklearn/ensemble/voting.py** | 275 | 302| 180 | 13293 | 61937 | 
| 24 | 16 sklearn/model_selection/_validation.py | 723 | 780| 592 | 13885 | 75357 | 
| 25 | 17 benchmarks/bench_lof.py | 36 | 107| 650 | 14535 | 76273 | 
| 26 | 18 sklearn/metrics/__init__.py | 1 | 81| 631 | 15166 | 77371 | 
| 27 | 18 sklearn/metrics/ranking.py | 246 | 345| 966 | 16132 | 77371 | 
| 28 | 18 sklearn/model_selection/_validation.py | 495 | 569| 699 | 16831 | 77371 | 
| 29 | 19 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 17116 | 78165 | 
| 30 | 19 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 17870 | 78165 | 
| 31 | 20 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 140| 524 | 18394 | 79290 | 
| 32 | 21 examples/model_selection/plot_grid_search_refit_callable.py | 78 | 117| 336 | 18730 | 80135 | 
| 33 | 22 examples/text/plot_document_classification_20newsgroups.py | 249 | 329| 673 | 19403 | 82735 | 
| 34 | **22 sklearn/ensemble/voting.py** | 454 | 490| 225 | 19628 | 82735 | 
| 35 | 23 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 20261 | 83461 | 
| 36 | 24 sklearn/utils/validation.py | 724 | 747| 276 | 20537 | 92701 | 
| 37 | 25 examples/ensemble/plot_stack_predictors.py | 52 | 124| 610 | 21147 | 93745 | 
| 38 | 26 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 86| 769 | 21916 | 94898 | 
| 39 | 27 sklearn/feature_selection/rfe.py | 9 | 23| 127 | 22043 | 99491 | 
| 40 | 27 sklearn/utils/estimator_checks.py | 1501 | 1556| 545 | 22588 | 99491 | 
| 41 | 28 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 23381 | 100736 | 
| 42 | 28 sklearn/metrics/scorer.py | 683 | 721| 489 | 23870 | 100736 | 
| 43 | 28 sklearn/metrics/__init__.py | 84 | 154| 466 | 24336 | 100736 | 
| 44 | 29 examples/linear_model/plot_lasso_model_selection.py | 94 | 158| 513 | 24849 | 102101 | 
| 45 | 30 sklearn/model_selection/__init__.py | 1 | 60| 405 | 25254 | 102506 | 
| 46 | **30 sklearn/ensemble/voting.py** | 1 | 32| 155 | 25409 | 102506 | 
| 47 | 30 sklearn/metrics/ranking.py | 378 | 458| 729 | 26138 | 102506 | 
| 48 | 31 sklearn/metrics/regression.py | 546 | 593| 476 | 26614 | 109560 | 
| 49 | 32 examples/linear_model/plot_ols_ridge_variance.py | 1 | 68| 484 | 27098 | 110052 | 
| 50 | 33 examples/decomposition/plot_pca_vs_fa_model_selection.py | 1 | 85| 641 | 27739 | 111153 | 
| 51 | 34 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 776 | 1057| 406 | 28145 | 120985 | 
| 52 | 35 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 61| 494 | 28639 | 122000 | 
| 53 | 35 sklearn/utils/estimator_checks.py | 2032 | 2077| 493 | 29132 | 122000 | 
| 54 | 36 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 29916 | 125108 | 
| 55 | 36 sklearn/metrics/ranking.py | 460 | 475| 204 | 30120 | 125108 | 
| 56 | 37 examples/model_selection/plot_cv_predict.py | 1 | 29| 193 | 30313 | 125301 | 
| 57 | 38 examples/compose/plot_digits_pipe.py | 1 | 77| 559 | 30872 | 125868 | 
| 58 | 39 doc/tutorial/text_analytics/skeletons/exercise_02_sentiment.py | 14 | 64| 441 | 31313 | 126402 | 
| 59 | 40 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 31845 | 128347 | 
| 60 | 40 sklearn/utils/estimator_checks.py | 1834 | 1906| 655 | 32500 | 128347 | 
| 61 | 40 sklearn/metrics/ranking.py | 741 | 784| 498 | 32998 | 128347 | 
| 62 | 41 sklearn/neighbors/classification.py | 391 | 450| 530 | 33528 | 133352 | 
| 63 | 42 benchmarks/bench_isolation_forest.py | 54 | 160| 1025 | 34553 | 134813 | 
| 64 | 42 sklearn/utils/estimator_checks.py | 2641 | 2672| 340 | 34893 | 134813 | 
| 65 | 43 examples/calibration/plot_compare_calibration.py | 1 | 78| 761 | 35654 | 135992 | 
| 66 | 43 sklearn/utils/estimator_checks.py | 2179 | 2228| 498 | 36152 | 135992 | 
| 67 | 44 sklearn/calibration.py | 109 | 201| 784 | 36936 | 140886 | 
| 68 | 45 examples/inspection/plot_permutation_importance.py | 106 | 179| 664 | 37600 | 142472 | 
| **-> 69 <-** | **45 sklearn/ensemble/voting.py** | 59 | 95| 323 | 37923 | 142472 | 
| 70 | 45 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 38296 | 142472 | 
| 71 | 46 sklearn/metrics/classification.py | 1845 | 1904| 597 | 38893 | 164422 | 
| 72 | 47 examples/plot_kernel_ridge_regression.py | 80 | 153| 738 | 39631 | 166134 | 
| 73 | 48 sklearn/linear_model/ransac.py | 327 | 421| 797 | 40428 | 170173 | 
| 74 | 49 examples/linear_model/plot_ransac.py | 1 | 60| 455 | 40883 | 170628 | 
| 75 | 50 sklearn/linear_model/omp.py | 856 | 908| 508 | 41391 | 178684 | 
| 76 | 50 benchmarks/bench_saga.py | 6 | 104| 835 | 42226 | 178684 | 
| 77 | 50 sklearn/utils/estimator_checks.py | 2386 | 2401| 266 | 42492 | 178684 | 
| 78 | 50 sklearn/model_selection/_validation.py | 220 | 259| 405 | 42897 | 178684 | 
| 79 | 51 sklearn/tree/tree.py | 376 | 392| 181 | 43078 | 193446 | 
| 80 | 51 sklearn/metrics/classification.py | 1905 | 1937| 316 | 43394 | 193446 | 
| 81 | 51 examples/calibration/plot_compare_calibration.py | 79 | 123| 392 | 43786 | 193446 | 
| 82 | 52 benchmarks/bench_covertype.py | 112 | 190| 757 | 44543 | 195324 | 
| 83 | 53 examples/plot_anomaly_comparison.py | 81 | 152| 757 | 45300 | 196842 | 
| 84 | 53 sklearn/utils/estimator_checks.py | 1690 | 1710| 237 | 45537 | 196842 | 
| 85 | 54 examples/model_selection/plot_cv_indices.py | 105 | 151| 388 | 45925 | 198164 | 


## Missing Patch Files

 * 1: sklearn/ensemble/_stacking.py
 * 2: sklearn/ensemble/base.py
 * 3: sklearn/ensemble/voting.py

### Hint

```
`Ridge` and `LinearRegression` are not classifiers, which makes them incompatible with `VotingClassifier`.
> Ridge and LinearRegression are not classifiers, which makes them incompatible with VotingClassifier.

+1 though maybe we should return a better error message.
Shall we check the base estimators with `sklearn.base.is_classifier` in the `VotingClassifier.__init__` or `fit` and raise a `ValueError`? 
> Shall we check the base estimators with sklearn.base.is_classifier in the VotingClassifier.__init__ or fit and raise a ValueError?

We have something similar for the StackingClassifier and StackingRegressor.

Actually, these 2 classes shared something in common by having `estimators` parameters. In some way they are an ensemble of "heterogeneous" estimators (i.e. multiple estimators) while bagging, RF, GBDT is an ensemble of "homogeneous" estimator (i.e. single "base_estimator").

We could have a separate base class for each type. Not sure about the naming but it will reduce code redundancy and make checking easier and consistent.
```

## Patch

```diff
diff --git a/sklearn/ensemble/_stacking.py b/sklearn/ensemble/_stacking.py
--- a/sklearn/ensemble/_stacking.py
+++ b/sklearn/ensemble/_stacking.py
@@ -15,6 +15,7 @@
 from ..base import MetaEstimatorMixin
 
 from .base import _parallel_fit_estimator
+from .base import _BaseHeterogeneousEnsemble
 
 from ..linear_model import LogisticRegression
 from ..linear_model import RidgeCV
@@ -32,80 +33,26 @@
 from ..utils.validation import column_or_1d
 
 
-class _BaseStacking(TransformerMixin, MetaEstimatorMixin, _BaseComposition,
+class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble,
                     metaclass=ABCMeta):
     """Base class for stacking method."""
-    _required_parameters = ['estimators']
 
     @abstractmethod
     def __init__(self, estimators, final_estimator=None, cv=None,
                  stack_method='auto', n_jobs=None, verbose=0):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.final_estimator = final_estimator
         self.cv = cv
         self.stack_method = stack_method
         self.n_jobs = n_jobs
         self.verbose = verbose
 
-    @abstractmethod
-    def _validate_estimators(self):
-        if self.estimators is None or len(self.estimators) == 0:
-            raise ValueError(
-                "Invalid 'estimators' attribute, 'estimators' should be a list"
-                " of (string, estimator) tuples."
-            )
-        names, estimators = zip(*self.estimators)
-        self._validate_names(names)
-        return names, estimators
-
     def _clone_final_estimator(self, default):
         if self.final_estimator is not None:
             self.final_estimator_ = clone(self.final_estimator)
         else:
             self.final_estimator_ = clone(default)
 
-    def set_params(self, **params):
-        """Set the parameters for the stacking estimator.
-
-        Valid parameter keys can be listed with `get_params()`.
-
-        Parameters
-        ----------
-        params : keyword arguments
-            Specific parameters using e.g.
-            `set_params(parameter_name=new_value)`. In addition, to setting the
-            parameters of the stacking estimator, the individual estimator of
-            the stacking estimators can also be set, or can be removed by
-            setting them to 'drop'.
-
-        Examples
-        --------
-        In this example, the RandomForestClassifier is removed.
-
-        >>> from sklearn.linear_model import LogisticRegression
-        >>> from sklearn.ensemble import RandomForestClassifier
-        >>> from sklearn.ensemble import VotingClassifier
-        >>> clf1 = LogisticRegression()
-        >>> clf2 = RandomForestClassifier()
-        >>> eclf = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2)])
-        >>> eclf.set_params(rf='drop')
-        StackingClassifier(estimators=[('lr', LogisticRegression()),
-                                        ('rf', 'drop')])
-        """
-        super()._set_params('estimators', **params)
-        return self
-
-    def get_params(self, deep=True):
-        """Get the parameters of the stacking estimator.
-
-        Parameters
-        ----------
-        deep : bool
-            Setting it to True gets the various classifiers and the parameters
-            of the classifiers as well.
-        """
-        return super()._get_params('estimators', deep=deep)
-
     def _concatenate_predictions(self, predictions):
         """Concatenate the predictions of each first layer learner.
 
@@ -172,13 +119,6 @@ def fit(self, X, y, sample_weight=None):
         names, all_estimators = self._validate_estimators()
         self._validate_final_estimator()
 
-        has_estimator = any(est != 'drop' for est in all_estimators)
-        if not has_estimator:
-            raise ValueError(
-                "All estimators are dropped. At least one is required "
-                "to be an estimator."
-            )
-
         stack_method = [self.stack_method] * len(all_estimators)
 
         # Fit the base estimators on the whole training data. Those
@@ -416,16 +356,6 @@ def __init__(self, estimators, final_estimator=None, cv=None,
             verbose=verbose
         )
 
-    def _validate_estimators(self):
-        names, estimators = super()._validate_estimators()
-        for est in estimators:
-            if est != 'drop' and not is_classifier(est):
-                raise ValueError(
-                    "The estimator {} should be a classifier."
-                    .format(est.__class__.__name__)
-                )
-        return names, estimators
-
     def _validate_final_estimator(self):
         self._clone_final_estimator(default=LogisticRegression())
         if not is_classifier(self.final_estimator_):
@@ -651,16 +581,6 @@ def __init__(self, estimators, final_estimator=None, cv=None, n_jobs=None,
             verbose=verbose
         )
 
-    def _validate_estimators(self):
-        names, estimators = super()._validate_estimators()
-        for est in estimators:
-            if est != 'drop' and not is_regressor(est):
-                raise ValueError(
-                    "The estimator {} should be a regressor."
-                    .format(est.__class__.__name__)
-                )
-        return names, estimators
-
     def _validate_final_estimator(self):
         self._clone_final_estimator(default=RidgeCV())
         if not is_regressor(self.final_estimator_):
diff --git a/sklearn/ensemble/base.py b/sklearn/ensemble/base.py
--- a/sklearn/ensemble/base.py
+++ b/sklearn/ensemble/base.py
@@ -5,16 +5,20 @@
 # Authors: Gilles Louppe
 # License: BSD 3 clause
 
-import numpy as np
+from abc import ABCMeta, abstractmethod
 import numbers
 
+import numpy as np
+
 from joblib import effective_n_jobs
 
 from ..base import clone
+from ..base import is_classifier, is_regressor
 from ..base import BaseEstimator
 from ..base import MetaEstimatorMixin
+from ..utils import Bunch
 from ..utils import check_random_state
-from abc import ABCMeta, abstractmethod
+from ..utils.metaestimators import _BaseComposition
 
 MAX_RAND_SEED = np.iinfo(np.int32).max
 
@@ -178,3 +182,92 @@ def _partition_estimators(n_estimators, n_jobs):
     starts = np.cumsum(n_estimators_per_job)
 
     return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
+
+
+class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition,
+                                 metaclass=ABCMeta):
+    """Base class for heterogeneous ensemble of learners.
+
+    Parameters
+    ----------
+    estimators : list of (str, estimator) tuples
+        The ensemble of estimators to use in the ensemble. Each element of the
+        list is defined as a tuple of string (i.e. name of the estimator) and
+        an estimator instance. An estimator can be set to `'drop'` using
+        `set_params`.
+
+    Attributes
+    ----------
+    estimators_ : list of estimators
+        The elements of the estimators parameter, having been fitted on the
+        training data. If an estimator has been set to `'drop'`, it will not
+        appear in `estimators_`.
+    """
+    _required_parameters = ['estimators']
+
+    @property
+    def named_estimators(self):
+        return Bunch(**dict(self.estimators))
+
+    @abstractmethod
+    def __init__(self, estimators):
+        self.estimators = estimators
+
+    def _validate_estimators(self):
+        if self.estimators is None or len(self.estimators) == 0:
+            raise ValueError(
+                "Invalid 'estimators' attribute, 'estimators' should be a list"
+                " of (string, estimator) tuples."
+            )
+        names, estimators = zip(*self.estimators)
+        # defined by MetaEstimatorMixin
+        self._validate_names(names)
+
+        has_estimator = any(est not in (None, 'drop') for est in estimators)
+        if not has_estimator:
+            raise ValueError(
+                "All estimators are dropped. At least one is required "
+                "to be an estimator."
+            )
+
+        is_estimator_type = (is_classifier if is_classifier(self)
+                             else is_regressor)
+
+        for est in estimators:
+            if est not in (None, 'drop') and not is_estimator_type(est):
+                raise ValueError(
+                    "The estimator {} should be a {}."
+                    .format(
+                        est.__class__.__name__, is_estimator_type.__name__[3:]
+                    )
+                )
+
+        return names, estimators
+
+    def set_params(self, **params):
+        """Set the parameters of an estimator from the ensemble.
+
+        Valid parameter keys can be listed with `get_params()`.
+
+        Parameters
+        ----------
+        **params : keyword arguments
+            Specific parameters using e.g.
+            `set_params(parameter_name=new_value)`. In addition, to setting the
+            parameters of the stacking estimator, the individual estimator of
+            the stacking estimators can also be set, or can be removed by
+            setting them to 'drop'.
+        """
+        super()._set_params('estimators', **params)
+        return self
+
+    def get_params(self, deep=True):
+        """Get the parameters of an estimator from the ensemble.
+
+        Parameters
+        ----------
+        deep : bool
+            Setting it to True gets the various classifiers and the parameters
+            of the classifiers as well.
+        """
+        return super()._get_params('estimators', deep=deep)
diff --git a/sklearn/ensemble/voting.py b/sklearn/ensemble/voting.py
--- a/sklearn/ensemble/voting.py
+++ b/sklearn/ensemble/voting.py
@@ -24,25 +24,20 @@
 from ..base import TransformerMixin
 from ..base import clone
 from .base import _parallel_fit_estimator
+from .base import _BaseHeterogeneousEnsemble
 from ..preprocessing import LabelEncoder
 from ..utils import Bunch
 from ..utils.validation import check_is_fitted
-from ..utils.metaestimators import _BaseComposition
 from ..utils.multiclass import check_classification_targets
 from ..utils.validation import column_or_1d
 
 
-class _BaseVoting(TransformerMixin, _BaseComposition):
+class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
     """Base class for voting.
 
     Warning: This class should not be used directly. Use derived classes
     instead.
     """
-    _required_parameters = ['estimators']
-
-    @property
-    def named_estimators(self):
-        return Bunch(**dict(self.estimators))
 
     @property
     def _weights_not_none(self):
@@ -61,10 +56,7 @@ def fit(self, X, y, sample_weight=None):
         """
         common fit operations.
         """
-        if self.estimators is None or len(self.estimators) == 0:
-            raise AttributeError('Invalid `estimators` attribute, `estimators`'
-                                 ' should be a list of (string, estimator)'
-                                 ' tuples')
+        names, clfs = self._validate_estimators()
 
         if (self.weights is not None and
                 len(self.weights) != len(self.estimators)):
@@ -72,17 +64,6 @@ def fit(self, X, y, sample_weight=None):
                              '; got %d weights, %d estimators'
                              % (len(self.weights), len(self.estimators)))
 
-        names, clfs = zip(*self.estimators)
-        self._validate_names(names)
-
-        n_isnone = np.sum(
-            [clf in (None, 'drop') for _, clf in self.estimators]
-        )
-        if n_isnone == len(self.estimators):
-            raise ValueError(
-                'All estimators are None or "drop". At least one is required!'
-            )
-
         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                 delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                                  sample_weight=sample_weight)
@@ -94,46 +75,6 @@ def fit(self, X, y, sample_weight=None):
             self.named_estimators_[k[0]] = e
         return self
 
-    def set_params(self, **params):
-        """ Setting the parameters for the ensemble estimator
-
-        Valid parameter keys can be listed with get_params().
-
-        Parameters
-        ----------
-        **params : keyword arguments
-            Specific parameters using e.g. set_params(parameter_name=new_value)
-            In addition, to setting the parameters of the ensemble estimator,
-            the individual estimators of the ensemble estimator can also be
-            set or replaced by setting them to None.
-
-        Examples
-        --------
-        In this example, the RandomForestClassifier is removed.
-
-        >>> from sklearn.linear_model import LogisticRegression
-        >>> from sklearn.ensemble import RandomForestClassifier
-        >>> from sklearn.ensemble import VotingClassifier
-        >>> clf1 = LogisticRegression()
-        >>> clf2 = RandomForestClassifier()
-        >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)])
-        >>> eclf.set_params(rf=None)
-        VotingClassifier(estimators=[('lr', LogisticRegression()),
-                                     ('rf', None)])
-        """
-        return self._set_params('estimators', **params)
-
-    def get_params(self, deep=True):
-        """ Get the parameters of the ensemble estimator
-
-        Parameters
-        ----------
-        deep : bool
-            Setting it to True gets the various estimators and the parameters
-            of the estimators as well
-        """
-        return self._get_params('estimators', deep=deep)
-
 
 class VotingClassifier(ClassifierMixin, _BaseVoting):
     """Soft Voting/Majority Rule classifier for unfitted estimators.
@@ -230,7 +171,7 @@ class VotingClassifier(ClassifierMixin, _BaseVoting):
 
     def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                  flatten_transform=True):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.voting = voting
         self.weights = weights
         self.n_jobs = n_jobs
@@ -423,7 +364,7 @@ class VotingRegressor(RegressorMixin, _BaseVoting):
     """
 
     def __init__(self, estimators, weights=None, n_jobs=None):
-        self.estimators = estimators
+        super().__init__(estimators=estimators)
         self.weights = weights
         self.n_jobs = n_jobs
 

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_voting.py b/sklearn/ensemble/tests/test_voting.py
--- a/sklearn/ensemble/tests/test_voting.py
+++ b/sklearn/ensemble/tests/test_voting.py
@@ -37,9 +37,9 @@
 
 def test_estimator_init():
     eclf = VotingClassifier(estimators=[])
-    msg = ('Invalid `estimators` attribute, `estimators` should be'
-           ' a list of (string, estimator) tuples')
-    assert_raise_message(AttributeError, msg, eclf.fit, X, y)
+    msg = ("Invalid 'estimators' attribute, 'estimators' should be"
+           " a list of (string, estimator) tuples.")
+    assert_raise_message(ValueError, msg, eclf.fit, X, y)
 
     clf = LogisticRegression(random_state=1)
 
@@ -417,7 +417,7 @@ def test_set_estimator_none(drop):
     eclf2.set_params(voting='soft').fit(X, y)
     assert_array_equal(eclf1.predict(X), eclf2.predict(X))
     assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
-    msg = 'All estimators are None or "drop". At least one is required!'
+    msg = 'All estimators are dropped. At least one is required'
     assert_raise_message(
         ValueError, msg, eclf2.set_params(lr=drop, rf=drop, nb=drop).fit, X, y)
 

```


## Code snippets

### 1 - examples/preprocessing/plot_scaling_importance.py:

Start line: 82, End line: 134

```python
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal components
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Use PCA without and with scale on X_train data for visualization.
X_train_transformed = pca.transform(X_train)
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Training dataset after PCA')
ax2.set_title('Standardized training dataset after PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()
```
### 2 - examples/ensemble/plot_voting_regressor.py:

Start line: 1, End line: 56

```python
"""
=================================================
Plot individual and voting regression predictions
=================================================

.. currentmodule:: sklearn

Plot individual and averaged regression predictions for Boston dataset.

First, three exemplary regressors are initialized
(:class:`~ensemble.GradientBoostingRegressor`,
:class:`~ensemble.RandomForestRegressor`, and
:class:`~linear_model.LinearRegression`) and used to initialize a
:class:`~ensemble.VotingRegressor`.

The red starred dots are the averaged predictions.

"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

# Loading some example data
X, y = datasets.load_boston(return_X_y=True)

# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
reg3 = LinearRegression()
ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)
ereg.fit(X, y)

xt = X[:20]

plt.figure()
plt.plot(reg1.predict(xt), 'gd', label='GradientBoostingRegressor')
plt.plot(reg2.predict(xt), 'b^', label='RandomForestRegressor')
plt.plot(reg3.predict(xt), 'ys', label='LinearRegression')
plt.plot(ereg.predict(xt), 'r*', label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()
```
### 3 - sklearn/ensemble/voting.py:

Start line: 231, End line: 273

```python
class VotingClassifier(ClassifierMixin, _BaseVoting):

    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=True):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        return super().fit(X, transformed_y, sample_weight)
```
### 4 - benchmarks/bench_hist_gradient_boosting_higgsboson.py:

Start line: 59, End line: 124

```python
df = load_data()
target = df.values[:, 0]
data = np.ascontiguousarray(df.values[:, 1:])
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.2, random_state=0)

if subsample is not None:
    data_train, target_train = data_train[:subsample], target_train[:subsample]

n_samples, n_features = data_train.shape
print(f"Training set with {n_samples} records with {n_features} features.")

print("Fitting a sklearn model...")
tic = time()
est = HistGradientBoostingClassifier(loss='binary_crossentropy',
                                     learning_rate=lr,
                                     max_iter=n_trees,
                                     max_bins=max_bins,
                                     max_leaf_nodes=n_leaf_nodes,
                                     n_iter_no_change=None,
                                     random_state=0,
                                     verbose=1)
est.fit(data_train, target_train)
toc = time()
predicted_test = est.predict(data_test)
predicted_proba_test = est.predict_proba(data_test)
roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
acc = accuracy_score(target_test, predicted_test)
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.lightgbm:
    print("Fitting a LightGBM model...")
    tic = time()
    lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')
    lightgbm_est.fit(data_train, target_train)
    toc = time()
    predicted_test = lightgbm_est.predict(data_test)
    predicted_proba_test = lightgbm_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.xgboost:
    print("Fitting an XGBoost model...")
    tic = time()
    xgboost_est = get_equivalent_estimator(est, lib='xgboost')
    xgboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = xgboost_est.predict(data_test)
    predicted_proba_test = xgboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.catboost:
    print("Fitting a Catboost model...")
    tic = time()
    catboost_est = get_equivalent_estimator(est, lib='catboost')
    catboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = catboost_est.predict(data_test)
    predicted_proba_test = catboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
```
### 5 - examples/model_selection/plot_roc_crossval.py:

Start line: 1, End line: 89

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

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
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

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
```
### 6 - sklearn/metrics/ranking.py:

Start line: 347, End line: 375

```python
def roc_auc_score(y_true, y_score, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):

    y_type = type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type == "multiclass" or (y_type == "binary" and
                                  y_score.ndim == 2 and
                                  y_score.shape[1] > 2):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.:
            raise ValueError("Partial AUC computation not available in "
                             "multiclass setting, 'max_fpr' must be"
                             " set to `None`, received `max_fpr={0}` "
                             "instead".format(max_fpr))
        if multi_class == 'raise':
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return _multiclass_roc_auc_score(y_true, y_score, labels,
                                         multi_class, average, sample_weight)
    elif y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, labels)[:, 0]
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
    else:  # multilabel-indicator
        return _average_binary_score(partial(_binary_roc_auc_score,
                                             max_fpr=max_fpr),
                                     y_true, y_score, average,
                                     sample_weight=sample_weight)
```
### 7 - sklearn/utils/estimator_checks.py:

Start line: 1988, End line: 2029

```python
@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)
    if name == 'BernoulliNB':
        X = X > X.mean()
    set_random_state(classifier)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        assert isinstance(decision, np.ndarray)
        if len(classes) == 2:
            dec_pred = (decision.ravel() > 0).astype(np.int)
            dec_exp = classifier.classes_[dec_pred]
            assert_array_equal(dec_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, dec_exp)),
                                ", ".join(map(str, y_pred))))
        elif getattr(classifier, 'decision_function_shape', 'ovr') == 'ovr':
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            assert_array_equal(y_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, y_exp)),
                                ", ".join(map(str, y_pred))))

    # training set performance
    if name != "ComplementNB":
        # This is a pathological data set for ComplementNB.
        # For some specific cases 'ComplementNB' predicts less classes
        # than expected
        assert_array_equal(np.unique(y), np.unique(y_pred))
    assert_array_equal(classes, classifier.classes_,
                       err_msg="Unexpected classes_ attribute for %r: "
                       "expected '%s', got '%s'" %
                       (classifier, ", ".join(map(str, classes)),
                        ", ".join(map(str, classifier.classes_))))
```
### 8 - examples/model_selection/plot_roc.py:

Start line: 1, End line: 91

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

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-label
classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

Another evaluation measure for multi-label classification is
macro-averaging, which gives equal weight to the classification of each
label.

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`

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
from sklearn.metrics import roc_auc_score

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
### 9 - sklearn/ensemble/voting.py:

Start line: 368, End line: 428

```python
class VotingRegressor(RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    .. versionadded:: 0.21

    A voting regressor is an ensemble meta-estimator that fits base
    regressors each on the whole dataset. It, then, averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``None`` or ``'drop'``
        using ``set_params``.

    weights : array-like, shape (n_regressors,), optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not `None`.

    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2)])
    >>> print(er.fit(X, y).predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]

    See also
    --------
    VotingClassifier: Soft Voting/Majority Rule classifier.
    """

    def __init__(self, estimators, weights=None, n_jobs=None):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
```
### 10 - examples/ensemble/plot_voting_probas.py:

Start line: 1, End line: 87

```python
"""
===========================================================
Plot class probabilities calculated by the VotingClassifier
===========================================================

.. currentmodule:: sklearn

Plot the class probabilities of the first sample in a toy dataset predicted by
three different classifiers and averaged by the
:class:`~ensemble.VotingClassifier`.

First, three examplary classifiers are initialized
(:class:`~linear_model.LogisticRegression`, :class:`~naive_bayes.GaussianNB`,
and :class:`~ensemble.RandomForestClassifier`) and used to initialize a
soft-voting :class:`~ensemble.VotingClassifier` with weights `[1, 1, 5]`, which
means that the predicted probabilities of the
:class:`~ensemble.RandomForestClassifier` count 5 times as much as the weights
of the other classifiers when the averaged probability is calculated.

To visualize the probability weighting, we fit each classifier on the training
set and plot the predicted class probabilities for the first sample in this
example dataset.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(max_iter=1000, random_state=123)
clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
clf3 = GaussianNB()
X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
y = np.array([1, 1, 2, 2])

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='soft',
                        weights=[1, 1, 5])

# predict class probabilities for all classifiers
probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]

# get class probabilities for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]


# plotting

N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')

# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')

# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.tight_layout()
plt.show()
```
### 15 - sklearn/ensemble/voting.py:

Start line: 430, End line: 452

```python
class VotingRegressor(RegressorMixin, _BaseVoting):

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        y = column_or_1d(y, warn=True)
        return super().fit(X, y, sample_weight)
```
### 18 - sklearn/ensemble/voting.py:

Start line: 138, End line: 229

```python
class VotingClassifier(ClassifierMixin, _BaseVoting):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <voting_classifier>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``None`` or ``'drop'``
        using ``set_params``.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like, shape (n_classifiers,), optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    flatten_transform : bool, optional (default=True)
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not `None`.

    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    classes_ : array-like, shape (n_predictions,)
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)

    See also
    --------
    VotingRegressor: Prediction voting regressor.
    """
```
### 19 - sklearn/ensemble/voting.py:

Start line: 304, End line: 332

```python
class VotingClassifier(ClassifierMixin, _BaseVoting):

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        check_is_fitted(self)
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg : array-like, shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        return self._predict_proba
```
### 23 - sklearn/ensemble/voting.py:

Start line: 275, End line: 302

```python
class VotingClassifier(ClassifierMixin, _BaseVoting):

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like, shape (n_samples,)
            Predicted class labels.
        """

        check_is_fitted(self)
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self._weights_not_none)),
                axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj
```
### 34 - sklearn/ensemble/voting.py:

Start line: 454, End line: 490

```python
class VotingRegressor(RegressorMixin, _BaseVoting):

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1,
                          weights=self._weights_not_none)

    def transform(self, X):
        """Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions
            array-like of shape (n_samples, n_classifiers), being
            values predicted by each regressor.
        """
        check_is_fitted(self)
        return self._predict(X)
```
### 46 - sklearn/ensemble/voting.py:

Start line: 1, End line: 32

```python
"""
Soft Voting/Majority Rule classifier and Voting regressor.

This module contains:
 - A Soft Voting/Majority Rule classifier for classification estimators.
 - A Voting regressor for regression estimators.
"""

from abc import abstractmethod

import numpy as np

from joblib import Parallel, delayed

from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import TransformerMixin
from ..base import clone
from .base import _parallel_fit_estimator
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils.validation import check_is_fitted
from ..utils.metaestimators import _BaseComposition
from ..utils.multiclass import check_classification_targets
from ..utils.validation import column_or_1d
```
### 69 - sklearn/ensemble/voting.py:

Start line: 59, End line: 95

```python
class _BaseVoting(TransformerMixin, _BaseComposition):

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        common fit operations.
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum(
            [clf in (None, 'drop') for _, clf in self.estimators]
        )
        if n_isnone == len(self.estimators):
            raise ValueError(
                'All estimators are None or "drop". At least one is required!'
            )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                                 sample_weight=sample_weight)
                for clf in clfs if clf not in (None, 'drop')
            )

        self.named_estimators_ = Bunch()
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e
        return self
```
