# scikit-learn__scikit-learn-11542

* repo: scikit-learn/scikit-learn
* base_commit: `cd7d9d985e1bbe2dbbbae17da0e9fbbba7e8c8c6`

## Problem statement

Change default n_estimators in RandomForest (to 100?)
Analysis of code on github shows that people use default parameters when they shouldn't. We can make that a little bit less bad by providing reasonable defaults. The default for n_estimators is not great imho and I think we should change it. I suggest 100.
We could probably run benchmarks with openml if we want to do something empirical, but I think anything is better than 10.

I'm not sure if I want to tag this 1.0 because really no-one should ever run a random forest with 10 trees imho and therefore deprecation of the current default will show people they have a bug.


## Patch

```diff
diff --git a/examples/applications/plot_prediction_latency.py b/examples/applications/plot_prediction_latency.py
--- a/examples/applications/plot_prediction_latency.py
+++ b/examples/applications/plot_prediction_latency.py
@@ -285,7 +285,7 @@ def plot_benchmark_throughput(throughputs, configuration):
          'complexity_label': 'non-zero coefficients',
          'complexity_computer': lambda clf: np.count_nonzero(clf.coef_)},
         {'name': 'RandomForest',
-         'instance': RandomForestRegressor(),
+         'instance': RandomForestRegressor(n_estimators=100),
          'complexity_label': 'estimators',
          'complexity_computer': lambda clf: clf.n_estimators},
         {'name': 'SVR',
diff --git a/examples/ensemble/plot_ensemble_oob.py b/examples/ensemble/plot_ensemble_oob.py
--- a/examples/ensemble/plot_ensemble_oob.py
+++ b/examples/ensemble/plot_ensemble_oob.py
@@ -45,15 +45,18 @@
 # error trajectory during training.
 ensemble_clfs = [
     ("RandomForestClassifier, max_features='sqrt'",
-        RandomForestClassifier(warm_start=True, oob_score=True,
+        RandomForestClassifier(n_estimators=100,
+                               warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE)),
     ("RandomForestClassifier, max_features='log2'",
-        RandomForestClassifier(warm_start=True, max_features='log2',
+        RandomForestClassifier(n_estimators=100,
+                               warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE)),
     ("RandomForestClassifier, max_features=None",
-        RandomForestClassifier(warm_start=True, max_features=None,
+        RandomForestClassifier(n_estimators=100,
+                               warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE))
 ]
diff --git a/examples/ensemble/plot_random_forest_regression_multioutput.py b/examples/ensemble/plot_random_forest_regression_multioutput.py
--- a/examples/ensemble/plot_random_forest_regression_multioutput.py
+++ b/examples/ensemble/plot_random_forest_regression_multioutput.py
@@ -44,11 +44,13 @@
                                                     random_state=4)
 
 max_depth = 30
-regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
+regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
+                                                          max_depth=max_depth,
                                                           random_state=0))
 regr_multirf.fit(X_train, y_train)
 
-regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
+regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
+                                random_state=2)
 regr_rf.fit(X_train, y_train)
 
 # Predict on new data
diff --git a/examples/ensemble/plot_voting_probas.py b/examples/ensemble/plot_voting_probas.py
--- a/examples/ensemble/plot_voting_probas.py
+++ b/examples/ensemble/plot_voting_probas.py
@@ -30,7 +30,7 @@
 from sklearn.ensemble import VotingClassifier
 
 clf1 = LogisticRegression(random_state=123)
-clf2 = RandomForestClassifier(random_state=123)
+clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
 clf3 = GaussianNB()
 X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
 y = np.array([1, 1, 2, 2])
diff --git a/sklearn/ensemble/forest.py b/sklearn/ensemble/forest.py
--- a/sklearn/ensemble/forest.py
+++ b/sklearn/ensemble/forest.py
@@ -135,7 +135,7 @@ class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):
     @abstractmethod
     def __init__(self,
                  base_estimator,
-                 n_estimators=10,
+                 n_estimators=100,
                  estimator_params=tuple(),
                  bootstrap=False,
                  oob_score=False,
@@ -242,6 +242,12 @@ def fit(self, X, y, sample_weight=None):
         -------
         self : object
         """
+
+        if self.n_estimators == 'warn':
+            warnings.warn("The default value of n_estimators will change from "
+                          "10 in version 0.20 to 100 in 0.22.", FutureWarning)
+            self.n_estimators = 10
+
         # Validate or convert input data
         X = check_array(X, accept_sparse="csc", dtype=DTYPE)
         y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
@@ -399,7 +405,7 @@ class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
     @abstractmethod
     def __init__(self,
                  base_estimator,
-                 n_estimators=10,
+                 n_estimators=100,
                  estimator_params=tuple(),
                  bootstrap=False,
                  oob_score=False,
@@ -408,7 +414,6 @@ def __init__(self,
                  verbose=0,
                  warm_start=False,
                  class_weight=None):
-
         super(ForestClassifier, self).__init__(
             base_estimator,
             n_estimators=n_estimators,
@@ -638,7 +643,7 @@ class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):
     @abstractmethod
     def __init__(self,
                  base_estimator,
-                 n_estimators=10,
+                 n_estimators=100,
                  estimator_params=tuple(),
                  bootstrap=False,
                  oob_score=False,
@@ -758,6 +763,10 @@ class RandomForestClassifier(ForestClassifier):
     n_estimators : integer, optional (default=10)
         The number of trees in the forest.
 
+        .. versionchanged:: 0.20
+           The default value of ``n_estimators`` will change from 10 in
+           version 0.20 to 100 in version 0.22.
+
     criterion : string, optional (default="gini")
         The function to measure the quality of a split. Supported criteria are
         "gini" for the Gini impurity and "entropy" for the information gain.
@@ -971,7 +980,7 @@ class labels (multi-output problem).
     DecisionTreeClassifier, ExtraTreesClassifier
     """
     def __init__(self,
-                 n_estimators=10,
+                 n_estimators='warn',
                  criterion="gini",
                  max_depth=None,
                  min_samples_split=2,
@@ -1032,6 +1041,10 @@ class RandomForestRegressor(ForestRegressor):
     n_estimators : integer, optional (default=10)
         The number of trees in the forest.
 
+        .. versionchanged:: 0.20
+           The default value of ``n_estimators`` will change from 10 in
+           version 0.20 to 100 in version 0.22.
+
     criterion : string, optional (default="mse")
         The function to measure the quality of a split. Supported criteria
         are "mse" for the mean squared error, which is equal to variance
@@ -1211,7 +1224,7 @@ class RandomForestRegressor(ForestRegressor):
     DecisionTreeRegressor, ExtraTreesRegressor
     """
     def __init__(self,
-                 n_estimators=10,
+                 n_estimators='warn',
                  criterion="mse",
                  max_depth=None,
                  min_samples_split=2,
@@ -1268,6 +1281,10 @@ class ExtraTreesClassifier(ForestClassifier):
     n_estimators : integer, optional (default=10)
         The number of trees in the forest.
 
+        .. versionchanged:: 0.20
+           The default value of ``n_estimators`` will change from 10 in
+           version 0.20 to 100 in version 0.22.
+
     criterion : string, optional (default="gini")
         The function to measure the quality of a split. Supported criteria are
         "gini" for the Gini impurity and "entropy" for the information gain.
@@ -1454,7 +1471,7 @@ class labels (multi-output problem).
         splits.
     """
     def __init__(self,
-                 n_estimators=10,
+                 n_estimators='warn',
                  criterion="gini",
                  max_depth=None,
                  min_samples_split=2,
@@ -1513,6 +1530,10 @@ class ExtraTreesRegressor(ForestRegressor):
     n_estimators : integer, optional (default=10)
         The number of trees in the forest.
 
+        .. versionchanged:: 0.20
+           The default value of ``n_estimators`` will change from 10 in
+           version 0.20 to 100 in version 0.22.
+
     criterion : string, optional (default="mse")
         The function to measure the quality of a split. Supported criteria
         are "mse" for the mean squared error, which is equal to variance
@@ -1666,7 +1687,7 @@ class ExtraTreesRegressor(ForestRegressor):
     RandomForestRegressor: Ensemble regressor using trees with optimal splits.
     """
     def __init__(self,
-                 n_estimators=10,
+                 n_estimators='warn',
                  criterion="mse",
                  max_depth=None,
                  min_samples_split=2,
@@ -1728,6 +1749,10 @@ class RandomTreesEmbedding(BaseForest):
     n_estimators : integer, optional (default=10)
         Number of trees in the forest.
 
+        .. versionchanged:: 0.20
+           The default value of ``n_estimators`` will change from 10 in
+           version 0.20 to 100 in version 0.22.
+
     max_depth : integer, optional (default=5)
         The maximum depth of each tree. If None, then nodes are expanded until
         all leaves are pure or until all leaves contain less than
@@ -1833,7 +1858,7 @@ class RandomTreesEmbedding(BaseForest):
     """
 
     def __init__(self,
-                 n_estimators=10,
+                 n_estimators='warn',
                  max_depth=5,
                  min_samples_split=2,
                  min_samples_leaf=1,
diff --git a/sklearn/utils/estimator_checks.py b/sklearn/utils/estimator_checks.py
--- a/sklearn/utils/estimator_checks.py
+++ b/sklearn/utils/estimator_checks.py
@@ -340,7 +340,13 @@ def set_checking_parameters(estimator):
         estimator.set_params(n_resampling=5)
     if "n_estimators" in params:
         # especially gradient boosting with default 100
-        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
+        # FIXME: The default number of trees was changed and is set to 'warn'
+        # for some of the ensemble methods. We need to catch this case to avoid
+        # an error during the comparison. To be reverted in 0.22.
+        if estimator.n_estimators == 'warn':
+            estimator.set_params(n_estimators=5)
+        else:
+            estimator.set_params(n_estimators=min(5, estimator.n_estimators))
     if "max_trials" in params:
         # RANSAC
         estimator.set_params(max_trials=10)

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_forest.py b/sklearn/ensemble/tests/test_forest.py
--- a/sklearn/ensemble/tests/test_forest.py
+++ b/sklearn/ensemble/tests/test_forest.py
@@ -31,6 +31,7 @@
 from sklearn.utils.testing import assert_raises
 from sklearn.utils.testing import assert_warns
 from sklearn.utils.testing import assert_warns_message
+from sklearn.utils.testing import assert_no_warnings
 from sklearn.utils.testing import ignore_warnings
 
 from sklearn import datasets
@@ -186,6 +187,7 @@ def check_regressor_attributes(name):
     assert_false(hasattr(r, "n_classes_"))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_REGRESSORS)
 def test_regressor_attributes(name):
     check_regressor_attributes(name)
@@ -432,6 +434,7 @@ def check_oob_score_raise_error(name):
                                                   bootstrap=False).fit, X, y)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_ESTIMATORS)
 def test_oob_score_raise_error(name):
     check_oob_score_raise_error(name)
@@ -489,6 +492,7 @@ def check_pickle(name, X, y):
     assert_equal(score, score2)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
 def test_pickle(name):
     if name in FOREST_CLASSIFIERS:
@@ -526,6 +530,7 @@ def check_multioutput(name):
             assert_equal(log_proba[1].shape, (4, 4))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
 def test_multioutput(name):
     check_multioutput(name)
@@ -549,6 +554,7 @@ def check_classes_shape(name):
     assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
 def test_classes_shape(name):
     check_classes_shape(name)
@@ -738,6 +744,7 @@ def check_min_samples_split(name):
                    "Failed with {0}".format(name))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_ESTIMATORS)
 def test_min_samples_split(name):
     check_min_samples_split(name)
@@ -775,6 +782,7 @@ def check_min_samples_leaf(name):
                    "Failed with {0}".format(name))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_ESTIMATORS)
 def test_min_samples_leaf(name):
     check_min_samples_leaf(name)
@@ -842,6 +850,7 @@ def check_sparse_input(name, X, X_sparse, y):
                                   dense.fit_transform(X).toarray())
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_ESTIMATORS)
 @pytest.mark.parametrize('sparse_matrix',
                          (csr_matrix, csc_matrix, coo_matrix))
@@ -899,6 +908,7 @@ def check_memory_layout(name, dtype):
     assert_array_almost_equal(est.fit(X, y).predict(X), y)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
 @pytest.mark.parametrize('dtype', (np.float64, np.float32))
 def test_memory_layout(name, dtype):
@@ -977,6 +987,7 @@ def check_class_weights(name):
     clf.fit(iris.data, iris.target, sample_weight=sample_weight)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
 def test_class_weights(name):
     check_class_weights(name)
@@ -996,6 +1007,7 @@ def check_class_weight_balanced_and_bootstrap_multi_output(name):
     clf.fit(X, _y)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
 def test_class_weight_balanced_and_bootstrap_multi_output(name):
     check_class_weight_balanced_and_bootstrap_multi_output(name)
@@ -1026,6 +1038,7 @@ def check_class_weight_errors(name):
     assert_raises(ValueError, clf.fit, X, _y)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 @pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
 def test_class_weight_errors(name):
     check_class_weight_errors(name)
@@ -1163,6 +1176,7 @@ def test_warm_start_oob(name):
     check_warm_start_oob(name)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_dtype_convert(n_classes=15):
     classifier = RandomForestClassifier(random_state=0, bootstrap=False)
 
@@ -1201,6 +1215,7 @@ def test_decision_path(name):
     check_decision_path(name)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_min_impurity_split():
     # Test if min_impurity_split of base estimators is set
     # Regression test for #8006
@@ -1216,6 +1231,7 @@ def test_min_impurity_split():
             assert_equal(tree.min_impurity_split, 0.1)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_min_impurity_decrease():
     X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
     all_estimators = [RandomForestClassifier, RandomForestRegressor,
@@ -1228,3 +1244,21 @@ def test_min_impurity_decrease():
             # Simply check if the parameter is passed on correctly. Tree tests
             # will suffice for the actual working of this param
             assert_equal(tree.min_impurity_decrease, 0.1)
+
+
+@pytest.mark.parametrize('forest',
+                         [RandomForestClassifier, RandomForestRegressor,
+                          ExtraTreesClassifier, ExtraTreesRegressor,
+                          RandomTreesEmbedding])
+def test_nestimators_future_warning(forest):
+    # FIXME: to be removed 0.22
+
+    # When n_estimators default value is used
+    msg_future = ("The default value of n_estimators will change from "
+                  "10 in version 0.20 to 100 in 0.22.")
+    est = forest()
+    est = assert_warns_message(FutureWarning, msg_future, est.fit, X, y)
+
+    # When n_estimators is a valid value not equal to the default
+    est = forest(n_estimators=100)
+    est = assert_no_warnings(est.fit, X, y)
diff --git a/sklearn/ensemble/tests/test_voting_classifier.py b/sklearn/ensemble/tests/test_voting_classifier.py
--- a/sklearn/ensemble/tests/test_voting_classifier.py
+++ b/sklearn/ensemble/tests/test_voting_classifier.py
@@ -1,6 +1,8 @@
 """Testing for the VotingClassifier"""
 
+import pytest
 import numpy as np
+
 from sklearn.utils.testing import assert_almost_equal, assert_array_equal
 from sklearn.utils.testing import assert_array_almost_equal
 from sklearn.utils.testing import assert_equal, assert_true, assert_false
@@ -74,6 +76,7 @@ def test_notfitted():
     assert_raise_message(NotFittedError, msg, eclf.predict_proba, X)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_majority_label_iris():
     """Check classification by majority label on dataset iris."""
     clf1 = LogisticRegression(random_state=123)
@@ -86,6 +89,7 @@ def test_majority_label_iris():
     assert_almost_equal(scores.mean(), 0.95, decimal=2)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_tie_situation():
     """Check voting classifier selects smaller class label in tie situation."""
     clf1 = LogisticRegression(random_state=123)
@@ -97,6 +101,7 @@ def test_tie_situation():
     assert_equal(eclf.fit(X, y).predict(X)[73], 1)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_weights_iris():
     """Check classification by average probabilities on dataset iris."""
     clf1 = LogisticRegression(random_state=123)
@@ -110,6 +115,7 @@ def test_weights_iris():
     assert_almost_equal(scores.mean(), 0.93, decimal=2)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_predict_on_toy_problem():
     """Manually check predicted class labels for toy dataset."""
     clf1 = LogisticRegression(random_state=123)
@@ -142,6 +148,7 @@ def test_predict_on_toy_problem():
     assert_equal(all(eclf.fit(X, y).predict(X)), all([1, 1, 1, 2, 2, 2]))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_predict_proba_on_toy_problem():
     """Calculate predicted probabilities on toy dataset."""
     clf1 = LogisticRegression(random_state=123)
@@ -209,6 +216,7 @@ def test_multilabel():
         return
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_gridsearch():
     """Check GridSearch support."""
     clf1 = LogisticRegression(random_state=1)
@@ -226,6 +234,7 @@ def test_gridsearch():
     grid.fit(iris.data, iris.target)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_parallel_fit():
     """Check parallel backend of VotingClassifier on toy dataset."""
     clf1 = LogisticRegression(random_state=123)
@@ -247,6 +256,7 @@ def test_parallel_fit():
     assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_sample_weight():
     """Tests sample_weight parameter of VotingClassifier"""
     clf1 = LogisticRegression(random_state=123)
@@ -290,6 +300,7 @@ def fit(self, X, y, *args, **sample_weight):
     eclf.fit(X, y, sample_weight=np.ones((len(y),)))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_set_params():
     """set_params should be able to set estimators"""
     clf1 = LogisticRegression(random_state=123, C=1.0)
@@ -324,6 +335,7 @@ def test_set_params():
                  eclf1.get_params()["lr"].get_params()['C'])
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_set_estimator_none():
     """VotingClassifier set_params should be able to set estimators as None"""
     # Test predict
@@ -376,6 +388,7 @@ def test_set_estimator_none():
     assert_array_equal(eclf2.transform(X1), np.array([[0], [1]]))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_estimator_weights_format():
     # Test estimator weights inputs as list and array
     clf1 = LogisticRegression(random_state=123)
@@ -393,6 +406,7 @@ def test_estimator_weights_format():
     assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_transform():
     """Check transform method of VotingClassifier on toy dataset."""
     clf1 = LogisticRegression(random_state=123)
diff --git a/sklearn/ensemble/tests/test_weight_boosting.py b/sklearn/ensemble/tests/test_weight_boosting.py
--- a/sklearn/ensemble/tests/test_weight_boosting.py
+++ b/sklearn/ensemble/tests/test_weight_boosting.py
@@ -1,6 +1,8 @@
 """Testing for the boost module (sklearn.ensemble.boost)."""
 
+import pytest
 import numpy as np
+
 from sklearn.utils.testing import assert_array_equal, assert_array_less
 from sklearn.utils.testing import assert_array_almost_equal
 from sklearn.utils.testing import assert_equal, assert_true, assert_greater
@@ -277,6 +279,7 @@ def test_error():
                   X, y_class, sample_weight=np.asarray([-1]))
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_base_estimator():
     # Test different base estimators.
     from sklearn.ensemble import RandomForestClassifier
diff --git a/sklearn/feature_selection/tests/test_from_model.py b/sklearn/feature_selection/tests/test_from_model.py
--- a/sklearn/feature_selection/tests/test_from_model.py
+++ b/sklearn/feature_selection/tests/test_from_model.py
@@ -1,3 +1,4 @@
+import pytest
 import numpy as np
 
 from sklearn.utils.testing import assert_true
@@ -32,6 +33,7 @@ def test_invalid_input():
         assert_raises(ValueError, model.transform, data)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_input_estimator_unchanged():
     # Test that SelectFromModel fits on a clone of the estimator.
     est = RandomForestClassifier()
@@ -119,6 +121,7 @@ def test_2d_coef():
             assert_array_almost_equal(X_new, X[:, feature_mask])
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_partial_fit():
     est = PassiveAggressiveClassifier(random_state=0, shuffle=False,
                                       max_iter=5, tol=None)
diff --git a/sklearn/feature_selection/tests/test_rfe.py b/sklearn/feature_selection/tests/test_rfe.py
--- a/sklearn/feature_selection/tests/test_rfe.py
+++ b/sklearn/feature_selection/tests/test_rfe.py
@@ -1,6 +1,7 @@
 """
 Testing Recursive feature elimination
 """
+import pytest
 import numpy as np
 from numpy.testing import assert_array_almost_equal, assert_array_equal
 from scipy import sparse
@@ -336,6 +337,7 @@ def test_rfe_cv_n_jobs():
     assert_array_almost_equal(rfecv.grid_scores_, rfecv_grid_scores)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_rfe_cv_groups():
     generator = check_random_state(0)
     iris = load_iris()
diff --git a/sklearn/tests/test_calibration.py b/sklearn/tests/test_calibration.py
--- a/sklearn/tests/test_calibration.py
+++ b/sklearn/tests/test_calibration.py
@@ -2,6 +2,7 @@
 # License: BSD 3 clause
 
 from __future__ import division
+import pytest
 import numpy as np
 from scipy import sparse
 from sklearn.model_selection import LeaveOneOut
@@ -24,7 +25,7 @@
 from sklearn.calibration import calibration_curve
 
 
-@ignore_warnings
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
 def test_calibration():
     """Test calibration objects with isotonic and sigmoid"""
     n_samples = 100

```
