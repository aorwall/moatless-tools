# scikit-learn__scikit-learn-11542

| **scikit-learn/scikit-learn** | `cd7d9d985e1bbe2dbbbae17da0e9fbbba7e8c8c6` |
| ---- | ---- |
| **No of patches** | 6 |
| **All found context length** | 5578 |
| **Any found context length** | 340 |
| **Avg pos** | 53.333333333333336 |
| **Min pos** | 1 |
| **Max pos** | 36 |
| **Top file pos** | 1 |
| **Missing snippets** | 20 |
| **Missing patch files** | 2 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| examples/applications/plot_prediction_latency.py | 288 | 288 | - | - | -
| examples/ensemble/plot_ensemble_oob.py | 48 | 56 | 5 | 2 | 5578
| examples/ensemble/plot_random_forest_regression_multioutput.py | 47 | 51 | 36 | 14 | 27529
| examples/ensemble/plot_voting_probas.py | 33 | 33 | - | - | -
| sklearn/ensemble/forest.py | 138 | 138 | 25 | 1 | 20733
| sklearn/ensemble/forest.py | 245 | 245 | - | 1 | -
| sklearn/ensemble/forest.py | 402 | 402 | 29 | 1 | 21870
| sklearn/ensemble/forest.py | 411 | 411 | 29 | 1 | 21870
| sklearn/ensemble/forest.py | 641 | 641 | 21 | 1 | 16008
| sklearn/ensemble/forest.py | 761 | 761 | 20 | 1 | 15832
| sklearn/ensemble/forest.py | 974 | 974 | 20 | 1 | 15832
| sklearn/ensemble/forest.py | 1035 | 1035 | 3 | 1 | 4569
| sklearn/ensemble/forest.py | 1214 | 1214 | 4 | 1 | 4897
| sklearn/ensemble/forest.py | 1271 | 1271 | 9 | 1 | 11002
| sklearn/ensemble/forest.py | 1457 | 1457 | 18 | 1 | 14153
| sklearn/ensemble/forest.py | 1516 | 1516 | 18 | 1 | 14153
| sklearn/ensemble/forest.py | 1669 | 1669 | 18 | 1 | 14153
| sklearn/ensemble/forest.py | 1731 | 1731 | 6 | 1 | 6776
| sklearn/ensemble/forest.py | 1836 | 1836 | 10 | 1 | 11341
| sklearn/utils/estimator_checks.py | 343 | 343 | 17 | 7 | 13810


## Problem Statement

```
Change default n_estimators in RandomForest (to 100?)
Analysis of code on github shows that people use default parameters when they shouldn't. We can make that a little bit less bad by providing reasonable defaults. The default for n_estimators is not great imho and I think we should change it. I suggest 100.
We could probably run benchmarks with openml if we want to do something empirical, but I think anything is better than 10.

I'm not sure if I want to tag this 1.0 because really no-one should ever run a random forest with 10 trees imho and therefore deprecation of the current default will show people they have a bug.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/ensemble/forest.py** | 973 | 1253| 340 | 340 | 16995 | 
| **-> 2 <-** | **1 sklearn/ensemble/forest.py** | 745 | 972| 2336 | 2676 | 16995 | 
| **-> 3 <-** | **1 sklearn/ensemble/forest.py** | 1019 | 1212| 1893 | 4569 | 16995 | 
| **-> 4 <-** | **1 sklearn/ensemble/forest.py** | 1213 | 1498| 328 | 4897 | 16995 | 
| **-> 5 <-** | **2 examples/ensemble/plot_ensemble_oob.py** | 1 | 87| 681 | 5578 | 17728 | 
| **-> 6 <-** | **2 sklearn/ensemble/forest.py** | 1711 | 1833| 1198 | 6776 | 17728 | 
| **-> 7 <-** | **2 sklearn/ensemble/forest.py** | 1502 | 1667| 1564 | 8340 | 17728 | 
| 8 | **2 sklearn/ensemble/forest.py** | 1 | 94| 670 | 9010 | 17728 | 
| **-> 9 <-** | **2 sklearn/ensemble/forest.py** | 1257 | 1455| 1992 | 11002 | 17728 | 
| **-> 10 <-** | **2 sklearn/ensemble/forest.py** | 1835 | 1876| 339 | 11341 | 17728 | 
| 11 | 3 benchmarks/bench_covertype.py | 100 | 110| 151 | 11492 | 19631 | 
| **-> 12 <-** | **3 sklearn/ensemble/forest.py** | 1668 | 1708| 321 | 11813 | 19631 | 
| 13 | **3 sklearn/ensemble/forest.py** | 298 | 340| 412 | 12225 | 19631 | 
| 14 | 4 benchmarks/bench_tree.py | 64 | 125| 523 | 12748 | 20501 | 
| 15 | 5 sklearn/ensemble/iforest.py | 130 | 162| 239 | 12987 | 23865 | 
| 16 | 6 benchmarks/bench_mnist.py | 85 | 106| 314 | 13301 | 25596 | 
| **-> 17 <-** | **7 sklearn/utils/estimator_checks.py** | 319 | 371| 509 | 13810 | 45641 | 
| **-> 18 <-** | **7 sklearn/ensemble/forest.py** | 1456 | 1708| 343 | 14153 | 45641 | 
| 19 | 8 sklearn/tree/tree.py | 1314 | 1449| 1325 | 15478 | 58719 | 
| **-> 20 <-** | **8 sklearn/ensemble/forest.py** | 701 | 1015| 354 | 15832 | 58719 | 
| **-> 21 <-** | **8 sklearn/ensemble/forest.py** | 631 | 658| 176 | 16008 | 58719 | 
| 22 | **8 sklearn/utils/estimator_checks.py** | 2014 | 2085| 578 | 16586 | 58719 | 
| 23 | 9 sklearn/ensemble/gradient_boosting.py | 1741 | 1977| 2381 | 18967 | 76997 | 
| 24 | 9 sklearn/tree/tree.py | 1136 | 1285| 1573 | 20540 | 76997 | 
| **-> 25 <-** | **9 sklearn/ensemble/forest.py** | 128 | 158| 193 | 20733 | 76997 | 
| 26 | 10 sklearn/neighbors/approximate.py | 207 | 220| 160 | 20893 | 81673 | 
| 27 | 10 sklearn/tree/tree.py | 111 | 187| 674 | 21567 | 81673 | 
| 28 | 10 sklearn/ensemble/iforest.py | 5 | 26| 117 | 21684 | 81673 | 
| **-> 29 <-** | **10 sklearn/ensemble/forest.py** | 391 | 422| 186 | 21870 | 81673 | 
| 30 | **10 sklearn/utils/estimator_checks.py** | 1884 | 1919| 352 | 22222 | 81673 | 
| 31 | 11 examples/model_selection/plot_randomized_search.py | 1 | 38| 254 | 22476 | 82387 | 
| 32 | 11 sklearn/ensemble/gradient_boosting.py | 1286 | 1527| 2411 | 24887 | 82387 | 
| 33 | 12 examples/ensemble/plot_forest_importances.py | 1 | 55| 373 | 25260 | 82760 | 
| 34 | 13 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 950 | 26210 | 87143 | 
| 35 | 13 benchmarks/bench_covertype.py | 1 | 68| 686 | 26896 | 87143 | 
| **-> 36 <-** | **14 examples/ensemble/plot_random_forest_regression_multioutput.py** | 1 | 77| 633 | 27529 | 87797 | 
| 37 | 15 sklearn/ensemble/bagging.py | 551 | 580| 197 | 27726 | 95729 | 
| 38 | **15 sklearn/utils/estimator_checks.py** | 1082 | 1150| 601 | 28327 | 95729 | 
| 39 | 15 benchmarks/bench_tree.py | 1 | 61| 347 | 28674 | 95729 | 
| 40 | 16 sklearn/datasets/samples_generator.py | 535 | 575| 372 | 29046 | 109747 | 
| 41 | 17 benchmarks/bench_isolation_forest.py | 54 | 160| 1025 | 30071 | 111208 | 
| 42 | 17 sklearn/ensemble/bagging.py | 919 | 942| 160 | 30231 | 111208 | 
| 43 | 17 sklearn/tree/tree.py | 189 | 262| 818 | 31049 | 111208 | 
| 44 | 18 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 31563 | 115121 | 
| 45 | **18 sklearn/utils/estimator_checks.py** | 638 | 686| 441 | 32004 | 115121 | 
| 46 | **18 sklearn/ensemble/forest.py** | 424 | 467| 387 | 32391 | 115121 | 
| 47 | 18 sklearn/ensemble/gradient_boosting.py | 810 | 893| 806 | 33197 | 115121 | 
| 48 | 19 sklearn/ensemble/base.py | 100 | 117| 181 | 33378 | 116268 | 
| 49 | 19 benchmarks/bench_isolation_forest.py | 44 | 53| 108 | 33486 | 116268 | 
| 50 | 20 examples/ensemble/plot_random_forest_embedding.py | 1 | 84| 757 | 34243 | 117230 | 
| 51 | 20 sklearn/ensemble/iforest.py | 29 | 128| 968 | 35211 | 117230 | 
| 52 | 20 sklearn/ensemble/bagging.py | 342 | 386| 377 | 35588 | 117230 | 
| 53 | 20 sklearn/ensemble/bagging.py | 984 | 1014| 248 | 35836 | 117230 | 
| 54 | 20 sklearn/tree/tree.py | 879 | 1064| 1850 | 37686 | 117230 | 
| 55 | 20 sklearn/ensemble/gradient_boosting.py | 926 | 955| 321 | 38007 | 117230 | 
| 56 | 20 examples/model_selection/plot_randomized_search.py | 54 | 89| 332 | 38339 | 117230 | 
| 57 | 21 sklearn/linear_model/randomized_l1.py | 390 | 506| 1055 | 39394 | 123001 | 
| 58 | **21 sklearn/ensemble/forest.py** | 469 | 515| 423 | 39817 | 123001 | 
| 59 | 21 sklearn/tree/tree.py | 264 | 348| 775 | 40592 | 123001 | 
| 60 | 22 examples/ensemble/plot_isolation_forest.py | 1 | 71| 598 | 41190 | 123599 | 
| 61 | **22 sklearn/utils/estimator_checks.py** | 1 | 82| 704 | 41894 | 123599 | 
| 62 | 23 examples/ensemble/plot_forest_iris.py | 1 | 71| 586 | 42480 | 125072 | 
| 63 | 24 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 142| 488 | 42968 | 126163 | 
| 64 | 24 benchmarks/bench_isolation_forest.py | 1 | 29| 213 | 43181 | 126163 | 
| 65 | **24 sklearn/utils/estimator_checks.py** | 1454 | 1516| 511 | 43692 | 126163 | 
| 66 | 25 sklearn/neighbors/lof.py | 138 | 152| 174 | 43866 | 129483 | 
| 67 | 25 sklearn/ensemble/iforest.py | 164 | 236| 661 | 44527 | 129483 | 
| 68 | 25 sklearn/tree/tree.py | 1 | 68| 379 | 44906 | 129483 | 
| 69 | 25 sklearn/linear_model/randomized_l1.py | 337 | 355| 223 | 45129 | 129483 | 
| 70 | **25 sklearn/utils/estimator_checks.py** | 1785 | 1827| 449 | 45578 | 129483 | 
| 71 | 26 sklearn/manifold/t_sne.py | 623 | 646| 251 | 45829 | 137970 | 
| 72 | 26 sklearn/ensemble/base.py | 1 | 57| 375 | 46204 | 137970 | 
| 73 | 27 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 46976 | 139134 | 
| 74 | 28 examples/ensemble/plot_adaboost_regression.py | 1 | 55| 389 | 47365 | 139551 | 
| 75 | 29 examples/ensemble/plot_adaboost_multiclass.py | 1 | 91| 757 | 48122 | 140574 | 
| 76 | 30 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 91| 772 | 48894 | 141822 | 
| 77 | 31 sklearn/utils/testing.py | 666 | 721| 478 | 49372 | 148703 | 
| 78 | 32 examples/ensemble/plot_feature_transformation.py | 1 | 91| 765 | 50137 | 149826 | 
| 79 | 32 sklearn/ensemble/bagging.py | 1 | 60| 407 | 50544 | 149826 | 
| 80 | 33 sklearn/linear_model/ransac.py | 1 | 52| 322 | 50866 | 153732 | 
| 81 | 33 sklearn/tree/tree.py | 373 | 389| 180 | 51046 | 153732 | 
| 82 | 33 sklearn/tree/tree.py | 519 | 725| 2126 | 53172 | 153732 | 
| 83 | 34 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 54486 | 155073 | 
| 84 | 35 sklearn/model_selection/_search.py | 1111 | 1404| 3028 | 57514 | 168248 | 
| 85 | 36 sklearn/linear_model/stochastic_gradient.py | 90 | 149| 708 | 58222 | 182222 | 
| 86 | 36 sklearn/ensemble/bagging.py | 810 | 917| 1069 | 59291 | 182222 | 
| 87 | 37 examples/ensemble/plot_forest_importances_faces.py | 1 | 50| 326 | 59617 | 182548 | 
| 88 | **37 sklearn/utils/estimator_checks.py** | 1688 | 1713| 277 | 59894 | 182548 | 
| 89 | 37 benchmarks/bench_mnist.py | 1 | 58| 512 | 60406 | 182548 | 
| 90 | 37 sklearn/linear_model/ransac.py | 311 | 408| 804 | 61210 | 182548 | 
| 91 | 37 sklearn/datasets/samples_generator.py | 841 | 900| 570 | 61780 | 182548 | 
| 92 | 37 sklearn/manifold/t_sne.py | 808 | 875| 726 | 62506 | 182548 | 
| 93 | 37 benchmarks/bench_plot_randomized_svd.py | 380 | 431| 540 | 63046 | 182548 | 
| 94 | 38 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 59| 462 | 63508 | 183536 | 
| 95 | 39 benchmarks/bench_sample_without_replacement.py | 38 | 208| 1233 | 64741 | 184957 | 
| 96 | 39 sklearn/datasets/samples_generator.py | 38 | 155| 1171 | 65912 | 184957 | 
| 97 | 40 examples/plot_johnson_lindenstrauss_bound.py | 94 | 160| 646 | 66558 | 186737 | 
| 98 | 41 benchmarks/bench_glmnet.py | 47 | 129| 796 | 67354 | 187824 | 
| 99 | **41 sklearn/ensemble/forest.py** | 358 | 388| 239 | 67593 | 187824 | 
| 100 | 42 examples/cluster/plot_birch_vs_minibatchkmeans.py | 1 | 87| 769 | 68362 | 188821 | 
| 101 | 43 benchmarks/bench_lof.py | 36 | 107| 650 | 69012 | 189737 | 
| 102 | 44 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 1 | 52| 391 | 69403 | 190796 | 
| 103 | 44 sklearn/linear_model/randomized_l1.py | 32 | 60| 260 | 69663 | 190796 | 
| 104 | 45 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 70230 | 191363 | 
| 105 | 46 sklearn/feature_selection/rfe.py | 385 | 477| 834 | 71064 | 195378 | 
| 106 | 46 sklearn/feature_selection/rfe.py | 36 | 116| 759 | 71823 | 195378 | 
| 107 | **46 sklearn/ensemble/forest.py** | 342 | 356| 151 | 71974 | 195378 | 
| 108 | 46 benchmarks/bench_mnist.py | 109 | 181| 695 | 72669 | 195378 | 
| 109 | 46 sklearn/linear_model/randomized_l1.py | 507 | 538| 332 | 73001 | 195378 | 
| 110 | 46 sklearn/feature_selection/rfe.py | 279 | 384| 955 | 73956 | 195378 | 
| 111 | 46 sklearn/ensemble/gradient_boosting.py | 1979 | 2002| 329 | 74285 | 195378 | 
| 112 | **46 sklearn/utils/estimator_checks.py** | 1033 | 1056| 269 | 74554 | 195378 | 
| 113 | 46 sklearn/ensemble/gradient_boosting.py | 564 | 590| 283 | 74837 | 195378 | 
| 114 | 46 sklearn/feature_selection/rfe.py | 142 | 223| 662 | 75499 | 195378 | 


## Missing Patch Files

 * 1: examples/applications/plot_prediction_latency.py
 * 2: examples/ensemble/plot_ensemble_oob.py
 * 3: examples/ensemble/plot_random_forest_regression_multioutput.py
 * 4: examples/ensemble/plot_voting_probas.py
 * 5: sklearn/ensemble/forest.py
 * 6: sklearn/utils/estimator_checks.py

### Hint

```
I would like to give it a shot. Is the default value 100 final? 
@ArihantJain456 I didn't tag this one as "help wanted" because I wanted to wait for other core devs to chime in before we do anything.
I agree. Bad defaults should be deprecated. The warning doesn't hurt.​

I'm also +1 for n_estimators=100 by default
Both for this and #11129 I would suggest that the deprecation warning includes a message to encourage the user to change the value manually.
Has someone taken this up? Can I jump on it?
Bagging* also have n_estimators=10. Is this also a bad default?
@jnothman I would think so, but I have less experience and it depends on the base estimator, I think? With trees, probably?
I am fine with changing the default to 100 for random forests and bagging. (with the usual FutureWarning cycle).

@jnothman I would rather reserve the Blocker label for things that are really harmful if not fixed. To me, fixing this issue is an easy nice-to-heave but does not justify delaying the release cycle.
Okay. But given the rate at which we release, I think it's worth making
sure that simple deprecations make it into the release.​ It's not a big
enough feature to *not* delay the release for it.

I agree.
I am currently working on this issue.
```

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


## Code snippets

### 1 - sklearn/ensemble/forest.py:

Start line: 973, End line: 1253

```python
class RandomForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class RandomForestRegressor(ForestRegressor):
```
### 2 - sklearn/ensemble/forest.py:

Start line: 745, End line: 972

```python
class RandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
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

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
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

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(clf.feature_importances_)
    [0.17287856 0.80608704 0.01884792 0.00218648]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """
```
### 3 - sklearn/ensemble/forest.py:

Start line: 1019, End line: 1212

```python
class RandomForestRegressor(ForestRegressor):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
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

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
    >>> regr.fit(X, y)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(regr.feature_importances_)
    [0.17339552 0.81594114 0.         0.01066333]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-2.50699856]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor
    """
```
### 4 - sklearn/ensemble/forest.py:

Start line: 1213, End line: 1498

```python
class RandomForestRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesClassifier(ForestClassifier):
```
### 5 - examples/ensemble/plot_ensemble_oob.py:

Start line: 1, End line: 87

```python
"""
=============================
OOB Errors for Random Forests
=============================

The ``RandomForestClassifier`` is trained using *bootstrap aggregation*, where
each new tree is fit from a bootstrap sample of the training observations
:math:`z_i = (x_i, y_i)`. The *out-of-bag* (OOB) error is the average error for
each :math:`z_i` calculated using predictions from the trees that do not
contain :math:`z_i` in their respective bootstrap sample. This allows the
``RandomForestClassifier`` to be fit and validated whilst being trained [1]_.

The example below demonstrates how the OOB error can be measured at the
addition of each new tree during training. The resulting plot allows a
practitioner to approximate a suitable value of ``n_estimators`` at which the
error stabilizes.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
       Learning Ed. 2", p592-593, Springer, 2009.

"""
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

print(__doc__)

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 175

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
```
### 6 - sklearn/ensemble/forest.py:

Start line: 1711, End line: 1833

```python
class RandomTreesEmbedding(BaseForest):
    """An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        Number of trees in the forest.

    max_depth : integer, optional (default=5)
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
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

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    sparse_output : bool, optional (default=True)
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    """
```
### 7 - sklearn/ensemble/forest.py:

Start line: 1502, End line: 1667

```python
class ExtraTreesRegressor(ForestRegressor):
    """An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
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

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

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

    See also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.
    """
```
### 8 - sklearn/ensemble/forest.py:

Start line: 1, End line: 94

```python
"""Forest of trees-based ensemble methods

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.

"""

from __future__ import division

import warnings
from warnings import warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack


from ..base import ClassifierMixin, RegressorMixin
from ..externals.joblib import Parallel, delayed
from ..externals import six
from ..metrics import r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from ..tree._tree import DTYPE, DOUBLE
from ..utils import check_random_state, check_array, compute_sample_weight
from ..exceptions import DataConversionWarning, NotFittedError
from .base import BaseEnsemble, _partition_estimators
from ..utils.fixes import parallel_helper
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted

__all__ = ["RandomForestClassifier",
           "RandomForestRegressor",
           "ExtraTreesClassifier",
           "ExtraTreesRegressor",
           "RandomTreesEmbedding"]

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices
```
### 9 - sklearn/ensemble/forest.py:

Start line: 1257, End line: 1455

```python
class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
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

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
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

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

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

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    """
```
### 10 - sklearn/ensemble/forest.py:

Start line: 1835, End line: 1876

```python
class RandomTreesEmbedding(BaseForest):

    def __init__(self,
                 n_estimators=10,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomTreesEmbedding, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")
```
### 12 - sklearn/ensemble/forest.py:

Start line: 1668, End line: 1708

```python
class ExtraTreesRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesRegressor, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
```
### 13 - sklearn/ensemble/forest.py:

Start line: 298, End line: 340

```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):

    def fit(self, X, y, sample_weight=None):
        # ... other code

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = []
            for i in range(n_more_estimators):
                tree = self._make_estimator(append=False,
                                            random_state=random_state)
                trees.append(tree)

            # Parallel loop: we use the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading always more efficient than multiprocessing in
            # that case.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             backend="threading")(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
```
### 17 - sklearn/utils/estimator_checks.py:

Start line: 319, End line: 371

```python
def set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    if ("n_iter" in params and estimator.__class__.__name__ != "TSNE"
            and not isinstance(estimator, BaseSGD)):
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
        # especially gradient boosting with default 100
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if "max_trials" in params:
        # RANSAC
        estimator.set_params(max_trials=10)
    if "n_init" in params:
        # K-Means
        estimator.set_params(n_init=2)
    if "decision_function_shape" in params:
        # SVC
        estimator.set_params(decision_function_shape='ovo')

    if estimator.__class__.__name__ == "SelectFdr":
        # be tolerant of noisy datasets (not actually speed)
        estimator.set_params(alpha=.5)

    if estimator.__class__.__name__ == "TheilSenRegressor":
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
```
### 18 - sklearn/ensemble/forest.py:

Start line: 1456, End line: 1708

```python
class ExtraTreesClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesRegressor(ForestRegressor):
```
### 20 - sklearn/ensemble/forest.py:

Start line: 701, End line: 1015

```python
class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_


class RandomForestClassifier(ForestClassifier):
```
### 21 - sklearn/ensemble/forest.py:

Start line: 631, End line: 658

```python
class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):
    """Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ForestRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
```
### 22 - sklearn/utils/estimator_checks.py:

Start line: 2014, End line: 2085

```python
def check_parameters_default_constructible(name, Estimator):
    # this check works on classes, not instances
    classifier = LinearDiscriminantAnalysis()
    # test default-constructibility
    # get rid of deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        if name in META_ESTIMATORS:
            estimator = Estimator(classifier)
        else:
            estimator = Estimator()
        # test cloning
        clone(estimator)
        # test __repr__
        repr(estimator)
        # test that set_params returns self
        assert_true(estimator.set_params() is estimator)

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

        if name in META_ESTIMATORS:
            # they can need a non-default argument
            init_params = init_params[1:]

        for init_param in init_params:
            assert_not_equal(init_param.default, init_param.empty,
                             "parameter %s for %s has no default value"
                             % (init_param.name, type(estimator).__name__))
            assert_in(type(init_param.default),
                      [str, int, float, bool, tuple, type(None),
                       np.float64, types.FunctionType, Memory])
            if init_param.name not in params.keys():
                # deprecated parameter, not in get_params
                assert_true(init_param.default is None)
                continue

            if (issubclass(Estimator, BaseSGD) and
                    init_param.name in ['tol', 'max_iter']):
                # To remove in 0.21, when they get their future default values
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
### 25 - sklearn/ensemble/forest.py:

Start line: 128, End line: 158

```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(BaseForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
```
### 29 - sklearn/ensemble/forest.py:

Start line: 391, End line: 422

```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):
    """Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(ForestClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
```
### 30 - sklearn/utils/estimator_checks.py:

Start line: 1884, End line: 1919

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=9)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert_equal(hash(new_value), hash(original_value),
                     "Estimator %s should not change or mutate "
                     " the parameter %s from %s to %s during fit."
                     % (name, param_name, original_value, new_value))
```
### 36 - examples/ensemble/plot_random_forest_regression_multioutput.py:

Start line: 1, End line: 77

```python
"""
============================================================
Comparing random forests and the multi-output meta estimator
============================================================

An example to compare multi-output regression with random forest and
the :ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator.

This example illustrates the use of the
:ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator
to perform multi-output regression. A random forest regressor is used,
which supports multi-output regression natively, so the results can be
compared.

The random forest regressor will only ever predict values within the
range of observations or closer to zero for each of the targets. As a
result the predictions are biased towards the centre of the circle.

Using a single underlying feature the model learns both the
x and y coordinate as output.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=400,
                                                    random_state=4)

max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
            c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
```
### 38 - sklearn/utils/estimator_checks.py:

Start line: 1082, End line: 1150

```python
@ignore_warnings(category=DeprecationWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    rnd = np.random.RandomState(0)
    X_train_finite = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                                  estimator_orig)
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    error_string_fit = "Estimator doesn't check for NaN and inf in fit."
    error_string_predict = ("Estimator doesn't check for NaN and inf in"
                            " predict.")
    error_string_transform = ("Estimator doesn't check for NaN and inf in"
                              " transform.")
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            try:
                estimator.fit(X_train, y)
            except ValueError as e:
                if 'inf' not in repr(e) and 'NaN' not in repr(e):
                    print(error_string_fit, estimator, e)
                    traceback.print_exc(file=sys.stdout)
                    raise e
            except Exception as exc:
                print(error_string_fit, estimator, exc)
                traceback.print_exc(file=sys.stdout)
                raise exc
            else:
                raise AssertionError(error_string_fit, estimator)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                try:
                    estimator.predict(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_predict, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_predict, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_predict, estimator)

            # transform
            if hasattr(estimator, "transform"):
                try:
                    estimator.transform(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_transform, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_transform, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_transform, estimator)
```
### 45 - sklearn/utils/estimator_checks.py:

Start line: 638, End line: 686

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [key for key in dict_after_fit.keys()
                             if is_public_parameter(key)]

    attrs_added_by_fit = [key for key in public_keys_after_fit
                          if key not in dict_before_fit.keys()]

    # check that fit doesn't add any public attribute
    assert_true(not attrs_added_by_fit,
                ('Estimator adds public attribute(s) during'
                 ' the fit method.'
                 ' Estimators are only allowed to add private attributes'
                 ' either started with _ or ended'
                 ' with _ but %s added' % ', '.join(attrs_added_by_fit)))

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [key for key in public_keys_after_fit
                            if (dict_before_fit[key]
                                is not dict_after_fit[key])]

    assert_true(not attrs_changed_by_fit,
                ('Estimator changes public attribute(s) during'
                 ' the fit method. Estimators are only allowed'
                 ' to change attributes started'
                 ' or ended with _, but'
                 ' %s changed' % ', '.join(attrs_changed_by_fit)))
```
### 46 - sklearn/ensemble/forest.py:

Start line: 424, End line: 467

```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):

    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_
```
### 58 - sklearn/ensemble/forest.py:

Start line: 469, End line: 515

```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, six.string_types):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample". Given "%s".'
                                     % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight("balanced", '
                         'classes, y). In place of y you can use a large '
                         'enough sample of the full training set target to '
                         'properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight != 'balanced_subsample' or
                    not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight,
                                                              y_original)

        return y, expanded_class_weight
```
### 61 - sklearn/utils/estimator_checks.py:

Start line: 1, End line: 82

```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
import struct
from functools import partial

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from sklearn.externals.six.moves import zip
from sklearn.externals.joblib import hash, Memory
from sklearn.utils.testing import assert_raises, _get_args
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import META_ESTIMATORS
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_dict_equal
from sklearn.utils.testing import create_memmap_backed_data
from sklearn.utils import is_scalar_nan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, ClusterMixin,
                          BaseEstimator, is_classifier, is_regressor,
                          is_outlier_detector)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.fixes import signature
from sklearn.utils.validation import (has_fit_parameter, _num_samples,
                                      LARGE_SPARSE_SUPPORTED)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']

ALLOW_NAN = ['Imputer', 'SimpleImputer', 'ChainedImputer',
             'MaxAbsScaler', 'MinMaxScaler', 'RobustScaler', 'StandardScaler',
             'PowerTransformer', 'QuantileTransformer']
```
### 65 - sklearn/utils/estimator_checks.py:

Start line: 1454, End line: 1516

```python
def check_outliers_train(name, estimator_orig, readonly_memmap=True):
    X, _ = make_blobs(n_samples=300, random_state=0)
    X = shuffle(X, random_state=7)

    if readonly_memmap:
        X = create_memmap_backed_data(X)

    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    # fit
    estimator.fit(X)
    # with lists
    estimator.fit(X.tolist())

    y_pred = estimator.predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    decision = estimator.decision_function(X)
    assert decision.dtype == np.dtype('float')

    score = estimator.score_samples(X)
    assert score.dtype == np.dtype('float')

    # raises error on malformed input for predict
    assert_raises(ValueError, estimator.predict, X.T)

    # decision_function agrees with predict
    decision = estimator.decision_function(X)
    assert decision.shape == (n_samples,)
    dec_pred = (decision >= 0).astype(np.int)
    dec_pred[dec_pred == 0] = -1
    assert_array_equal(dec_pred, y_pred)

    # raises error on malformed input for decision_function
    assert_raises(ValueError, estimator.decision_function, X.T)

    # decision_function is a translation of score_samples
    y_scores = estimator.score_samples(X)
    assert y_scores.shape == (n_samples,)
    y_dec = y_scores - estimator.offset_
    assert_array_equal(y_dec, decision)

    # raises error on malformed input for score_samples
    assert_raises(ValueError, estimator.score_samples, X.T)

    # contamination parameter (not for OneClassSVM which has the nu parameter)
    if hasattr(estimator, "contamination"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'
        contamination = 0.1
        estimator.set_params(contamination=contamination)
        estimator.fit(X)
        y_pred = estimator.predict(X)
        assert_almost_equal(np.mean(y_pred != 1), contamination)

        # raises error when contamination is a scalar and not in [0,1]
        for contamination in [-0.5, 2.3]:
            estimator.set_params(contamination=contamination)
            assert_raises(ValueError, estimator.fit, X)
```
### 70 - sklearn/utils/estimator_checks.py:

Start line: 1785, End line: 1827

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_classifiers(name, classifier_orig):
    if name == "NuSVC":
        # the sparse version has a parameter that doesn't do anything
        raise SkipTest("Not testing NuSVC class weight as it is ignored.")
    if name.endswith("NB"):
        # NaiveBayes classifiers have a somewhat different interface.
        # FIXME SOON!
        raise SkipTest

    for n_centers in [2, 3]:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0)

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _is_pairwise(classifier_orig):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(
            class_weight=class_weight)
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
        #      0.88 (Issue #9111)
        assert_greater(np.mean(y_pred == 0), 0.87)
```
### 88 - sklearn/utils/estimator_checks.py:

Start line: 1688, End line: 1713

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_int(name, regressor_orig):
    X, _ = _boston_subset()
    X = pairwise_estimator_convert_X(X[:50], regressor_orig)
    rnd = np.random.RandomState(0)
    y = rnd.randint(3, size=X.shape[0])
    y = multioutput_estimator_convert_y_2d(regressor_orig, y)
    rnd = np.random.RandomState(0)
    # separate estimators to control random seeds
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    # fit
    regressor_1.fit(X, y_)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y_.astype(np.float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)
```
### 99 - sklearn/ensemble/forest.py:

Start line: 358, End line: 388

```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        check_is_fitted(self, 'estimators_')

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   backend="threading")(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_)

        return sum(all_importances) / len(self.estimators_)


# This is a utility function for joblib's Parallel. It can't go locally in
# ForestClassifier or ForestRegressor, because joblib complains that it cannot
# pickle it when placed there.

def accumulate_prediction(predict, X, out, lock):
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]
```
### 107 - sklearn/ensemble/forest.py:

Start line: 342, End line: 356

```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return self.estimators_[0]._validate_X_predict(X, check_input=True)
```
### 112 - sklearn/utils/estimator_checks.py:

Start line: 1033, End line: 1056

```python
@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_32 = pairwise_estimator_convert_X(X_train_32, estimator_orig)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = X_train_int_64[:, 0]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        if name == 'PowerTransformer':
            # Box-Cox requires positive, non-zero data
            X_train = np.abs(X_train) + 1
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)
```
