# scikit-learn__scikit-learn-14309

| **scikit-learn/scikit-learn** | `f7e082d24ef9f3f9dea14ad82a9a8b2351715f54` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4388 |
| **Any found context length** | 4388 |
| **Avg pos** | 5.0 |
| **Min pos** | 5 |
| **Max pos** | 5 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/inspection/partial_dependence.py b/sklearn/inspection/partial_dependence.py
--- a/sklearn/inspection/partial_dependence.py
+++ b/sklearn/inspection/partial_dependence.py
@@ -286,9 +286,15 @@ def partial_dependence(estimator, X, features, response_method='auto',
         raise ValueError(
             "'estimator' must be a fitted regressor or classifier.")
 
-    if (hasattr(estimator, 'classes_') and
-            isinstance(estimator.classes_[0], np.ndarray)):
-        raise ValueError('Multiclass-multioutput estimators are not supported')
+    if is_classifier(estimator):
+        if not hasattr(estimator, 'classes_'):
+            raise ValueError(
+                "'estimator' parameter must be a fitted estimator"
+            )
+        if isinstance(estimator.classes_[0], np.ndarray):
+            raise ValueError(
+                'Multiclass-multioutput estimators are not supported'
+            )
 
     X = check_array(X)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/inspection/partial_dependence.py | 289 | 291 | 5 | 1 | 4388


## Problem Statement

```
 plot_partial_dependence() fails when used on DecisionTreeRegressor
<!--
If your issue is a usage question, submit it here instead:
- StackOverflow with the scikit-learn tag: https://stackoverflow.com/questions/tagged/scikit-learn
- Mailing List: https://mail.python.org/mailman/listinfo/scikit-learn
For more information, see User Questions: http://scikit-learn.org/stable/support.html#user-questions
-->

<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
\`\`\`sklearn.inspection.plot_partial_dependence()\`\`\` fails when using a \`\`\`sklearn.tree.DecisionTreeRegressor\`\`\` as the estimator. The problem appears to be related to the presence of a \`\`\`classes_\`\`\` attribute (with a value of \`\`\`None\`\`\`) on the estimator, despite it being a regressor and not a classifier. Deleting the \`\`\`classes_\`\`\` attribute from the estimator allows \`\`\`plot_partial_dependence()\`\`\` to successfully run.

#### Steps/Code to Reproduce
\`\`\`python

from sklearn.inspection import plot_partial_dependence
from sklearn.tree import DecisionTreeRegressor
import numpy as np
X = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[3.0], [7.0]])
learn = DecisionTreeRegressor().fit(X, y)
assert getattr(learn, 'classes_') is None
delete_classes_attribute = False
if delete_classes_attribute:
    # Deleting the 'classes_' attribute will allow plot_partial_dependence() to run
    delattr(learn, 'classes_')
plot_partial_dependence(learn, X, features=[0])


\`\`\`
<!--
Example:
\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = ["Help I have a bug" for i in range(1000)]

vectorizer = CountVectorizer(input=docs, analyzer='word')
lda_features = vectorizer.fit_transform(docs)

lda_model = LatentDirichletAllocation(
    n_topics=10,
    learning_method='online',
    evaluate_every=10,
    n_jobs=4,
)
model = lda_model.fit(lda_features)
\`\`\`
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
No error is thrown.
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->
A \`\`\`TypeError\`\`\` is thrown:
\`\`\`Python traceback
Traceback (most recent call last):
  File "Partial Dependence Plot Bug Illustration.py", line 13, in <module>
    plot_partial_dependence(learn, X, features=[0])
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/sklearn/inspection/partial_dependence.py", line 561, in plot_partial_dependence
    for fxs in features)
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/parallel.py", line 921, in __call__
    if self.dispatch_one_batch(iterator):
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/parallel.py", line 759, in dispatch_one_batch
    self._dispatch(tasks)
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/parallel.py", line 716, in _dispatch
    job = self._backend.apply_async(batch, callback=cb)
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 182, in apply_async
    result = ImmediateResult(func)
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/_parallel_backends.py", line 549, in __init__
    self.results = batch()
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/parallel.py", line 225, in __call__
    for func, args, kwargs in self.items]
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/joblib/parallel.py", line 225, in <listcomp>
    for func, args, kwargs in self.items]
  File "/anaconda3/envs/newsklearn/lib/python3.7/site-packages/sklearn/inspection/partial_dependence.py", line 293, in partial_dependence
    isinstance(estimator.classes_[0], np.ndarray)):
TypeError: 'NoneType' object is not subscriptable
\`\`\`
#### Versions
<!--
Please run the following snippet and paste the output below.
For scikit-learn >= 0.20:
import sklearn; sklearn.show_versions()
For scikit-learn < 0.20:
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->
\`\`\`
System:
    python: 3.7.3 (default, Mar 27 2019, 16:54:48)  [Clang 4.0.1 (tags/RELEASE_401/final)]
executable: /anaconda3/envs/newsklearn/bin/python
   machine: Darwin-18.5.0-x86_64-i386-64bit

BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /anaconda3/envs/newsklearn/lib
cblas_libs: mkl_rt, pthread

Python deps:
       pip: 19.1.1
setuptools: 41.0.1
   sklearn: 0.21.2
     numpy: 1.16.4
     scipy: 1.2.1
    Cython: None
    pandas: 0.24.2
\`\`\`

<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/inspection/partial_dependence.py** | 490 | 574| 817 | 817 | 6498 | 
| 2 | **1 sklearn/inspection/partial_dependence.py** | 367 | 489| 1456 | 2273 | 6498 | 
| 3 | **1 sklearn/inspection/partial_dependence.py** | 614 | 665| 654 | 2927 | 6498 | 
| 4 | 2 examples/inspection/plot_partial_dependence.py | 1 | 92| 760 | 3687 | 8153 | 
| **-> 5 <-** | **2 sklearn/inspection/partial_dependence.py** | 285 | 364| 701 | 4388 | 8153 | 
| 6 | **2 sklearn/inspection/partial_dependence.py** | 575 | 613| 413 | 4801 | 8153 | 
| 7 | **2 sklearn/inspection/partial_dependence.py** | 1 | 28| 162 | 4963 | 8153 | 
| 8 | 3 sklearn/ensemble/partial_dependence.py | 370 | 422| 716 | 5679 | 12133 | 
| 9 | 3 examples/inspection/plot_partial_dependence.py | 93 | 168| 764 | 6443 | 12133 | 
| 10 | 3 sklearn/ensemble/partial_dependence.py | 266 | 348| 816 | 7259 | 12133 | 
| 11 | **3 sklearn/inspection/partial_dependence.py** | 166 | 283| 1343 | 8602 | 12133 | 
| 12 | 3 sklearn/ensemble/partial_dependence.py | 349 | 369| 324 | 8926 | 12133 | 
| 13 | **3 sklearn/inspection/partial_dependence.py** | 96 | 163| 598 | 9524 | 12133 | 
| 14 | 3 sklearn/ensemble/partial_dependence.py | 1 | 29| 111 | 9635 | 12133 | 
| 15 | 3 sklearn/ensemble/partial_dependence.py | 182 | 265| 870 | 10505 | 12133 | 
| 16 | 3 sklearn/ensemble/partial_dependence.py | 148 | 179| 402 | 10907 | 12133 | 
| 17 | 3 examples/inspection/plot_partial_dependence.py | 169 | 183| 130 | 11037 | 12133 | 
| 18 | 3 sklearn/ensemble/partial_dependence.py | 80 | 146| 751 | 11788 | 12133 | 
| 19 | 4 sklearn/utils/estimator_checks.py | 1455 | 1554| 914 | 12702 | 34327 | 
| 20 | 5 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 86| 769 | 13471 | 35480 | 
| 21 | 6 sklearn/tree/tree.py | 201 | 274| 804 | 14275 | 49003 | 
| 22 | 6 sklearn/tree/tree.py | 898 | 1088| 1941 | 16216 | 49003 | 
| 23 | 7 examples/tree/plot_tree_regression.py | 1 | 52| 413 | 16629 | 49416 | 
| 24 | 8 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 17196 | 49983 | 
| 25 | 9 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 17736 | 51507 | 
| 26 | 9 sklearn/utils/estimator_checks.py | 1300 | 1323| 185 | 17921 | 51507 | 
| 27 | 10 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 775 | 18696 | 53331 | 
| 28 | 10 sklearn/tree/tree.py | 276 | 361| 800 | 19496 | 53331 | 
| 29 | 10 sklearn/tree/tree.py | 123 | 199| 670 | 20166 | 53331 | 
| 30 | 10 sklearn/utils/estimator_checks.py | 1 | 57| 412 | 20578 | 53331 | 
| 31 | 10 sklearn/tree/tree.py | 1 | 65| 373 | 20951 | 53331 | 
| 32 | 10 sklearn/tree/tree.py | 533 | 744| 2217 | 23168 | 53331 | 
| 33 | 10 sklearn/utils/estimator_checks.py | 1886 | 1907| 210 | 23378 | 53331 | 
| 34 | 10 sklearn/utils/estimator_checks.py | 2377 | 2409| 348 | 23726 | 53331 | 
| 35 | 11 examples/applications/plot_out_of_core_classification.py | 187 | 214| 241 | 23967 | 56636 | 
| 36 | 12 examples/text/plot_document_classification_20newsgroups.py | 247 | 323| 658 | 24625 | 59171 | 
| 37 | 12 examples/applications/plot_out_of_core_classification.py | 267 | 360| 807 | 25432 | 59171 | 
| 38 | 13 benchmarks/bench_tree.py | 1 | 61| 347 | 25779 | 60041 | 
| 39 | 14 sklearn/tree/export.py | 79 | 178| 864 | 26643 | 68114 | 
| 40 | 15 examples/plot_anomaly_comparison.py | 81 | 152| 757 | 27400 | 69632 | 
| 41 | 16 benchmarks/bench_mnist.py | 84 | 105| 306 | 27706 | 71343 | 
| 42 | 17 examples/tree/plot_iris_dtc.py | 1 | 72| 597 | 28303 | 71940 | 
| 43 | 18 examples/ensemble/plot_forest_iris.py | 1 | 71| 586 | 28889 | 73413 | 
| 44 | 19 examples/preprocessing/plot_discretization.py | 1 | 87| 779 | 29668 | 74222 | 
| 45 | 19 sklearn/tree/tree.py | 1089 | 1114| 191 | 29859 | 74222 | 
| 46 | 19 sklearn/utils/estimator_checks.py | 1180 | 1248| 597 | 30456 | 74222 | 
| 47 | 19 sklearn/utils/estimator_checks.py | 1841 | 1883| 462 | 30918 | 74222 | 
| 48 | 20 examples/ensemble/plot_adaboost_regression.py | 1 | 55| 389 | 31307 | 74639 | 
| 49 | 20 sklearn/utils/estimator_checks.py | 1577 | 1649| 655 | 31962 | 74639 | 
| 50 | 21 benchmarks/bench_isolation_forest.py | 54 | 160| 1025 | 32987 | 76100 | 
| 51 | 22 examples/classification/plot_classifier_comparison.py | 1 | 78| 633 | 33620 | 77452 | 
| 52 | 22 sklearn/utils/estimator_checks.py | 1813 | 1838| 273 | 33893 | 77452 | 
| 53 | 23 examples/ensemble/plot_isolation_forest.py | 1 | 71| 598 | 34491 | 78050 | 
| 54 | 24 examples/ensemble/plot_bias_variance.py | 116 | 192| 684 | 35175 | 79864 | 
| 55 | 25 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 59 | 124| 700 | 35875 | 81046 | 
| 56 | 25 examples/ensemble/plot_forest_iris.py | 73 | 161| 887 | 36762 | 81046 | 
| 57 | 26 benchmarks/bench_hist_gradient_boosting.py | 1 | 37| 349 | 37111 | 83272 | 
| 58 | 26 sklearn/tree/tree.py | 362 | 384| 201 | 37312 | 83272 | 
| 59 | 27 sklearn/__init__.py | 70 | 93| 280 | 37592 | 84216 | 
| 60 | 28 examples/linear_model/plot_theilsen.py | 1 | 88| 785 | 38377 | 85220 | 
| 61 | 29 examples/ensemble/plot_voting_decision_regions.py | 1 | 74| 644 | 39021 | 85864 | 
| 62 | 29 sklearn/tree/tree.py | 1484 | 1508| 176 | 39197 | 85864 | 
| 63 | 30 sklearn/cross_decomposition/pls_.py | 289 | 351| 838 | 40035 | 94058 | 
| 64 | 31 examples/tree/plot_unveil_tree_structure.py | 1 | 96| 826 | 40861 | 95199 | 
| 65 | 31 sklearn/utils/estimator_checks.py | 493 | 548| 546 | 41407 | 95199 | 
| 66 | 31 examples/tree/plot_unveil_tree_structure.py | 98 | 135| 316 | 41723 | 95199 | 
| 67 | 32 examples/linear_model/plot_ard.py | 1 | 101| 769 | 42492 | 96123 | 
| 68 | 33 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 42777 | 96917 | 
| 69 | 34 examples/classification/plot_lda.py | 38 | 69| 325 | 43102 | 97501 | 
| 70 | 35 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 90| 749 | 43851 | 98467 | 
| 71 | 36 sklearn/tree/__init__.py | 1 | 15| 107 | 43958 | 98574 | 
| 72 | 37 examples/ensemble/plot_ensemble_oob.py | 1 | 87| 677 | 44635 | 99303 | 
| 73 | 37 sklearn/tree/tree.py | 1343 | 1483| 1416 | 46051 | 99303 | 
| 74 | 37 sklearn/utils/estimator_checks.py | 1434 | 1453| 231 | 46282 | 99303 | 
| 75 | 38 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 47041 | 100786 | 
| 76 | 38 benchmarks/bench_tree.py | 64 | 125| 523 | 47564 | 100786 | 
| 77 | 38 examples/linear_model/plot_theilsen.py | 89 | 112| 196 | 47760 | 100786 | 
| 78 | 38 examples/text/plot_document_classification_20newsgroups.py | 1 | 119| 783 | 48543 | 100786 | 
| 79 | 39 examples/decomposition/plot_faces_decomposition.py | 63 | 141| 623 | 49166 | 102230 | 
| 80 | 39 sklearn/utils/estimator_checks.py | 1910 | 1959| 498 | 49664 | 102230 | 
| 81 | 40 examples/impute/plot_missing_values.py | 1 | 48| 397 | 50061 | 103466 | 
| 82 | 41 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 50607 | 104040 | 
| 83 | 42 examples/ensemble/plot_forest_importances.py | 1 | 55| 373 | 50980 | 104413 | 
| 84 | 43 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 358 | 51338 | 118070 | 
| 85 | 44 examples/linear_model/plot_sgd_early_stopping.py | 89 | 150| 534 | 51872 | 119367 | 
| 86 | 44 examples/ensemble/plot_bias_variance.py | 69 | 96| 220 | 52092 | 119367 | 
| 87 | 45 benchmarks/bench_covertype.py | 112 | 190| 757 | 52849 | 121258 | 
| 88 | 45 examples/impute/plot_iterative_imputer_variants_comparison.py | 87 | 133| 384 | 53233 | 121258 | 
| 89 | 45 examples/preprocessing/plot_discretization_classification.py | 109 | 192| 870 | 54103 | 121258 | 
| 90 | 46 examples/ensemble/plot_adaboost_multiclass.py | 1 | 89| 748 | 54851 | 122272 | 
| 91 | 47 examples/ensemble/plot_voting_regressor.py | 1 | 54| 363 | 55214 | 122635 | 
| 92 | 47 sklearn/tree/tree.py | 745 | 772| 202 | 55416 | 122635 | 
| 93 | 47 benchmarks/bench_mnist.py | 108 | 180| 695 | 56111 | 122635 | 
| 94 | 48 sklearn/ensemble/forest.py | 1542 | 1719| 1734 | 57845 | 140453 | 
| 95 | 49 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 58599 | 141672 | 
| 96 | 49 benchmarks/bench_covertype.py | 99 | 109| 151 | 58750 | 141672 | 
| 97 | 49 examples/ensemble/plot_adaboost_multiclass.py | 90 | 119| 243 | 58993 | 141672 | 
| 98 | 50 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 772 | 59765 | 142593 | 
| 99 | 51 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 61079 | 143934 | 
| 100 | 51 sklearn/utils/estimator_checks.py | 903 | 934| 289 | 61368 | 143934 | 
| 101 | 51 examples/cross_decomposition/plot_compare_cross_decomposition.py | 1 | 82| 724 | 62092 | 143934 | 
| 102 | 52 examples/model_selection/plot_learning_curve.py | 102 | 140| 481 | 62573 | 145626 | 
| 103 | 53 examples/decomposition/plot_pca_vs_fa_model_selection.py | 88 | 126| 437 | 63010 | 146747 | 
| 104 | 54 examples/ensemble/plot_feature_transformation.py | 1 | 87| 761 | 63771 | 147895 | 
| 105 | 55 examples/ensemble/plot_voting_probas.py | 1 | 84| 781 | 64552 | 148676 | 
| 106 | 56 sklearn/ensemble/_hist_gradient_boosting/predictor.py | 67 | 83| 133 | 64685 | 149175 | 
| 107 | 57 sklearn/multiclass.py | 219 | 273| 513 | 65198 | 155607 | 
| 108 | 58 examples/cluster/plot_inductive_clustering.py | 59 | 121| 493 | 65691 | 156516 | 
| 109 | 58 sklearn/utils/estimator_checks.py | 830 | 868| 371 | 66062 | 156516 | 
| 110 | 59 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 612 | 66674 | 157153 | 
| 111 | 60 examples/ensemble/plot_random_forest_embedding.py | 1 | 84| 757 | 67431 | 158115 | 
| 112 | 60 sklearn/tree/tree.py | 1160 | 1314| 1664 | 69095 | 158115 | 
| 113 | 61 examples/applications/plot_outlier_detection_housing.py | 79 | 134| 633 | 69728 | 159516 | 
| 114 | 61 examples/ensemble/plot_bias_variance.py | 1 | 64| 761 | 70489 | 159516 | 
| 115 | 62 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 71065 | 160651 | 
| 116 | 62 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 71624 | 160651 | 
| 117 | 62 examples/impute/plot_missing_values.py | 51 | 96| 446 | 72070 | 160651 | 
| 118 | 63 sklearn/linear_model/coordinate_descent.py | 8 | 27| 146 | 72216 | 181212 | 
| 119 | 64 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 564 | 778| 322 | 72538 | 190554 | 
| 120 | 64 examples/decomposition/plot_faces_decomposition.py | 142 | 194| 377 | 72915 | 190554 | 
| 121 | 65 examples/datasets/plot_random_dataset.py | 1 | 68| 645 | 73560 | 191199 | 
| 122 | 65 sklearn/ensemble/forest.py | 1285 | 1495| 2163 | 75723 | 191199 | 
| 123 | 66 examples/plot_multilabel.py | 53 | 93| 436 | 76159 | 192294 | 
| 124 | 67 examples/ensemble/plot_adaboost_twoclass.py | 1 | 85| 705 | 76864 | 193164 | 
| 125 | 67 sklearn/ensemble/forest.py | 1033 | 1240| 2077 | 78941 | 193164 | 
| 126 | 68 examples/applications/plot_species_distribution_modeling.py | 128 | 206| 796 | 79737 | 194935 | 
| 127 | 68 sklearn/utils/estimator_checks.py | 1728 | 1769| 417 | 80154 | 194935 | 
| 128 | 69 examples/classification/plot_lda_qda.py | 57 | 103| 573 | 80727 | 196542 | 
| 129 | 69 sklearn/utils/estimator_checks.py | 2133 | 2153| 200 | 80927 | 196542 | 
| 130 | 70 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 81623 | 197266 | 
| 131 | 71 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 95 | 109| 174 | 81797 | 198250 | 
| 132 | 71 examples/plot_anomaly_comparison.py | 1 | 80| 718 | 82515 | 198250 | 
| 133 | 71 examples/linear_model/plot_robust_fit.py | 1 | 73| 509 | 83024 | 198250 | 
| 134 | 72 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 83521 | 199279 | 
| 135 | 72 examples/decomposition/plot_pca_vs_fa_model_selection.py | 1 | 85| 661 | 84182 | 199279 | 
| 136 | 72 sklearn/utils/estimator_checks.py | 673 | 724| 431 | 84613 | 199279 | 


## Patch

```diff
diff --git a/sklearn/inspection/partial_dependence.py b/sklearn/inspection/partial_dependence.py
--- a/sklearn/inspection/partial_dependence.py
+++ b/sklearn/inspection/partial_dependence.py
@@ -286,9 +286,15 @@ def partial_dependence(estimator, X, features, response_method='auto',
         raise ValueError(
             "'estimator' must be a fitted regressor or classifier.")
 
-    if (hasattr(estimator, 'classes_') and
-            isinstance(estimator.classes_[0], np.ndarray)):
-        raise ValueError('Multiclass-multioutput estimators are not supported')
+    if is_classifier(estimator):
+        if not hasattr(estimator, 'classes_'):
+            raise ValueError(
+                "'estimator' parameter must be a fitted estimator"
+            )
+        if isinstance(estimator.classes_[0], np.ndarray):
+            raise ValueError(
+                'Multiclass-multioutput estimators are not supported'
+            )
 
     X = check_array(X)
 

```

## Test Patch

```diff
diff --git a/sklearn/inspection/tests/test_partial_dependence.py b/sklearn/inspection/tests/test_partial_dependence.py
--- a/sklearn/inspection/tests/test_partial_dependence.py
+++ b/sklearn/inspection/tests/test_partial_dependence.py
@@ -21,6 +21,7 @@
 from sklearn.linear_model import LinearRegression
 from sklearn.linear_model import LogisticRegression
 from sklearn.linear_model import MultiTaskLasso
+from sklearn.tree import DecisionTreeRegressor
 from sklearn.datasets import load_boston, load_iris
 from sklearn.datasets import make_classification, make_regression
 from sklearn.cluster import KMeans
@@ -58,6 +59,7 @@
     (GradientBoostingClassifier, 'brute', multiclass_classification_data),
     (GradientBoostingRegressor, 'recursion', regression_data),
     (GradientBoostingRegressor, 'brute', regression_data),
+    (DecisionTreeRegressor, 'brute', regression_data),
     (LinearRegression, 'brute', regression_data),
     (LinearRegression, 'brute', multioutput_regression_data),
     (LogisticRegression, 'brute', binary_classification_data),
@@ -261,7 +263,6 @@ def test_partial_dependence_easy_target(est, power):
     assert r2 > .99
 
 
-@pytest.mark.filterwarnings('ignore:The default value of ')  # 0.22
 @pytest.mark.parametrize('Estimator',
                          (sklearn.tree.DecisionTreeClassifier,
                           sklearn.tree.ExtraTreeClassifier,
@@ -288,6 +289,8 @@ def test_multiclass_multioutput(Estimator):
 
 class NoPredictProbaNoDecisionFunction(BaseEstimator, ClassifierMixin):
     def fit(self, X, y):
+        # simulate that we have some classes
+        self.classes_ = [0, 1]
         return self
 
 

```


## Code snippets

### 1 - sklearn/inspection/partial_dependence.py:

Start line: 490, End line: 574

```python
def plot_partial_dependence(estimator, X, features, feature_names=None,
                            target=None, response_method='auto', n_cols=3,
                            grid_resolution=100, percentiles=(0.05, 0.95),
                            method='auto', n_jobs=None, verbose=0, fig=None,
                            line_kw=None, contour_kw=None):
    check_matplotlib_support('plot_partial_dependence')  # noqa
    import matplotlib.pyplot as plt  # noqa
    from matplotlib import transforms  # noqa
    from matplotlib.ticker import MaxNLocator  # noqa
    from matplotlib.ticker import ScalarFormatter  # noqa

    # set target_idx for multi-class estimators
    if hasattr(estimator, 'classes_') and np.size(estimator.classes_) > 2:
        if target is None:
            raise ValueError('target must be specified for multi-class')
        target_idx = np.searchsorted(estimator.classes_, target)
        if (not (0 <= target_idx < len(estimator.classes_)) or
                estimator.classes_[target_idx] != target):
            raise ValueError('target not in est.classes_, got {}'.format(
                target))
    else:
        # regression and binary classification
        target_idx = 0

    X = check_array(X)
    n_features = X.shape[1]

    # convert feature_names to list
    if feature_names is None:
        # if feature_names is None, use feature indices as name
        feature_names = [str(i) for i in range(n_features)]
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('feature_names should not contain duplicates.')

    def convert_feature(fx):
        if isinstance(fx, str):
            try:
                fx = feature_names.index(fx)
            except ValueError:
                raise ValueError('Feature %s not in feature_names' % fx)
        return int(fx)

    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral, str)):
            fxs = (fxs,)
        try:
            fxs = [convert_feature(fx) for fx in fxs]
        except TypeError:
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')
        if not (1 <= np.size(fxs) <= 2):
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')

        tmp_features.append(fxs)

    features = tmp_features

    names = []
    try:
        for fxs in features:
            names_ = []
            # explicit loop so "i" is bound for exception below
            for i in fxs:
                names_.append(feature_names[i])
            names.append(names_)
    except IndexError:
        raise ValueError('All entries of features must be less than '
                         'len(feature_names) = {0}, got {1}.'
                         .format(len(feature_names), i))

    # compute averaged predictions
    pd_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(estimator, X, fxs,
                                    response_method=response_method,
                                    method=method,
                                    grid_resolution=grid_resolution,
                                    percentiles=percentiles)
        for fxs in features)

    # For multioutput regression, we can only check the validity of target
    # now that we have the predictions.
    # Also note: as multiclass-multioutput classifiers are not supported,
    # multiclass and multioutput scenario are mutually exclusive. So there is
    # no risk of overwriting target_idx here.
    avg_preds, _ = pd_result[0]  # checking the first result is enough
    # ... other code
```
### 2 - sklearn/inspection/partial_dependence.py:

Start line: 367, End line: 489

```python
def plot_partial_dependence(estimator, X, features, feature_names=None,
                            target=None, response_method='auto', n_cols=3,
                            grid_resolution=100, percentiles=(0.05, 0.95),
                            method='auto', n_jobs=None, verbose=0, fig=None,
                            line_kw=None, contour_kw=None):
    """Partial dependence plots.

    The ``len(features)`` plots are arranged in a grid with ``n_cols``
    columns. Two-way partial dependence plots are plotted as contour plots.

    Read more in the :ref:`User Guide <partial_dependence>`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing `predict`, `predict_proba`,
        or `decision_function`. Multioutput-multiclass classifiers are not
        supported.
    X : array-like, shape (n_samples, n_features)
        The data to use to build the grid of values on which the dependence
        will be evaluated. This is usually the training data.
    features : list of {int, str, pair of int, pair of str}
        The target features for which to create the PDPs.
        If features[i] is an int or a string, a one-way PDP is created; if
        features[i] is a tuple, a two-way PDP is created. Each tuple must be
        of size 2.
        if any entry is a string, then it must be in ``feature_names``.
    feature_names : seq of str, shape (n_features,), optional
        Name of each feature; feature_names[i] holds the name of the feature
        with index i. By default, the name of the feature corresponds to
        their numerical index.
    target : int, optional (default=None)
        - In a multiclass setting, specifies the class for which the PDPs
          should be computed. Note that for binary classification, the
          positive class (index 1) is always used.
        - In a multioutput setting, specifies the task for which the PDPs
          should be computed
        Ignored in binary classification or classical regression settings.
    response_method : 'auto', 'predict_proba' or 'decision_function', \
            optional (default='auto') :
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.
    n_cols : int, optional (default=3)
        The maximum number of columns in the grid plot.
    grid_resolution : int, optional (default=100)
        The number of equally spaced points on the axes of the plots, for each
        target feature.
    percentiles : tuple of float, optional (default=(0.05, 0.95))
        The lower and upper percentile used to create the extreme values
        for the PDP axes. Must be in [0, 1].
    method : str, optional (default='auto')
        The method to use to calculate the partial dependence predictions:

        - 'recursion' is only supported for gradient boosting estimator (namely
          :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`,
          :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`,
          :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`,
          :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`)
          but is more efficient in terms of speed.
          With this method, ``X`` is optional and is only used to build the
          grid and the partial dependences are computed using the training
          data. This method does not account for the ``init`` predicor of
          the boosting process, which may lead to incorrect values (see
          warning below. With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities.

        - 'brute' is supported for any estimator, but is more
          computationally intensive.

        - 'auto':
          - 'recursion' is used for estimators that supports it.
          - 'brute' is used for all other estimators.
    n_jobs : int, optional (default=None)
        The number of CPUs to use to compute the partial dependences.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, optional (default=0)
        Verbose output during PD computations.
    fig : Matplotlib figure object, optional (default=None)
        A figure object onto which the plots will be drawn, after the figure
        has been cleared. By default, a new one is created.
    line_kw : dict, optional
        Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.
        For one-way partial dependence plots.
    contour_kw : dict, optional
        Dict with keywords passed to the ``matplotlib.pyplot.plot`` call.
        For two-way partial dependence plots.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> X, y = make_friedman1()
    >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
    >>> plot_partial_dependence(clf, X, [0, (0, 1)]) #doctest: +SKIP

    See also
    --------
    sklearn.inspection.partial_dependence: Return raw partial
      dependence values

    Warnings
    --------
    The 'recursion' method only works for gradient boosting estimators, and
    unlike the 'brute' method, it does not account for the ``init``
    predictor of the boosting process. In practice this will produce the
    same values as 'brute' up to a constant offset in the target response,
    provided that ``init`` is a consant estimator (which is the default).
    However, as soon as ``init`` is not a constant estimator, the partial
    dependence values are incorrect for 'recursion'. This is not relevant for
    :class:`HistGradientBoostingClassifier
    <sklearn.ensemble.HistGradientBoostingClassifier>` and
    :class:`HistGradientBoostingRegressor
    <sklearn.ensemble.HistGradientBoostingRegressor>`, which do not have an
    ``init`` parameter.
    """
    # ... other code
```
### 3 - sklearn/inspection/partial_dependence.py:

Start line: 614, End line: 665

```python
def plot_partial_dependence(estimator, X, features, feature_names=None,
                            target=None, response_method='auto', n_cols=3,
                            grid_resolution=100, percentiles=(0.05, 0.95),
                            method='auto', n_jobs=None, verbose=0, fig=None,
                            line_kw=None, contour_kw=None):
    # ... other code
    for i, fx, name, (avg_preds, values) in zip(
            count(), features, names, pd_result):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        if len(values) == 1:
            ax.plot(values[0], avg_preds[target_idx].ravel(), **line_kw)
        else:
            # make contour plot
            assert len(values) == 2
            XX, YY = np.meshgrid(values[0], values[1])
            Z = avg_preds[target_idx].T
            CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5,
                            colors='k')
            ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1],
                        vmin=Z_level[0], alpha=0.75, **contour_kw)
            ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)

        # plot data deciles + axes labels
        deciles = mquantiles(X[:, fx[0]], prob=np.arange(0.1, 1.0, 0.1))
        trans = transforms.blended_transform_factory(ax.transData,
                                                     ax.transAxes)
        ylim = ax.get_ylim()
        ax.vlines(deciles, [0], 0.05, transform=trans, color='k')
        ax.set_xlabel(name[0])
        ax.set_ylim(ylim)

        # prevent x-axis ticks from overlapping
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
        tick_formatter = ScalarFormatter()
        tick_formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(tick_formatter)

        if len(values) > 1:
            # two-way PDP - y-axis deciles + labels
            deciles = mquantiles(X[:, fx[1]], prob=np.arange(0.1, 1.0, 0.1))
            trans = transforms.blended_transform_factory(ax.transAxes,
                                                         ax.transData)
            xlim = ax.get_xlim()
            ax.hlines(deciles, [0], 0.05, transform=trans, color='k')
            ax.set_ylabel(name[1])
            # hline erases xlim
            ax.set_xlim(xlim)
        else:
            ax.set_ylabel('Partial dependence')

        if len(values) == 1:
            ax.set_ylim(pdp_lim[1])
        axs.append(ax)

    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.4,
                        hspace=0.3)
```
### 4 - examples/inspection/plot_partial_dependence.py:

Start line: 1, End line: 92

```python
"""
========================
Partial Dependence Plots
========================

Partial dependence plots show the dependence between the target function [2]_
and a set of 'target' features, marginalizing over the
values of all other features (the complement features). Due to the limits
of human perception the size of the target feature set must be small (usually,
one or two) thus the target features are usually chosen among the most
important features.

This example shows how to obtain partial dependence plots from a
:class:`~sklearn.neural_network.MLPRegressor` and a
:class:`~sklearn.ensemble.HistGradientBoostingRegressor` trained on the
California housing dataset. The example is taken from [1]_.

The plots show four 1-way and two 1-way partial dependence plots (ommitted for
:class:`~sklearn.neural_network.MLPRegressor` due to computation time).
The target variables for the one-way PDP are: median income (`MedInc`),
average occupants per household (`AvgOccup`), median house age (`HouseAge`),
and average rooms per household (`AveRooms`).

.. [1] T. Hastie, R. Tibshirani and J. Friedman,
    "Elements of Statistical Learning Ed. 2", Springer, 2009.

.. [2] For classification you can think of it as the regression score before
       the link function.
"""
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets.california_housing import fetch_california_housing


##############################################################################
# California Housing data preprocessing
# -------------------------------------
#
# Center target to avoid gradient boosting init bias: gradient boosting
# with the 'recursion' method does not account for the initial estimator
# (here the average target, by default)
#
cal_housing = fetch_california_housing()
names = cal_housing.feature_names
X, y = cal_housing.data, cal_housing.target

y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)

##############################################################################
# Partial Dependence computation for multi-layer perceptron
# ---------------------------------------------------------
#
# Let's fit a MLPRegressor and compute single-variable partial dependence
# plots

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(50, 50),
                                 learning_rate_init=0.01,
                                 max_iter=200,
                                 early_stopping=True,
                                 n_iter_no_change=10,
                                 validation_fraction=0.1))
est.fit(X_train, y_train)
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))

print('Computing partial dependence plots...')
tic = time()
# We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
# with the brute method.
features = [0, 5, 1, 2]
plot_partial_dependence(est, X_train, features, feature_names=names,
                        n_jobs=3, grid_resolution=20)
```
### 5 - sklearn/inspection/partial_dependence.py:

Start line: 285, End line: 364

```python
def partial_dependence(estimator, X, features, response_method='auto',
                       percentiles=(0.05, 0.95), grid_resolution=100,
                       method='auto'):

    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError(
            "'estimator' must be a fitted regressor or classifier.")

    if (hasattr(estimator, 'classes_') and
            isinstance(estimator.classes_[0], np.ndarray)):
        raise ValueError('Multiclass-multioutput estimators are not supported')

    X = check_array(X)

    accepted_responses = ('auto', 'predict_proba', 'decision_function')
    if response_method not in accepted_responses:
        raise ValueError(
            'response_method {} is invalid. Accepted response_method names '
            'are {}.'.format(response_method, ', '.join(accepted_responses)))

    if is_regressor(estimator) and response_method != 'auto':
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )
    accepted_methods = ('brute', 'recursion', 'auto')
    if method not in accepted_methods:
        raise ValueError(
            'method {} is invalid. Accepted method names are {}.'.format(
                method, ', '.join(accepted_methods)))

    if method == 'auto':
        if (isinstance(estimator, BaseGradientBoosting) and
                estimator.init is None):
            method = 'recursion'
        elif isinstance(estimator, BaseHistGradientBoosting):
            method = 'recursion'
        else:
            method = 'brute'

    if method == 'recursion':
        if not isinstance(estimator,
                          (BaseGradientBoosting, BaseHistGradientBoosting)):
            supported_classes_recursion = (
                'GradientBoostingClassifier',
                'GradientBoostingRegressor',
                'HistGradientBoostingClassifier',
                'HistGradientBoostingRegressor',
            )
            raise ValueError(
                "Only the following estimators support the 'recursion' "
                "method: {}. Try using method='brute'."
                .format(', '.join(supported_classes_recursion)))
        if response_method == 'auto':
            response_method = 'decision_function'

        if response_method != 'decision_function':
            raise ValueError(
                "With the 'recursion' method, the response_method must be "
                "'decision_function'. Got {}.".format(response_method)
            )

    n_features = X.shape[1]
    features = np.asarray(features, dtype=np.int32, order='C').ravel()
    if any(not (0 <= f < n_features) for f in features):
        raise ValueError('all features must be in [0, %d]'
                         % (n_features - 1))

    grid, values = _grid_from_X(X[:, features], percentiles,
                                grid_resolution)
    if method == 'brute':
        averaged_predictions = _partial_dependence_brute(estimator, grid,
                                                         features, X,
                                                         response_method)
    else:
        averaged_predictions = _partial_dependence_recursion(estimator, grid,
                                                             features)

    # reshape averaged_predictions to
    # (n_outputs, n_values_feature_0, n_values_feature_1, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values])

    return averaged_predictions, values
```
### 6 - sklearn/inspection/partial_dependence.py:

Start line: 575, End line: 613

```python
def plot_partial_dependence(estimator, X, features, feature_names=None,
                            target=None, response_method='auto', n_cols=3,
                            grid_resolution=100, percentiles=(0.05, 0.95),
                            method='auto', n_jobs=None, verbose=0, fig=None,
                            line_kw=None, contour_kw=None):
    # ... other code
    if is_regressor(estimator) and avg_preds.shape[0] > 1:
        if target is None:
            raise ValueError(
                'target must be specified for multi-output regressors')
        if not 0 <= target <= avg_preds.shape[0]:
            raise ValueError(
                'target must be in [0, n_tasks], got {}.'.format(target))
        target_idx = target
    else:
        target_idx = 0

    # get global min and max values of PD grouped by plot type
    pdp_lim = {}
    for avg_preds, values in pd_result:
        min_pd = avg_preds[target_idx].min()
        max_pd = avg_preds[target_idx].max()
        n_fx = len(values)
        old_min_pd, old_max_pd = pdp_lim.get(n_fx, (min_pd, max_pd))
        min_pd = min(min_pd, old_min_pd)
        max_pd = max(max_pd, old_max_pd)
        pdp_lim[n_fx] = (min_pd, max_pd)

    # create contour levels for two-way plots
    if 2 in pdp_lim:
        Z_level = np.linspace(*pdp_lim[2], num=8)

    if fig is None:
        fig = plt.figure()
    else:
        fig.clear()

    if line_kw is None:
        line_kw = {'color': 'green'}
    if contour_kw is None:
        contour_kw = {}

    n_cols = min(n_cols, len(features))
    n_rows = int(np.ceil(len(features) / float(n_cols)))
    axs = []
    # ... other code
```
### 7 - sklearn/inspection/partial_dependence.py:

Start line: 1, End line: 28

```python
"""Partial dependence plots for regression and classification models."""

from itertools import count
import numbers
from collections.abc import Iterable

import numpy as np
from scipy.stats.mstats import mquantiles
from joblib import Parallel, delayed

from ..base import is_classifier, is_regressor
from ..utils.extmath import cartesian
from ..utils import check_array
from ..utils import check_matplotlib_support  # noqa
from ..utils.validation import check_is_fitted
from ..tree._tree import DTYPE
from ..exceptions import NotFittedError
from ..ensemble.gradient_boosting import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting)


__all__ = ['partial_dependence', 'plot_partial_dependence']
```
### 8 - sklearn/ensemble/partial_dependence.py:

Start line: 370, End line: 422

```python
@deprecated("The function ensemble.plot_partial_dependence has been "
            "deprecated in favour of "
            "sklearn.inspection.plot_partial_dependence in "
            " 0.21 and will be removed in 0.23.")
def plot_partial_dependence(gbrt, X, features, feature_names=None,
                            label=None, n_cols=3, grid_resolution=100,
                            percentiles=(0.05, 0.95), n_jobs=None,
                            verbose=0, ax=None, line_kw=None,
                            contour_kw=None, **fig_kw):
    # ... other code
    for i, fx, name, (pdp, axes) in zip(count(), features, names,
                                        pd_result):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        if len(axes) == 1:
            ax.plot(axes[0], pdp[label_idx].ravel(), **line_kw)
        else:
            # make contour plot
            assert len(axes) == 2
            XX, YY = np.meshgrid(axes[0], axes[1])
            Z = pdp[label_idx].reshape(list(map(np.size, axes))).T
            CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5,
                            colors='k')
            ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1],
                        vmin=Z_level[0], alpha=0.75, **contour_kw)
            ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)

        # plot data deciles + axes labels
        deciles = mquantiles(X[:, fx[0]], prob=np.arange(0.1, 1.0, 0.1))
        trans = transforms.blended_transform_factory(ax.transData,
                                                     ax.transAxes)
        ylim = ax.get_ylim()
        ax.vlines(deciles, [0], 0.05, transform=trans, color='k')
        ax.set_xlabel(name[0])
        ax.set_ylim(ylim)

        # prevent x-axis ticks from overlapping
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
        tick_formatter = ScalarFormatter()
        tick_formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(tick_formatter)

        if len(axes) > 1:
            # two-way PDP - y-axis deciles + labels
            deciles = mquantiles(X[:, fx[1]], prob=np.arange(0.1, 1.0, 0.1))
            trans = transforms.blended_transform_factory(ax.transAxes,
                                                         ax.transData)
            xlim = ax.get_xlim()
            ax.hlines(deciles, [0], 0.05, transform=trans, color='k')
            ax.set_ylabel(name[1])
            # hline erases xlim
            ax.set_xlim(xlim)
        else:
            ax.set_ylabel('Partial dependence')

        if len(axes) == 1:
            ax.set_ylim(pdp_lim[1])
        axs.append(ax)

    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.4,
                        hspace=0.3)
    return fig, axs
```
### 9 - examples/inspection/plot_partial_dependence.py:

Start line: 93, End line: 168

```python
print("done in {:.3f}s".format(time() - tic))
fig = plt.gcf()
fig.suptitle('Partial dependence of house value on non-location features\n'
             'for the California housing dataset, with MLPRegressor')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

##############################################################################
# Partial Dependence computation for Gradient Boosting
# ----------------------------------------------------
#
# Let's now fit a GradientBoostingRegressor and compute the partial dependence
# plots either or one or two variables at a time.

print("Training GradientBoostingRegressor...")
tic = time()
est = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=64,
                                    learning_rate=0.1, random_state=1)
est.fit(X_train, y_train)
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))

print('Computing partial dependence plots...')
tic = time()
features = [0, 5, 1, 2, (5, 1)]
plot_partial_dependence(est, X_train, features, feature_names=names,
                        n_jobs=3, grid_resolution=20)
print("done in {:.3f}s".format(time() - tic))
fig = plt.gcf()
fig.suptitle('Partial dependence of house value on non-location features\n'
             'for the California housing dataset, with Gradient Boosting')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

##############################################################################
# Analysis of the plots
# ---------------------
#
# We can clearly see that the median house price shows a linear relationship
# with the median income (top left) and that the house price drops when the
# average occupants per household increases (top middle).
# The top right plot shows that the house age in a district does not have
# a strong influence on the (median) house price; so does the average rooms
# per household.
# The tick marks on the x-axis represent the deciles of the feature values
# in the training data.
#
# We also observe that :class:`~sklearn.neural_network.MLPRegressor` has much
# smoother predictions than
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`. For the plots to be
# comparable, it is necessary to subtract the average value of the target
# ``y``: The 'recursion' method, used by default for
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`, does not account
# for the initial predictor (in our case the average target). Setting the
# target average to 0 avoids this bias.
#
# Partial dependence plots with two target features enable us to visualize
# interactions among them. The two-way partial dependence plot shows the
# dependence of median house price on joint values of house age and average
# occupants per household. We can clearly see an interaction between the
# two features: for an average occupancy greater than two, the house price is
# nearly independent of the house age, whereas for values less than two there
# is a strong dependence on age.

##############################################################################
# 3D interaction plots
# --------------------
#
# Let's make the same partial dependence plot for the 2 features interaction,
# this time in 3 dimensions.

fig = plt.figure()

target_feature = (1, 5)
pdp, axes = partial_dependence(est, X_train, target_feature,
                               grid_resolution=20)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].T
```
### 10 - sklearn/ensemble/partial_dependence.py:

Start line: 266, End line: 348

```python
@deprecated("The function ensemble.plot_partial_dependence has been "
            "deprecated in favour of "
            "sklearn.inspection.plot_partial_dependence in "
            " 0.21 and will be removed in 0.23.")
def plot_partial_dependence(gbrt, X, features, feature_names=None,
                            label=None, n_cols=3, grid_resolution=100,
                            percentiles=(0.05, 0.95), n_jobs=None,
                            verbose=0, ax=None, line_kw=None,
                            contour_kw=None, **fig_kw):
    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import ScalarFormatter

    if not isinstance(gbrt, BaseGradientBoosting):
        raise ValueError('gbrt has to be an instance of BaseGradientBoosting')
    check_is_fitted(gbrt, 'estimators_')

    # set label_idx for multi-class GBRT
    if hasattr(gbrt, 'classes_') and np.size(gbrt.classes_) > 2:
        if label is None:
            raise ValueError('label is not given for multi-class PDP')
        label_idx = np.searchsorted(gbrt.classes_, label)
        if gbrt.classes_[label_idx] != label:
            raise ValueError('label %s not in ``gbrt.classes_``' % str(label))
    else:
        # regression and binary classification
        label_idx = 0

    X = check_array(X, dtype=DTYPE, order='C')
    if gbrt.n_features_ != X.shape[1]:
        raise ValueError('X.shape[1] does not match gbrt.n_features_')

    if line_kw is None:
        line_kw = {'color': 'green'}
    if contour_kw is None:
        contour_kw = {}

    # convert feature_names to list
    if feature_names is None:
        # if not feature_names use fx indices as name
        feature_names = [str(i) for i in range(gbrt.n_features_)]
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    def convert_feature(fx):
        if isinstance(fx, str):
            try:
                fx = feature_names.index(fx)
            except ValueError:
                raise ValueError('Feature %s not in feature_names' % fx)
        return fx

    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral, str)):
            fxs = (fxs,)
        try:
            fxs = np.array([convert_feature(fx) for fx in fxs], dtype=np.int32)
        except TypeError:
            raise ValueError('features must be either int, str, or tuple '
                             'of int/str')
        if not (1 <= np.size(fxs) <= 2):
            raise ValueError('target features must be either one or two')

        tmp_features.append(fxs)

    features = tmp_features

    names = []
    try:
        for fxs in features:
            l = []
            # explicit loop so "i" is bound for exception below
            for i in fxs:
                l.append(feature_names[i])
            names.append(l)
    except IndexError:
        raise ValueError('All entries of features must be less than '
                         'len(feature_names) = {0}, got {1}.'
                         .format(len(feature_names), i))

    # compute PD functions
    pd_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(gbrt, fxs, X=X,
                                    grid_resolution=grid_resolution,
                                    percentiles=percentiles)
        for fxs in features)

    # get global min and max values of PD grouped by plot type
    pdp_lim = {}
    # ... other code
```
### 11 - sklearn/inspection/partial_dependence.py:

Start line: 166, End line: 283

```python
def partial_dependence(estimator, X, features, response_method='auto',
                       percentiles=(0.05, 0.95), grid_resolution=100,
                       method='auto'):
    """Partial dependence of ``features``.

    Partial dependence of a feature (or a set of features) corresponds to
    the average response of an estimator for each possible value of the
    feature.

    Read more in the :ref:`User Guide <partial_dependence>`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing `predict`, `predict_proba`,
        or `decision_function`. Multioutput-multiclass classifiers are not
        supported.
    X : array-like, shape (n_samples, n_features)
        ``X`` is used both to generate a grid of values for the
        ``features``, and to compute the averaged predictions when
        method is 'brute'.
    features : list or array-like of int
        The target features for which the partial dependency should be
        computed.
    response_method : 'auto', 'predict_proba' or 'decision_function', \
            optional (default='auto')
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.
    percentiles : tuple of float, optional (default=(0.05, 0.95))
        The lower and upper percentile used to create the extreme values
        for the grid. Must be in [0, 1].
    grid_resolution : int, optional (default=100)
        The number of equally spaced points on the grid, for each target
        feature.
    method : str, optional (default='auto')
        The method used to calculate the averaged predictions:

        - 'recursion' is only supported for gradient boosting estimator (namely
          :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`,
          :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`,
          :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`,
          :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`)
          but is more efficient in terms of speed.
          With this method, ``X`` is only used to build the
          grid and the partial dependences are computed using the training
          data. This method does not account for the ``init`` predicor of
          the boosting process, which may lead to incorrect values (see
          warning below). With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities.

        - 'brute' is supported for any estimator, but is more
          computationally intensive.

        - 'auto':

          - 'recursion' is used for
            :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
            and
            :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
            if ``init=None``, and for
            :class:`HistGradientBoostingClassifier<sklearn.ensemble.HistGradientBoostingClassifier>`
            and
            :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`.
          - 'brute' is used for all other estimators.

    Returns
    -------
    averaged_predictions : ndarray, \
            shape (n_outputs, len(values[0]), len(values[1]), ...)
        The predictions for all the points in the grid, averaged over all
        samples in X (or over the training data if ``method`` is
        'recursion'). ``n_outputs`` corresponds to the number of classes in
        a multi-class setting, or to the number of tasks for multi-output
        regression. For classical regression and binary classification
        ``n_outputs==1``. ``n_values_feature_j`` corresponds to the size
        ``values[j]``.
    values : seq of 1d ndarrays
        The values with which the grid has been created. The generated grid
        is a cartesian product of the arrays in ``values``. ``len(values) ==
        len(features)``. The size of each array ``values[j]`` is either
        ``grid_resolution``, or the number of unique values in ``X[:, j]``,
        whichever is smaller.

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])

    See also
    --------
    sklearn.inspection.plot_partial_dependence: Plot partial dependence

    Warnings
    --------
    The 'recursion' method only works for gradient boosting estimators, and
    unlike the 'brute' method, it does not account for the ``init``
    predictor of the boosting process. In practice this will produce the
    same values as 'brute' up to a constant offset in the target response,
    provided that ``init`` is a consant estimator (which is the default).
    However, as soon as ``init`` is not a constant estimator, the partial
    dependence values are incorrect for 'recursion'. This is not relevant for
    :class:`HistGradientBoostingClassifier
    <sklearn.ensemble.HistGradientBoostingClassifier>` and
    :class:`HistGradientBoostingRegressor
    <sklearn.ensemble.HistGradientBoostingRegressor>`, which do not have an
    ``init`` parameter.
    """
    # ... other code
```
### 13 - sklearn/inspection/partial_dependence.py:

Start line: 96, End line: 163

```python
def _partial_dependence_recursion(est, grid, features):
    return est._compute_partial_dependence_recursion(grid, features)


def _partial_dependence_brute(est, grid, features, X, response_method):
    averaged_predictions = []

    # define the prediction_method (predict, predict_proba, decision_function).
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)
        if response_method == 'auto':
            # try predict_proba, then decision_function if it doesn't exist
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = (predict_proba if response_method ==
                                 'predict_proba' else decision_function)
        if prediction_method is None:
            if response_method == 'auto':
                raise ValueError(
                    'The estimator has no predict_proba and no '
                    'decision_function method.'
                )
            elif response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            else:
                raise ValueError(
                    'The estimator has no decision_function method.')

    for new_values in grid:
        X_eval = X.copy()
        for i, variable in enumerate(features):
            X_eval[:, variable] = new_values[i]

        try:
            predictions = prediction_method(X_eval)
        except NotFittedError:
            raise ValueError(
                "'estimator' parameter must be a fitted estimator")

        # Note: predictions is of shape
        # (n_points,) for non-multioutput regressors
        # (n_points, n_tasks) for multioutput regressors
        # (n_points, 1) for the regressors in cross_decomposition (I think)
        # (n_points, 2)  for binary classifaction
        # (n_points, n_classes) for multiclass classification

        # average over samples
        averaged_predictions.append(np.mean(predictions, axis=0))

    # reshape to (n_targets, n_points) where n_targets is:
    # - 1 for non-multioutput regression and binary classification (shape is
    #   already correct in those cases)
    # - n_tasks for multi-output regression
    # - n_classes for multiclass classification.
    averaged_predictions = np.array(averaged_predictions).T
    if is_regressor(est) and averaged_predictions.ndim == 1:
        # non-multioutput regression, shape is (n_points,)
        averaged_predictions = averaged_predictions.reshape(1, -1)
    elif is_classifier(est) and averaged_predictions.shape[0] == 2:
        # Binary classification, shape is (2, n_points).
        # we output the effect of **positive** class
        averaged_predictions = averaged_predictions[1]
        averaged_predictions = averaged_predictions.reshape(1, -1)

    return averaged_predictions
```
