# scikit-learn__scikit-learn-13368

| **scikit-learn/scikit-learn** | `afd432137fd840adc182f0bad87f405cb80efac7` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3341 |
| **Any found context length** | 3341 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/model_selection/_validation.py b/sklearn/model_selection/_validation.py
--- a/sklearn/model_selection/_validation.py
+++ b/sklearn/model_selection/_validation.py
@@ -876,10 +876,11 @@ def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
             float_min = np.finfo(predictions.dtype).min
             default_values = {'decision_function': float_min,
                               'predict_log_proba': float_min,
-                              'predict_proba': 0}
+                              'predict_proba': 0.0}
             predictions_for_all_classes = np.full((_num_samples(predictions),
                                                    n_classes),
-                                                  default_values[method])
+                                                  default_values[method],
+                                                  predictions.dtype)
             predictions_for_all_classes[:, estimator.classes_] = predictions
             predictions = predictions_for_all_classes
     return predictions, test

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/model_selection/_validation.py | 879 | 882 | 7 | 2 | 3341


## Problem Statement

```
cross_val_predict returns bad prediction when evaluated on a dataset with very few samples
#### Description
`cross_val_predict` returns bad prediction when evaluated on a dataset with very few samples on 1 class, causing class being ignored on some CV splits.

#### Steps/Code to Reproduce
\`\`\`python
from sklearn.datasets import *
from sklearn.linear_model import *
from sklearn.model_selection import *
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
# Change the first sample to a new class
y[0] = 2
clf = LogisticRegression()
cv = StratifiedKFold(n_splits=2, random_state=1)
train, test = list(cv.split(X, y))
yhat_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
print(yhat_proba)
\`\`\`

#### Expected Results
\`\`\`
[[0.06105412 0.93894588 0.        ]
 [0.92512247 0.07487753 0.        ]
 [0.93896471 0.06103529 0.        ]
 [0.04345507 0.95654493 0.        ]
\`\`\`

#### Actual Results
\`\`\`
[[0. 0. 0.        ]
 [0. 0. 0.        ]
 [0. 0. 0.        ]
 [0. 0. 0.        ]
\`\`\`
#### Versions
Verified on the scikit latest dev version.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 381 | 1315 | 
| 2 | **2 sklearn/model_selection/_validation.py** | 750 | 782| 348 | 729 | 14384 | 
| 3 | 3 sklearn/model_selection/_split.py | 621 | 671| 583 | 1312 | 33778 | 
| 4 | 4 examples/model_selection/plot_cv_predict.py | 1 | 29| 189 | 1501 | 33967 | 
| 5 | 5 sklearn/utils/estimator_checks.py | 1719 | 1760| 417 | 1918 | 55902 | 
| 6 | **5 sklearn/model_selection/_validation.py** | 640 | 749| 961 | 2879 | 55902 | 
| **-> 7 <-** | **5 sklearn/model_selection/_validation.py** | 840 | 885| 462 | 3341 | 55902 | 
| 8 | 5 sklearn/model_selection/_split.py | 1343 | 1410| 835 | 4176 | 55902 | 
| 9 | 5 sklearn/model_selection/_split.py | 1609 | 1670| 758 | 4934 | 55902 | 
| 10 | 5 examples/model_selection/plot_cv_indices.py | 1 | 48| 368 | 5302 | 55902 | 
| 11 | 6 examples/exercises/plot_cv_diabetes.py | 1 | 80| 676 | 5978 | 56578 | 
| 12 | 6 sklearn/model_selection/_split.py | 559 | 619| 584 | 6562 | 56578 | 
| 13 | 6 sklearn/model_selection/_split.py | 1216 | 1266| 483 | 7045 | 56578 | 
| 14 | 7 sklearn/dummy.py | 160 | 233| 617 | 7662 | 60981 | 
| 15 | 8 examples/feature_selection/plot_rfe_with_cross_validation.py | 1 | 38| 300 | 7962 | 61281 | 
| 16 | 9 examples/model_selection/plot_roc_crossval.py | 1 | 87| 740 | 8702 | 62215 | 
| 17 | 9 sklearn/model_selection/_split.py | 348 | 421| 673 | 9375 | 62215 | 
| 18 | 10 examples/feature_selection/plot_permutation_test_for_classification.py | 1 | 70| 498 | 9873 | 62739 | 
| 19 | 11 sklearn/neighbors/classification.py | 133 | 173| 345 | 10218 | 66127 | 
| 20 | 11 sklearn/utils/estimator_checks.py | 1373 | 1404| 258 | 10476 | 66127 | 
| 21 | 11 sklearn/dummy.py | 478 | 513| 289 | 10765 | 66127 | 
| 22 | 12 sklearn/svm/classes.py | 1215 | 1234| 138 | 10903 | 77559 | 
| 23 | 13 examples/datasets/plot_random_dataset.py | 1 | 68| 645 | 11548 | 78204 | 
| 24 | 14 examples/model_selection/plot_nested_cross_validation_iris.py | 1 | 73| 640 | 12188 | 79278 | 
| 25 | 14 examples/model_selection/plot_cv_indices.py | 64 | 102| 419 | 12607 | 79278 | 
| 26 | 14 sklearn/model_selection/_split.py | 1163 | 1213| 502 | 13109 | 79278 | 
| 27 | 15 examples/model_selection/plot_grid_search_digits.py | 1 | 78| 629 | 13738 | 79907 | 
| 28 | 15 sklearn/model_selection/_split.py | 1970 | 1994| 142 | 13880 | 79907 | 
| 29 | 16 sklearn/ensemble/gradient_boosting.py | 306 | 326| 150 | 14030 | 101148 | 
| 30 | 16 sklearn/utils/estimator_checks.py | 1546 | 1618| 655 | 14685 | 101148 | 
| 31 | 16 examples/model_selection/plot_nested_cross_validation_iris.py | 74 | 119| 433 | 15118 | 101148 | 
| 32 | 16 sklearn/model_selection/_split.py | 119 | 167| 467 | 15585 | 101148 | 
| 33 | 16 sklearn/utils/estimator_checks.py | 1424 | 1523| 940 | 16525 | 101148 | 
| 34 | **16 sklearn/model_selection/_validation.py** | 216 | 255| 409 | 16934 | 101148 | 
| 35 | 17 sklearn/datasets/samples_generator.py | 222 | 254| 368 | 17302 | 115158 | 
| 36 | 18 examples/impute/plot_missing_values.py | 49 | 93| 440 | 17742 | 116362 | 
| 37 | 19 benchmarks/bench_lof.py | 36 | 107| 650 | 18392 | 117278 | 
| 38 | 19 sklearn/model_selection/_split.py | 1135 | 1160| 206 | 18598 | 117278 | 
| 39 | 20 benchmarks/bench_saga.py | 107 | 189| 637 | 19235 | 119744 | 
| 40 | 20 sklearn/utils/estimator_checks.py | 2108 | 2128| 200 | 19435 | 119744 | 
| 41 | 21 sklearn/ensemble/bagging.py | 622 | 650| 231 | 19666 | 127728 | 
| 42 | 22 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 20459 | 128973 | 
| 43 | 22 sklearn/utils/estimator_checks.py | 853 | 878| 234 | 20693 | 128973 | 
| 44 | 22 sklearn/utils/estimator_checks.py | 1763 | 1798| 399 | 21092 | 128973 | 
| 45 | 22 sklearn/utils/estimator_checks.py | 1898 | 1940| 449 | 21541 | 128973 | 
| 46 | 22 sklearn/model_selection/_split.py | 1672 | 1735| 607 | 22148 | 128973 | 
| 47 | 23 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 22941 | 130819 | 
| 48 | 23 sklearn/ensemble/gradient_boosting.py | 2101 | 2119| 150 | 23091 | 130819 | 
| 49 | 24 sklearn/multiclass.py | 110 | 130| 137 | 23228 | 137258 | 
| 50 | 25 sklearn/linear_model/base.py | 276 | 294| 126 | 23354 | 142019 | 
| 51 | 26 examples/datasets/plot_random_multilabel_dataset.py | 1 | 56| 432 | 23786 | 142902 | 
| 52 | 27 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 24283 | 143931 | 
| 53 | 28 examples/classification/plot_classifier_comparison.py | 79 | 145| 710 | 24993 | 145283 | 
| 54 | 29 sklearn/svm/base.py | 307 | 324| 152 | 25145 | 153499 | 
| 55 | 29 sklearn/model_selection/_split.py | 423 | 436| 135 | 25280 | 153499 | 
| 56 | 30 examples/applications/plot_out_of_core_classification.py | 187 | 214| 241 | 25521 | 156804 | 
| 57 | 31 sklearn/model_selection/__init__.py | 1 | 60| 405 | 25926 | 157209 | 
| 58 | 32 examples/model_selection/plot_precision_recall.py | 101 | 205| 770 | 26696 | 159551 | 
| 59 | 33 benchmarks/bench_20newsgroups.py | 1 | 97| 778 | 27474 | 160329 | 
| 60 | 33 sklearn/model_selection/_split.py | 716 | 775| 646 | 28120 | 160329 | 
| 61 | 33 sklearn/svm/base.py | 558 | 575| 146 | 28266 | 160329 | 
| 62 | 34 examples/svm/plot_oneclass.py | 1 | 67| 695 | 28961 | 161024 | 
| 63 | 34 sklearn/model_selection/_split.py | 326 | 345| 119 | 29080 | 161024 | 
| 64 | 34 sklearn/datasets/samples_generator.py | 380 | 404| 295 | 29375 | 161024 | 
| 65 | 35 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 77| 753 | 30128 | 162068 | 
| 66 | 35 sklearn/neighbors/classification.py | 338 | 402| 538 | 30666 | 162068 | 
| 67 | 35 sklearn/utils/estimator_checks.py | 1152 | 1220| 601 | 31267 | 162068 | 
| 68 | 36 examples/calibration/plot_calibration_multiclass.py | 1 | 80| 764 | 32031 | 164199 | 
| 69 | 37 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 32542 | 164732 | 
| 70 | 37 sklearn/utils/estimator_checks.py | 2384 | 2428| 427 | 32969 | 164732 | 
| 71 | 38 sklearn/utils/mocking.py | 108 | 138| 220 | 33189 | 165641 | 
| 72 | 38 sklearn/model_selection/_split.py | 1314 | 1340| 160 | 33349 | 165641 | 
| 73 | 38 sklearn/model_selection/_split.py | 1062 | 1100| 314 | 33663 | 165641 | 
| 74 | 38 sklearn/datasets/samples_generator.py | 154 | 221| 785 | 34448 | 165641 | 
| 75 | 39 sklearn/linear_model/ridge.py | 1337 | 1440| 977 | 35425 | 179026 | 
| 76 | **39 sklearn/model_selection/_validation.py** | 258 | 384| 1143 | 36568 | 179026 | 
| 77 | 40 examples/exercises/plot_cv_digits.py | 1 | 45| 313 | 36881 | 179339 | 


## Patch

```diff
diff --git a/sklearn/model_selection/_validation.py b/sklearn/model_selection/_validation.py
--- a/sklearn/model_selection/_validation.py
+++ b/sklearn/model_selection/_validation.py
@@ -876,10 +876,11 @@ def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
             float_min = np.finfo(predictions.dtype).min
             default_values = {'decision_function': float_min,
                               'predict_log_proba': float_min,
-                              'predict_proba': 0}
+                              'predict_proba': 0.0}
             predictions_for_all_classes = np.full((_num_samples(predictions),
                                                    n_classes),
-                                                  default_values[method])
+                                                  default_values[method],
+                                                  predictions.dtype)
             predictions_for_all_classes[:, estimator.classes_] = predictions
             predictions = predictions_for_all_classes
     return predictions, test

```

## Test Patch

```diff
diff --git a/sklearn/model_selection/tests/test_validation.py b/sklearn/model_selection/tests/test_validation.py
--- a/sklearn/model_selection/tests/test_validation.py
+++ b/sklearn/model_selection/tests/test_validation.py
@@ -975,6 +975,26 @@ def test_cross_val_predict_pandas():
         cross_val_predict(clf, X_df, y_ser)
 
 
+@pytest.mark.filterwarnings('ignore: Default solver will be changed')  # 0.22
+@pytest.mark.filterwarnings('ignore: Default multi_class will')  # 0.22
+def test_cross_val_predict_unbalanced():
+    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
+                               n_informative=2, n_clusters_per_class=1,
+                               random_state=1)
+    # Change the first sample to a new class
+    y[0] = 2
+    clf = LogisticRegression(random_state=1)
+    cv = StratifiedKFold(n_splits=2, random_state=1)
+    train, test = list(cv.split(X, y))
+    yhat_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
+    assert y[test[0]][0] == 2  # sanity check for further assertions
+    assert np.all(yhat_proba[test[0]][:, 2] == 0)
+    assert np.all(yhat_proba[test[0]][:, 0:1] > 0)
+    assert np.all(yhat_proba[test[1]] > 0)
+    assert_array_almost_equal(yhat_proba.sum(axis=1), np.ones(y.shape),
+                              decimal=12)
+
+
 @pytest.mark.filterwarnings('ignore: You should specify a value')  # 0.22
 def test_cross_val_score_sparse_fit_params():
     iris = load_iris()

```


## Code snippets

### 1 - examples/model_selection/plot_cv_indices.py:

Start line: 105, End line: 150

```python
###############################################################################
# Let's see how it looks for the `KFold` cross-validation object:

fig, ax = plt.subplots()
cv = KFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)

###############################################################################
# As you can see, by default the KFold cross-validation iterator does not
# take either datapoint class or group into consideration. We can change this
# by using the ``StratifiedKFold`` like so.

fig, ax = plt.subplots()
cv = StratifiedKFold(n_splits)
plot_cv_indices(cv, X, y, groups, ax, n_splits)

###############################################################################
# In this case, the cross-validation retained the same ratio of classes across
# each CV split. Next we'll visualize this behavior for a number of CV
# iterators.
#
# Visualize cross-validation indices for many CV objects
# ------------------------------------------------------
#
# Let's visually compare the cross validation behavior for many
# scikit-learn cross-validation objects. Below we will loop through several
# common cross-validation objects, visualizing the behavior of each.
#
# Note how some use the group/class information while others do not.

cvs = [KFold, GroupKFold, ShuffleSplit, StratifiedKFold,
       GroupShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit]


for cv in cvs:
    this_cv = cv(n_splits=n_splits)
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_cv_indices(this_cv, X, y, groups, ax, n_splits)

    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=.7)
plt.show()
```
### 2 - sklearn/model_selection/_validation.py:

Start line: 750, End line: 782

```python
def cross_val_predict(estimator, X, y=None, groups=None, cv='warn',
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict'):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
```
### 3 - sklearn/model_selection/_split.py:

Start line: 621, End line: 671

```python
class StratifiedKFold(_BaseKFold):

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        n_samples = y.shape[0]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of members in any class cannot"
                           " be less than n_splits=%d."
                           % (min_groups, self.n_splits)), Warning)

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        # NOTE: Passing the data corresponding to ith class say X[y==class_i]
        # will break when the data is not 100% stratifiable for all classes.
        # So we pass np.zeroes(max(c, n_splits)) as data to the KFold
        per_cls_cvs = [
            KFold(self.n_splits, shuffle=self.shuffle,
                  random_state=rng).split(np.zeros(max(count, self.n_splits)))
            for count in y_counts]

        test_folds = np.zeros(n_samples, dtype=np.int)
        for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_y, per_cls_splits):
                cls_test_folds = test_folds[y == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[y == cls] = cls_test_folds

        return test_folds
```
### 4 - examples/model_selection/plot_cv_predict.py:

Start line: 1, End line: 29

```python
"""
====================================
Plotting Cross-Validated Predictions
====================================

This example shows how to use `cross_val_predict` to visualize prediction
errors.

"""
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```
### 5 - sklearn/utils/estimator_checks.py:

Start line: 1719, End line: 1760

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
### 6 - sklearn/model_selection/_validation.py:

Start line: 640, End line: 749

```python
def cross_val_predict(estimator, X, y=None, groups=None, cv='warn',
                      n_jobs=None, verbose=0, fit_params=None,
                      pre_dispatch='2*n_jobs', method='predict'):
    """Generate cross-validated estimates for each input data point

    It is not appropriate to pass these predictions into an evaluation
    metric. Use :func:`cross_validate` to measure generalization error.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

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

    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``

    See also
    --------
    cross_val_score : calculate score for each CV split

    cross_validate : calculate one or more scores and timings for each CV split

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    # ... other code
```
### 7 - sklearn/model_selection/_validation.py:

Start line: 840, End line: 885

```python
def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
                     method):
    # ... other code
    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
```
### 8 - sklearn/model_selection/_split.py:

Start line: 1343, End line: 1410

```python
class ShuffleSplit(BaseShuffleSplit):
    """Random permutation cross-validator

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=0.1
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default (the parameter is
        unspecified), the value is set to 0.1.
        The default will change in version 0.21. It will remain 0.1 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1, 2, 1, 2])
    >>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    >>> rs.get_n_splits(X)
    5
    >>> print(rs)
    ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
    >>> for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [1 3 0 4] TEST: [5 2]
    TRAIN: [4 0 2 5] TEST: [1 3]
    TRAIN: [1 2 4 0] TEST: [3 5]
    TRAIN: [3 4 1 0] TEST: [5 2]
    TRAIN: [3 5 1 0] TEST: [2 4]
    >>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    >>> for train_index, test_index in rs.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...  # doctest: +ELLIPSIS
    TRAIN: [1 3 0] TEST: [5 2]
    TRAIN: [4 0 2] TEST: [1 3]
    TRAIN: [1 2 4] TEST: [3 5]
    TRAIN: [3 4 1] TEST: [5 2]
    TRAIN: [3 5 1] TEST: [2 4]
    """
```
### 9 - sklearn/model_selection/_split.py:

Start line: 1609, End line: 1670

```python
class StratifiedShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.1.
        The default will change in version 0.21. It will remain 0.1 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    >>> sss.get_n_splits(X, y)
    5
    >>> print(sss)       # doctest: +ELLIPSIS
    StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    >>> for train_index, test_index in sss.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [5 2 3] TEST: [4 1 0]
    TRAIN: [5 1 4] TEST: [0 2 3]
    TRAIN: [5 0 2] TEST: [4 3 1]
    TRAIN: [4 1 0] TEST: [2 3 5]
    TRAIN: [0 5 1] TEST: [3 4 2]
    """
```
### 10 - examples/model_selection/plot_cv_indices.py:

Start line: 1, End line: 48

```python
"""
Visualizing cross-validation behavior in scikit-learn
=====================================================

Choosing the right cross-validation object is a crucial part of fitting a
model properly. There are many ways to split data into training and test
sets in order to avoid model overfitting, to standardize the number of
groups in test sets, etc.

This example visualizes the behavior of several common scikit-learn objects
for comparison.
"""

from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4

###############################################################################
# Visualize our data
# ------------------
#
# First, we must understand the structure of our data. It has 100 randomly
# generated input datapoints, 3 classes split unevenly across datapoints,
# and 10 "groups" split evenly across datapoints.
#
# As we'll see, some cross-validation objects do specific things with
# labeled data, others behave differently with grouped data, and others
# do not use this information.
#
# To begin, we'll visualize our data.

# Generate the class/group data
n_points = 100
X = np.random.randn(100, 10)

percentiles_classes = [.1, .3, .6]
y = np.hstack([[ii] * int(100 * perc)
               for ii, perc in enumerate(percentiles_classes)])

# Evenly spaced groups repeated once
groups = np.hstack([[ii] * 10 for ii in range(10)])
```
### 34 - sklearn/model_selection/_validation.py:

Start line: 216, End line: 255

```python
def cross_validate(estimator, X, y=None, groups=None, scoring=None, cv='warn',
                   n_jobs=None, verbose=0, fit_params=None,
                   pre_dispatch='2*n_jobs', return_train_score=False,
                   return_estimator=False, error_score='raise-deprecating'):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=error_score)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    ret = {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])

    return ret
```
### 76 - sklearn/model_selection/_validation.py:

Start line: 258, End line: 384

```python
def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv='warn',
                    n_jobs=None, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', error_score='raise-deprecating'):
    """Evaluate a score by cross-validation

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

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

    error_score : 'raise' | 'raise-deprecating' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If set to 'raise-deprecating', a FutureWarning is printed before the
        error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        Default is 'raise-deprecating' but from version 0.22 it will change
        to np.nan.

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y, cv=3))  # doctest: +ELLIPSIS
    [0.33150734 0.08022311 0.03531764]

    See Also
    ---------
    :func:`sklearn.model_selection.cross_validate`:
        To run cross-validation on multiple metrics and also to return
        train scores, fit times and score times.

    :func:`sklearn.model_selection.cross_val_predict`:
        Get predictions from each split of cross-validation for diagnostic
        purposes.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                scoring={'score': scorer}, cv=cv,
                                n_jobs=n_jobs, verbose=verbose,
                                fit_params=fit_params,
                                pre_dispatch=pre_dispatch,
                                error_score=error_score)
    return cv_results['test_score']
```
