# scikit-learn__scikit-learn-25805

| **scikit-learn/scikit-learn** | `67ea7206bc052eb752f7881eb6043a00fe27c800` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 3877 |
| **Any found context length** | 3877 |
| **Avg pos** | 12.0 |
| **Min pos** | 5 |
| **Max pos** | 7 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -308,9 +308,6 @@ def fit(self, X, y, sample_weight=None, **fit_params):
         if sample_weight is not None:
             sample_weight = _check_sample_weight(sample_weight, X)
 
-        for sample_aligned_params in fit_params.values():
-            check_consistent_length(y, sample_aligned_params)
-
         # TODO(1.4): Remove when base_estimator is removed
         if self.base_estimator != "deprecated":
             if self.estimator is not None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/calibration.py | 311 | 313 | 7 | 1 | 4715


## Problem Statement

```
CalibratedClassifierCV fails on lgbm fit_params
Hi,

I'm trying to use CalibratedClassifierCV to calibrate the probabilities from a LGBM model. The issue is that when I try CalibratedClassifierCV with eval_set, I get an error ValueError: Found input variables with inconsistent numbers of samples: [43364, 1] which is caused by check_consistent_length function in validation.py The input to eval set is [X_valid,Y_valid] where both X_valid,Y_valid are arrays with different shape. Since this format is a requirement for LGBM eval set https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html, I am not sure how will I make the check_consistent_length pass in my scenario. Full code is given below:

\`\`\`python
import lightgbm as lgbm
from sklearn.calibration import CalibratedClassifierCV

def _train_lgbm_model():
    model = lgbm.LGBMClassifier(**parameters.lgbm_params)
    fit_params = {
        "eval_set": [(X_valid, Y_valid)],
        "eval_names": ["train", "valid"],
        "verbose": 0,
    }
    return model, fit_params

model = CalibratedClassifierCV(model, method='isotonic')
trained_model = model.fit(X_train, Y_train, **fit_param)

Error: ValueError: Found input variables with inconsistent numbers of samples: [43364, 1]

\`\`\` 
X_train.shape = (43364, 152)
X_valid.shape = (43364,)
Y_train.shape = (43364, 152)
Y_valid.shape = (43364,)


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/calibration.py** | 336 | 453| 932 | 932 | 10987 | 
| 2 | **1 sklearn/calibration.py** | 56 | 244| 1916 | 2848 | 10987 | 
| 3 | **1 sklearn/calibration.py** | 246 | 279| 227 | 3075 | 10987 | 
| 4 | 2 benchmarks/bench_rcv1_logreg_convergence.py | 256 | 313| 397 | 3472 | 13033 | 
| **-> 5 <-** | **2 sklearn/calibration.py** | 281 | 335| 405 | 3877 | 13033 | 
| 6 | **2 sklearn/calibration.py** | 993 | 1042| 557 | 4434 | 13033 | 
| **-> 7 <-** | **2 sklearn/calibration.py** | 1 | 511| 281 | 4715 | 13033 | 
| 8 | 3 sklearn/utils/validation.py | 1097 | 1126| 284 | 4999 | 29978 | 
| 9 | 4 examples/calibration/plot_calibration_curve.py | 103 | 197| 853 | 5852 | 32956 | 
| 10 | 5 examples/calibration/plot_calibration.py | 1 | 88| 768 | 6620 | 34220 | 
| 11 | 5 examples/calibration/plot_calibration.py | 89 | 145| 412 | 7032 | 34220 | 
| 12 | 5 examples/calibration/plot_calibration_curve.py | 1 | 102| 702 | 7734 | 34220 | 
| 13 | 6 sklearn/linear_model/_logistic.py | 1847 | 1910| 600 | 8334 | 53578 | 
| 14 | 6 examples/calibration/plot_calibration_curve.py | 297 | 337| 414 | 8748 | 53578 | 
| 15 | 7 examples/calibration/plot_compare_calibration.py | 85 | 211| 1338 | 10086 | 55615 | 
| 16 | 8 sklearn/linear_model/_coordinate_descent.py | 1561 | 1647| 739 | 10825 | 80617 | 
| 17 | 8 sklearn/linear_model/_logistic.py | 2037 | 2071| 479 | 11304 | 80617 | 
| 18 | 8 sklearn/utils/validation.py | 981 | 1096| 1027 | 12331 | 80617 | 
| 19 | 9 sklearn/model_selection/_search_successive_halving.py | 126 | 190| 595 | 12926 | 89927 | 
| 20 | 9 sklearn/linear_model/_coordinate_descent.py | 1648 | 1730| 727 | 13653 | 89927 | 
| 21 | 9 sklearn/linear_model/_logistic.py | 1738 | 1846| 802 | 14455 | 89927 | 
| 22 | 9 sklearn/linear_model/_logistic.py | 1941 | 2035| 859 | 15314 | 89927 | 
| 23 | 9 examples/calibration/plot_calibration_curve.py | 222 | 296| 659 | 15973 | 89927 | 
| 24 | 9 examples/calibration/plot_compare_calibration.py | 1 | 62| 387 | 16360 | 89927 | 
| 25 | **9 sklearn/calibration.py** | 1295 | 1313| 185 | 16545 | 89927 | 
| 26 | 10 sklearn/model_selection/_validation.py | 1890 | 1920| 290 | 16835 | 106041 | 
| 27 | 10 sklearn/model_selection/_validation.py | 658 | 731| 720 | 17555 | 106041 | 
| 28 | 10 sklearn/linear_model/_logistic.py | 1683 | 1736| 398 | 17953 | 106041 | 
| 29 | 10 sklearn/model_selection/_validation.py | 1563 | 1649| 760 | 18713 | 106041 | 
| 30 | 11 examples/calibration/plot_calibration_multiclass.py | 229 | 275| 545 | 19258 | 108904 | 
| 31 | 12 sklearn/linear_model/_stochastic_gradient.py | 141 | 167| 263 | 19521 | 128749 | 
| 32 | 12 sklearn/linear_model/_logistic.py | 1911 | 1940| 395 | 19916 | 128749 | 
| 33 | 12 sklearn/linear_model/_logistic.py | 1406 | 1681| 3127 | 23043 | 128749 | 
| 34 | 13 sklearn/ensemble/_hist_gradient_boosting/grower.py | 335 | 359| 185 | 23228 | 135405 | 
| 35 | 13 sklearn/model_selection/_validation.py | 263 | 332| 571 | 23799 | 135405 | 
| 36 | 13 sklearn/model_selection/_validation.py | 977 | 1035| 592 | 24391 | 135405 | 
| 37 | 14 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 484 | 24875 | 136416 | 
| 38 | 14 sklearn/model_selection/_validation.py | 1 | 332| 308 | 25183 | 136416 | 
| 39 | 15 benchmarks/bench_hist_gradient_boosting_categorical_only.py | 1 | 81| 651 | 25834 | 137067 | 
| 40 | 16 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 182 | 266| 713 | 26547 | 154270 | 
| 41 | 16 sklearn/model_selection/_validation.py | 733 | 770| 428 | 26975 | 154270 | 
| 42 | 17 examples/linear_model/plot_quantile_regression.py | 290 | 317| 219 | 27194 | 157241 | 
| 43 | 18 examples/linear_model/plot_lasso_model_selection.py | 102 | 194| 772 | 27966 | 159374 | 
| 44 | 19 examples/model_selection/plot_likelihood_ratios.py | 104 | 174| 743 | 28709 | 162165 | 
| 45 | 20 sklearn/linear_model/_least_angle.py | 1696 | 2012| 837 | 29546 | 182036 | 
| 46 | 20 sklearn/linear_model/_coordinate_descent.py | 2827 | 2983| 1350 | 30896 | 182036 | 
| 47 | 21 sklearn/base.py | 574 | 610| 452 | 31348 | 190472 | 
| 48 | **21 sklearn/calibration.py** | 514 | 587| 506 | 31854 | 190472 | 


### Hint

```
Once we have metadata routing with would have an explicit way to tell how to broadcast such parameters to the base estimators. However as far as I know, the current state of SLEP6 does not have a way to tell whether or not we want to apply cross-validation indexing and therefore disable the length consistency check for the things that are not meant to be the sample-aligned fit params.

Maybe @adrinjalali knows if this topic was already addressed in SLEP6 or not.
I think this is orthogonal to SLEP006, and a bug in `CalibratedClassifierCV`, which has these lines:

https://github.com/scikit-learn/scikit-learn/blob/c991e30c96ace1565604b429de22e36ed6b1e7bd/sklearn/calibration.py#L311-L312

I'm not sure why this check is there. IMO the logic should be like `GridSearchCV`, where we split the data where the length is the same as `y`, otherwise pass unchanged.

cc @glemaitre 
@adrinjalali - Thanks for your response Adrin, this was my conclusion as well. What would be the path forward to resolve this bug?
I'll open a PR to propose a fix @Sidshroff 
I looked at other estimators and indeed we never check that `fit_params` are sample-aligned but only propagate. I could not find the reasoning in the original PR, we were a bit sloppy regarding this aspect. I think a PR removing this check is fine.
```

## Patch

```diff
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -308,9 +308,6 @@ def fit(self, X, y, sample_weight=None, **fit_params):
         if sample_weight is not None:
             sample_weight = _check_sample_weight(sample_weight, X)
 
-        for sample_aligned_params in fit_params.values():
-            check_consistent_length(y, sample_aligned_params)
-
         # TODO(1.4): Remove when base_estimator is removed
         if self.base_estimator != "deprecated":
             if self.estimator is not None:

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_calibration.py b/sklearn/tests/test_calibration.py
--- a/sklearn/tests/test_calibration.py
+++ b/sklearn/tests/test_calibration.py
@@ -974,23 +974,6 @@ def fit(self, X, y, **fit_params):
         pc_clf.fit(X, y, sample_weight=sample_weight)
 
 
-def test_calibration_with_fit_params_inconsistent_length(data):
-    """fit_params having different length than data should raise the
-    correct error message.
-    """
-    X, y = data
-    fit_params = {"a": y[:5]}
-    clf = CheckingClassifier(expected_fit_params=fit_params)
-    pc_clf = CalibratedClassifierCV(clf)
-
-    msg = (
-        r"Found input variables with inconsistent numbers of "
-        r"samples: \[" + str(N_SAMPLES) + r", 5\]"
-    )
-    with pytest.raises(ValueError, match=msg):
-        pc_clf.fit(X, y, **fit_params)
-
-
 @pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
 @pytest.mark.parametrize("ensemble", [True, False])
 def test_calibrated_classifier_cv_zeros_sample_weights_equivalence(method, ensemble):
@@ -1054,3 +1037,17 @@ def test_calibrated_classifier_deprecation_base_estimator(data):
     warn_msg = "`base_estimator` was renamed to `estimator`"
     with pytest.warns(FutureWarning, match=warn_msg):
         calibrated_classifier.fit(*data)
+
+
+def test_calibration_with_non_sample_aligned_fit_param(data):
+    """Check that CalibratedClassifierCV does not enforce sample alignment
+    for fit parameters."""
+
+    class TestClassifier(LogisticRegression):
+        def fit(self, X, y, sample_weight=None, fit_param=None):
+            assert fit_param is not None
+            return super().fit(X, y, sample_weight=sample_weight)
+
+    CalibratedClassifierCV(estimator=TestClassifier()).fit(
+        *data, fit_param=np.ones(len(data[1]) + 1)
+    )

```


## Code snippets

### 1 - sklearn/calibration.py:

Start line: 336, End line: 453

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None, **fit_params):
        # ... other code
        if self.cv == "prefit":
            # `classes_` should be consistent with that of estimator
            check_is_fitted(self.estimator, attributes=["classes_"])
            self.classes_ = self.estimator.classes_

            pred_method, method_name = _get_prediction_method(estimator)
            n_classes = len(self.classes_)
            predictions = _compute_predictions(pred_method, method_name, X, n_classes)

            calibrated_classifier = _fit_calibrator(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                sample_weight,
            )
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            # Set `classes_` using all `y`
            label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = label_encoder_.classes_
            n_classes = len(self.classes_)

            # sample_weight checks
            fit_parameters = signature(estimator.fit).parameters
            supports_sw = "sample_weight" in fit_parameters
            if sample_weight is not None and not supports_sw:
                estimator_name = type(estimator).__name__
                warnings.warn(
                    f"Since {estimator_name} does not appear to accept sample_weight, "
                    "sample weights will only be used for the calibration itself. This "
                    "can be caused by a limitation of the current scikit-learn API. "
                    "See the following issue for more details: "
                    "https://github.com/scikit-learn/scikit-learn/issues/21134. Be "
                    "warned that the result of the calibration is likely to be "
                    "incorrect."
                )

            # Check that each cross-validation fold can have at least one
            # example per class
            if isinstance(self.cv, int):
                n_folds = self.cv
            elif hasattr(self.cv, "n_splits"):
                n_folds = self.cv.n_splits
            else:
                n_folds = None
            if n_folds and np.any(
                [np.sum(y == class_) < n_folds for class_ in self.classes_]
            ):
                raise ValueError(
                    f"Requesting {n_folds}-fold "
                    "cross-validation but provided less than "
                    f"{n_folds} examples for at least one class."
                )
            cv = check_cv(self.cv, y, classifier=True)

            if self.ensemble:
                parallel = Parallel(n_jobs=self.n_jobs)
                self.calibrated_classifiers_ = parallel(
                    delayed(_fit_classifier_calibrator_pair)(
                        clone(estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        method=self.method,
                        classes=self.classes_,
                        supports_sw=supports_sw,
                        sample_weight=sample_weight,
                        **fit_params,
                    )
                    for train, test in cv.split(X, y)
                )
            else:
                this_estimator = clone(estimator)
                _, method_name = _get_prediction_method(this_estimator)
                fit_params = (
                    {"sample_weight": sample_weight}
                    if sample_weight is not None and supports_sw
                    else None
                )
                pred_method = partial(
                    cross_val_predict,
                    estimator=this_estimator,
                    X=X,
                    y=y,
                    cv=cv,
                    method=method_name,
                    n_jobs=self.n_jobs,
                    fit_params=fit_params,
                )
                predictions = _compute_predictions(
                    pred_method, method_name, X, n_classes
                )

                if sample_weight is not None and supports_sw:
                    this_estimator.fit(X, y, sample_weight=sample_weight)
                else:
                    this_estimator.fit(X, y)
                # Note: Here we don't pass on fit_params because the supported
                # calibrators don't support fit_params anyway
                calibrated_classifier = _fit_calibrator(
                    this_estimator,
                    predictions,
                    y,
                    self.classes_,
                    self.method,
                    sample_weight,
                )
                self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self
```
### 2 - sklearn/calibration.py:

Start line: 56, End line: 244

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Probability calibration with isotonic regression or logistic regression.

    This class uses cross-validation to both estimate the parameters of a
    classifier and subsequently calibrate a classifier. With default
    `ensemble=True`, for each cv split it
    fits a copy of the base estimator to the training subset, and calibrates it
    using the testing subset. For prediction, predicted probabilities are
    averaged across these individual calibrated classifiers. When
    `ensemble=False`, cross-validation is used to obtain unbiased predictions,
    via :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. For prediction, the base estimator, trained using all
    the data, is used. This is the method implemented when `probabilities=True`
    for :mod:`sklearn.svm` estimators.

    Already fitted classifiers can be calibrated via the parameter
    `cv="prefit"`. In this case, no cross-validation is used and all provided
    data is used for calibration. The user has to take care manually that data
    for model fitting and calibration are disjoint.

    The calibration is based on the :term:`decision_function` method of the
    `estimator` if it exists, else on :term:`predict_proba`.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs. The default classifier is
        a :class:`~sklearn.svm.LinearSVC`.

        .. versionadded:: 1.2

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method (i.e. a logistic regression model) or
        'isotonic' which is a non-parametric approach. It is not advised to
        use isotonic calibration with too few calibration samples
        ``(<<1000)`` since it tends to overfit.

    cv : int, cross-validation generator, iterable or "prefit", \
            default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`~sklearn.model_selection.KFold`
        is used.

        Refer to the :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that `estimator` has been
        fitted already and all data is used for calibration.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        Base estimator clones are fitted in parallel across cross-validation
        iterations. Therefore parallelism happens only when `cv != "prefit"`.

        See :term:`Glossary <n_jobs>` for more details.

        .. versionadded:: 0.24

    ensemble : bool, default=True
        Determines how the calibrator is fitted when `cv` is not `'prefit'`.
        Ignored if `cv='prefit'`.

        If `True`, the `estimator` is fitted using training data, and
        calibrated using testing data, for each `cv` fold. The final estimator
        is an ensemble of `n_cv` fitted classifier and calibrator pairs, where
        `n_cv` is the number of cross-validation folds. The output is the
        average predicted probabilities of all pairs.

        If `False`, `cv` is used to compute unbiased predictions, via
        :func:`~sklearn.model_selection.cross_val_predict`, which are then
        used for calibration. At prediction time, the classifier used is the
        `estimator` trained on all the data.
        Note that this method is also internally implemented  in
        :mod:`sklearn.svm` estimators with the `probabilities=True` parameter.

        .. versionadded:: 0.24

    base_estimator : estimator instance
        This parameter is deprecated. Use `estimator` instead.

        .. deprecated:: 1.2
           The parameter `base_estimator` is deprecated in 1.2 and will be
           removed in 1.4. Use `estimator` instead.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    calibrated_classifiers_ : list (len() equal to cv or 1 if `cv="prefit"` \
            or `ensemble=False`)
        The list of classifier and calibrator pairs.

        - When `cv="prefit"`, the fitted `estimator` and fitted
          calibrator.
        - When `cv` is not "prefit" and `ensemble=True`, `n_cv` fitted
          `estimator` and calibrator pairs. `n_cv` is the number of
          cross-validation folds.
        - When `cv` is not "prefit" and `ensemble=False`, the `estimator`,
          fitted on all the data, and fitted calibrator.

        .. versionchanged:: 0.24
            Single calibrated classifier case when `ensemble=False`.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities
        for a calibration curve.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> base_clf = GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
    >>> calibrated_clf.fit(X, y)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    3
    >>> calibrated_clf.predict_proba(X)[:5, :]
    array([[0.110..., 0.889...],
           [0.072..., 0.927...],
           [0.928..., 0.071...],
           [0.928..., 0.071...],
           [0.071..., 0.928...]])
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...        X, y, random_state=42
    ... )
    >>> base_clf = GaussianNB()
    >>> base_clf.fit(X_train, y_train)
    GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit")
    >>> calibrated_clf.fit(X_calib, y_calib)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    1
    >>> calibrated_clf.predict_proba([[-0.5, 0.5]])
    array([[0.936..., 0.063...]])
    """
```
### 3 - sklearn/calibration.py:

Start line: 246, End line: 279

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
        ],
        "method": [StrOptions({"isotonic", "sigmoid"})],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [Integral, None],
        "ensemble": ["boolean"],
        "base_estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
            Hidden(StrOptions({"deprecated"})),
        ],
    }

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        cv=None,
        n_jobs=None,
        ensemble=True,
        base_estimator="deprecated",
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.base_estimator = base_estimator
```
### 4 - benchmarks/bench_rcv1_logreg_convergence.py:

Start line: 256, End line: 313

```python
if lightning_clf is not None and not fit_intercept:
    alpha = 1.0 / C / n_samples
    # compute the same step_size than in LR-sag
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log", fit_intercept)

    clfs.append(
        (
            "Lightning-SVRG",
            lightning_clf.SVRGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )
    clfs.append(
        (
            "Lightning-SAG",
            lightning_clf.SAGClassifier(
                alpha=alpha, eta=step_size, tol=tol, loss="log"
            ),
            sag_iter_range,
            [],
            [],
            [],
            [],
        )
    )

    # We keep only 200 features, to have a dense dataset,
    # and compare to lightning SAG, which seems incorrect in the sparse case.
    X_csc = X.tocsc()
    nnz_in_each_features = X_csc.indptr[1:] - X_csc.indptr[:-1]
    X = X_csc[:, np.argsort(nnz_in_each_features)[-200:]]
    X = X.toarray()
    print("dataset: %.3f MB" % (X.nbytes / 1e6))


# Split training and testing. Switch train and test subset compared to
# LYRL2004 split, to have a larger training dataset.
n = 23149
X_test = X[:n, :]
y_test = y[:n]
X = X[n:, :]
y = y[n:]

clfs = bench(clfs)

plot_train_scores(clfs)
plot_test_scores(clfs)
plot_train_losses(clfs)
plot_dloss(clfs)
plt.show()
```
### 5 - sklearn/calibration.py:

Start line: 281, End line: 335

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the calibrated model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        **fit_params : dict
            Parameters to pass to the `fit` method of the underlying
            classifier.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._validate_params()

        check_classification_targets(y)
        X, y = indexable(X, y)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        for sample_aligned_params in fit_params.values():
            check_consistent_length(y, sample_aligned_params)

        # TODO(1.4): Remove when base_estimator is removed
        if self.base_estimator != "deprecated":
            if self.estimator is not None:
                raise ValueError(
                    "Both `base_estimator` and `estimator` are set. Only set "
                    "`estimator` since `base_estimator` is deprecated."
                )
            warnings.warn(
                "`base_estimator` was renamed to `estimator` in version 1.2 and "
                "will be removed in 1.4.",
                FutureWarning,
            )
            estimator = self.base_estimator
        else:
            estimator = self.estimator

        if estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            estimator = LinearSVC(random_state=0)

        self.calibrated_classifiers_ = []
        # ... other code
```
### 6 - sklearn/calibration.py:

Start line: 993, End line: 1042

```python
def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    normalize="deprecated",
    n_bins=5,
    strategy="uniform",
):
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # TODO(1.3): Remove normalize conditional block.
    if normalize != "deprecated":
        warnings.warn(
            "The normalize argument is deprecated in v1.1 and will be removed in v1.3."
            " Explicitly normalizing y_prob will reproduce this behavior, but it is"
            " recommended that a proper probability is used (i.e. a classifier's"
            " `predict_proba` positive class or `decision_function` output calibrated"
            " with `CalibratedClassifierCV`).",
            FutureWarning,
        )
        if normalize:  # Normalize predicted values into interval [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred
```
### 7 - sklearn/calibration.py:

Start line: 1, End line: 511

```python
"""Calibration of predicted probabilities."""

from numbers import Integral
import warnings
from inspect import signature
from functools import partial

from math import log
import numpy as np

from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    MetaEstimatorMixin,
    is_classifier,
)
from .preprocessing import label_binarize, LabelEncoder
from .utils import (
    column_or_1d,
    indexable,
    check_matplotlib_support,
)

from .utils.multiclass import check_classification_targets
from .utils.parallel import delayed, Parallel
from .utils._param_validation import StrOptions, HasMethods, Hidden
from .utils.validation import (
    _check_fit_params,
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
)
from .utils import _safe_indexing
from .isotonic import IsotonicRegression
from .svm import LinearSVC
from .model_selection import check_cv, cross_val_predict
from .metrics._base import _check_pos_label_consistency
from .metrics._plot.base import _get_response


class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
```
### 8 - sklearn/utils/validation.py:

Start line: 1097, End line: 1126

```python
def check_X_y(
    X,
    y,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
):
    if y is None:
        if estimator is None:
            estimator_name = "estimator"
        else:
            estimator_name = _check_estimator_name(estimator)
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )

    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name="X",
    )

    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)

    check_consistent_length(X, y)

    return X, y
```
### 9 - examples/calibration/plot_calibration_curve.py:

Start line: 103, End line: 197

```python
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()

# %%
# Uncalibrated :class:`~sklearn.naive_bayes.GaussianNB` is poorly calibrated
# because of
# the redundant features which violate the assumption of feature-independence
# and result in an overly confident classifier, which is indicated by the
# typical transposed-sigmoid curve. Calibration of the probabilities of
# :class:`~sklearn.naive_bayes.GaussianNB` with :ref:`isotonic` can fix
# this issue as can be seen from the nearly diagonal calibration curve.
# :ref:`Sigmoid regression <sigmoid_regressor>` also improves calibration
# slightly,
# albeit not as strongly as the non-parametric isotonic regression. This can be
# attributed to the fact that we have plenty of calibration data such that the
# greater flexibility of the non-parametric model can be exploited.
#
# Below we will make a quantitative analysis considering several classification
# metrics: :ref:`brier_score_loss`, :ref:`log_loss`,
# :ref:`precision, recall, F1 score <precision_recall_f_measure_metrics>` and
# :ref:`ROC AUC <roc_metrics>`.

from collections import defaultdict

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

scores = defaultdict(list)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)

    for metric in [brier_score_loss, log_loss]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))

    for metric in [precision_score, recall_score, f1_score, roc_auc_score]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_pred))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df.round(decimals=3)

score_df

# %%
# Notice that although calibration improves the :ref:`brier_score_loss` (a
# metric composed
# of calibration term and refinement term) and :ref:`log_loss`, it does not
# significantly alter the prediction accuracy measures (precision, recall and
# F1 score).
# This is because calibration should not significantly change prediction
# probabilities at the location of the decision threshold (at x = 0.5 on the
# graph). Calibration should however, make the predicted probabilities more
# accurate and thus more useful for making allocation decisions under
# uncertainty.
# Further, ROC AUC, should not change at all because calibration is a
# monotonic transformation. Indeed, no rank metrics are affected by
# calibration.
#
# Linear support vector classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we will compare:
#
# * :class:`~sklearn.linear_model.LogisticRegression` (baseline)
# * Uncalibrated :class:`~sklearn.svm.LinearSVC`. Since SVC does not output
#   probabilities by default, we naively scale the output of the
#   :term:`decision_function` into [0, 1] by applying min-max scaling.
# * :class:`~sklearn.svm.LinearSVC` with isotonic and sigmoid
#   calibration (see :ref:`User Guide <calibration>`)

import numpy as np
```
### 10 - examples/calibration/plot_calibration.py:

Start line: 1, End line: 88

```python
"""
======================================
Probability calibration of classifiers
======================================

When performing classification you often want to predict not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. However, not all
classifiers provide well-calibrated probabilities, some being over-confident
while others being under-confident. Thus, a separate calibration of predicted
probabilities is often desirable as a postprocessing. This example illustrates
two different methods for this calibration and evaluates the quality of the
returned probabilities using Brier's score
(see https://en.wikipedia.org/wiki/Brier_score).

Compared are the estimated probability using a Gaussian naive Bayes classifier
without calibration, with a sigmoid calibration, and with a non-parametric
isotonic calibration. One can observe that only the non-parametric model is
able to provide a probability calibration that returns probabilities close
to the expected 0.5 for most of the samples belonging to the middle
cluster with heterogeneous labels. This results in a significantly improved
Brier score.

"""
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)

y[: n_samples // 2] = 0
y[n_samples // 2 :] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.9, random_state=42
)

# %%
# Gaussian Naive-Bayes
# --------------------
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB

# With no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# With isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# With sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method="sigmoid")
clf_sigmoid.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier score losses: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sample_weight=sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sample_weight=sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sample_weight=sw_test)
```
### 25 - sklearn/calibration.py:

Start line: 1295, End line: 1313

```python
class CalibrationDisplay:

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        n_bins=5,
        strategy="uniform",
        pos_label=None,
        name=None,
        ref_line=True,
        ax=None,
        **kwargs,
    ):
        # ... other code

        if not is_classifier(estimator):
            raise ValueError("'estimator' should be a fitted classifier.")

        y_prob, pos_label = _get_response(
            X, estimator, response_method="predict_proba", pos_label=pos_label
        )

        name = name if name is not None else estimator.__class__.__name__
        return cls.from_predictions(
            y,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=name,
            ref_line=ref_line,
            ax=ax,
            **kwargs,
        )
```
### 48 - sklearn/calibration.py:

Start line: 514, End line: 587

```python
def _fit_classifier_calibrator_pair(
    estimator,
    X,
    y,
    train,
    test,
    supports_sw,
    method,
    classes,
    sample_weight=None,
    **fit_params,
):
    """Fit a classifier/calibration pair on a given train/test split.

    Fit the classifier on the train set, compute its predictions on the test
    set and use the predictions as input to fit the calibrator along with the
    test labels.

    Parameters
    ----------
    estimator : estimator instance
        Cloned base estimator.

    X : array-like, shape (n_samples, n_features)
        Sample data.

    y : array-like, shape (n_samples,)
        Targets.

    train : ndarray, shape (n_train_indices,)
        Indices of the training subset.

    test : ndarray, shape (n_test_indices,)
        Indices of the testing subset.

    supports_sw : bool
        Whether or not the `estimator` supports sample weights.

    method : {'sigmoid', 'isotonic'}
        Method to use for calibration.

    classes : ndarray, shape (n_classes,)
        The target classes.

    sample_weight : array-like, default=None
        Sample weights for `X`.

    **fit_params : dict
        Parameters to pass to the `fit` method of the underlying
        classifier.

    Returns
    -------
    calibrated_classifier : _CalibratedClassifier instance
    """
    fit_params_train = _check_fit_params(X, fit_params, train)
    X_train, y_train = _safe_indexing(X, train), _safe_indexing(y, train)
    X_test, y_test = _safe_indexing(X, test), _safe_indexing(y, test)

    if sample_weight is not None and supports_sw:
        sw_train = _safe_indexing(sample_weight, train)
        estimator.fit(X_train, y_train, sample_weight=sw_train, **fit_params_train)
    else:
        estimator.fit(X_train, y_train, **fit_params_train)

    n_classes = len(classes)
    pred_method, method_name = _get_prediction_method(estimator)
    predictions = _compute_predictions(pred_method, method_name, X_test, n_classes)

    sw_test = None if sample_weight is None else _safe_indexing(sample_weight, test)
    calibrated_classifier = _fit_calibrator(
        estimator, predictions, y_test, classes, method, sample_weight=sw_test
    )
    return calibrated_classifier
```
