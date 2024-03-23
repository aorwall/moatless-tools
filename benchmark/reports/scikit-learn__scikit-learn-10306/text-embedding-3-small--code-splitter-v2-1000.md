0 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import FitFailedWarning
    >>> import warnings
    >>> warnings.simplefilter('always', FitFailedWarning)
    >>> gs = GridSearchCV(LinearSVC(), {'C': [-1, -2]}, error_score=0, cv=2)
    >>> X, y = [[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1]
    >>> with warnings.catch_warnings(record=True) as w:
    ...     try:
    ...         gs.fit(X, y)   # This will raise a ValueError since C is < 0
    ...     except ValueError:
    ...         pass
    ...     print(repr(w[-1].message))
    ... # doctest: +NORMALIZE_WHITESPACE
    FitFailedWarning('Estimator fit failed. The score on this train-test
    partition for these parameters will be set to 0.000000.
    Details: \\nValueError: Penalty term must be positive; got (C=-2)\\n',)

    .. versionchanged:: 0.18
       Moved from sklearn.cross_validation.
    """


class NonBLASDotWarning(EfficiencyWarning):
    """Warning used when the dot operation does not use BLAS.

    This warning is used to notify the user that BLAS was not used for dot
    operation and hence the efficiency may be affected.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation, extends EfficiencyWarning.
    """


class SkipTestWarning(UserWarning):
    """Warning class used to notify the user of a test that was skipped.

    For example, one of the estimator checks requires a pandas import.
    If the pandas package cannot be imported, the test will be skipped rather
    than register as a failure.
    """


class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """

```
1 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    if name == 'AffinityPropagation':
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert_equal(pred.shape, (n_samples,))
    assert_greater(adjusted_rand_score(pred, y), 0.4)
    # fit another time with ``fit_predict`` and compare results
    if name == 'SpectralClustering':
        # there is no way to make Spectral clustering deterministic :(
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert_in(pred.dtype, [np.dtype('int32'), np.dtype('int64')])
    assert_in(pred2.dtype, [np.dtype('int32'), np.dtype('int64')])

    # Add noise to X to test the possible values of the labels
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(labels_sorted[0],
                                                labels_sorted[-1] + 1))

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert_true(labels_sorted[0] in [0, -1])
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert_greater_equal(n_clusters - 1, labels_sorted[-1])
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=DeprecationWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels"""
    X, y = make_blobs(n_samples=20, random_state=0)
    clusterer = clone(clusterer_orig)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        if hasattr(clusterer, "random_state"):
            clusterer.set_params(random_state=0)

        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=DeprecationWarning)
def check_classifiers_one_label(name, classifier_orig):
    error_string_fit = "Classifier can't train when only one class is present."
    error_string_predict = ("Classifier can't predict when only one class is "
                            "present.")
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 3))
    X_test = rnd.uniform(size=(10, 3))
    y = np.ones(10)
    # catch deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        classifier = clone(classifier_orig)
        # try to fit
        try:
            classifier.fit(X_train, y)
        except ValueError as e:
            if 'class' not in repr(e):
                print(error_string_fit, classifier, e)
                traceback.print_exc(file=sys.stdout)
                raise e
            else:
                return
        except Exception as exc:
            print(error_string_fit, classifier, exc)
            traceback.print_exc(file=sys.stdout)
            raise exc
        # predict
        try:
            assert_array_equal(classifier.predict(X_test), y)
        except Exception as exc:
            print(error_string_predict, classifier, exc)
            raise exc


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    
```
2 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
"""
The :mod:`sklearn.exceptions` module includes all custom warnings and error
classes used across scikit-learn.
"""

__all__ = ['NotFittedError',
           'ChangedBehaviorWarning',
           'ConvergenceWarning',
           'DataConversionWarning',
           'DataDimensionalityWarning',
           'EfficiencyWarning',
           'FitFailedWarning',
           'NonBLASDotWarning',
           'SkipTestWarning',
           'UndefinedMetricWarning']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet',)

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class ChangedBehaviorWarning(UserWarning):
    """Warning class used to notify the user of any change in the behavior.

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class DataDimensionalityWarning(UserWarning):
    """Custom warning to notify potential issues with data dimensionality.

    For example, in random projection, this warning is raised when the
    number of components, which quantifies the dimensionality of the target
    projection space, is higher than the number of features, which quantifies
    the dimensionality of the original source space, to imply that the
    dimensionality of the problem will not be reduced.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """



```
3 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`

    Returns
    -------

    result : the return value of `func`

    """
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError("%s did not give warning: %s( is %s)"
                                 % (func.__name__, warning_class, w))
    return result


def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str | callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    """
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Let's not catch the numpy internal DeprecationWarnings
            warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError("No warning raised for %s with class "
                                 "%s"
                                 % (func.__name__, warning_class))

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, 'args') else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:
                check_in_message = lambda msg: message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError("Did not receive the message you expected "
                                 "('%s') for <%s>, got: '%s'"
                                 % (message, func.__name__, msg))

    return result


def assert_warns_div0(func, *args, **kw):
    """Assume that numpy's warning for divide by zero is raised

    Handles the case of platforms that do not support warning on divide by zero
    """

    with np.errstate(divide='warn', invalid='warn'):
        try:
            assert_warns(RuntimeWarning, np.divide, 1, np.zeros(1))
        except AssertionError:
            # This platform does not report numpy divide by zeros
            return func(*args, **kw)
        return assert_warns_message(RuntimeWarning,
                                    'invalid value encountered',
                                    func, *args, **kw)


# To remove when we support numpy 1.7

```
4 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import struct

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, TransformerMixin, ClusterMixin,
                          BaseEstimator, is_classifier, is_regressor,
                          is_outlier_detector)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.fixes import signature
from sklearn.utils.validation import has_fit_parameter, _num_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GaussianProcess',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']


def _yield_non_meta_checks(name, estimator):
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_dtype_object
    yield check_sample_weights_pandas_series
    yield check_sample_weights_list
    yield check_estimators_fit_returns_self
    yield check_complex_data

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION + ['SpectralEmbedding']:
        # SpectralEmbedding is non-deterministic,
        # see issue #4236
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if name not in ['SimpleImputer', 'Imputer']:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    if name not in ['GaussianProcess']:
        # FIXME!
        # in particular GaussianProcess!
        yield check_estimators_overwrite_params
    if hasattr(estimator, 'sparsify'):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle



```
5 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _yield_classifier_checks(name, classifier):
    # test classifiers can handle non-array data
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    # basic consistency testing
    yield check_classifiers_train
    yield check_classifiers_regression_target
    if (name not in ["MultinomialNB", "ComplementNB", "LabelPropagation",
                     "LabelSpreading"] and
        # TODO some complication with -1 label
            name not in ["DecisionTreeClassifier", "ExtraTreeClassifier"]):
        # We don't raise a warning in these classifiers, as
        # the column y interface is used by the forests.

        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    # test if NotFittedError is raised
    yield check_estimators_unfitted
    if 'class_weight' in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    estimator = clone(estimator_orig)
    rng = np.random.RandomState(888)
    X = rng.randn(10, 5)
    y = np.ones(10) * np.inf
    y = multioutput_estimator_convert_y_2d(estimator, y)

    errmsg = "Input contains NaN, infinity or a value too large for " \
             "dtype('float64')."
    try:
        estimator.fit(X, y)
    except ValueError as e:
        if str(e) != errmsg:
            raise ValueError("Estimator {0} raised error as expected, but "
                             "does not match expected error message"
                             .format(name))
    else:
        raise ValueError("Estimator {0} should have raised error on fitting "
                         "array y with NaN value.".format(name))


def _yield_regressor_checks(name, regressor):
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    yield check_regressors_no_decision_function
    yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    if name != 'CCA':
        # check that the regressor handles int input
        yield check_regressors_int
    if name != "GaussianProcessRegressor":
        # Test if NotFittedError is raised
        yield check_estimators_unfitted
    yield check_non_transformer_estimators_n_iter


def _yield_transformer_checks(name, transformer):
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if name not in ['AdditiveChi2Sampler', 'Binarizer', 'Normalizer',
                    'PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    if name not in ['AdditiveChi2Sampler', 'Binarizer',
                    'FunctionTransformer', 'Normalizer']:
        # basic tests
        yield check_transformer_general
        yield check_transformers_unfitted
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = ['Isomap', 'KernelPCA', 'LocallyLinearEmbedding',
                       'RandomizedLasso', 'LogisticRegressionCV']
    if name not in external_solver:
        yield check_transformer_n_iter


def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ('WardAgglomeration', "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter



```
6 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_balanced_classifiers(name, classifier_orig, X_train,
                                            y_train, X_test, y_test, weights):
    classifier = clone(classifier_orig)
    if hasattr(classifier, "n_iter"):
        classifier.set_params(n_iter=100)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)

    set_random_state(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    classifier.set_params(class_weight='balanced')
    classifier.fit(X_train, y_train)
    y_pred_balanced = classifier.predict(X_test)
    assert_greater(f1_score(y_test, y_pred_balanced, average='weighted'),
                   f1_score(y_test, y_pred, average='weighted'))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_balanced_linear_classifier(name, Classifier):
    """Test class weights with non-contiguous class labels."""
    # this is run on classes, not instances, though this should be changed
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-.8, -1.0],
                  [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    classifier = Classifier()

    if hasattr(classifier, "n_iter"):
        # This is a very small dataset, default n_iter are likely to prevent
        # convergence
        classifier.set_params(n_iter=1000)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)
    set_random_state(classifier)

    # Let the model compute the class frequencies
    classifier.set_params(class_weight='balanced')
    coef_balanced = classifier.fit(X, y).coef_.copy()

    # Count each label occurrence to reweight manually
    n_samples = len(y)
    n_classes = float(len(np.unique(y)))

    class_weight = {1: n_samples / (np.sum(y == 1) * n_classes),
                    -1: n_samples / (np.sum(y == -1) * n_classes)}
    classifier.set_params(class_weight=class_weight)
    coef_manual = classifier.fit(X, y).coef_.copy()

    assert_allclose(coef_balanced, coef_manual)



```
7 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    estimator = clone(estimator_orig)
    if has_fit_parameter(estimator, "sample_weight"):
        try:
            import pandas as pd
            X = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
            X = pd.DataFrame(pairwise_estimator_convert_X(X, estimator_orig))
            y = pd.Series([1, 1, 1, 2, 2, 2])
            weights = pd.Series([1] * 6)
            try:
                estimator.fit(X, y, sample_weight=weights)
            except ValueError:
                raise ValueError("Estimator {0} raises error if "
                                 "'sample_weight' parameter is of "
                                 "type pandas.Series".format(name))
        except ImportError:
            raise SkipTest("pandas is not installed: not testing for "
                           "input of type pandas.Series to class weight.")


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    if has_fit_parameter(estimator_orig, "sample_weight"):
        estimator = clone(estimator_orig)
        rnd = np.random.RandomState(0)
        X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                         estimator_orig)
        y = np.arange(10) % 3
        y = multioutput_estimator_convert_y_2d(estimator, y)
        sample_weight = [3] * 10
        # Test that estimators don't raise any exception
        estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    y = (X[:, 0] * 4).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    try:
        estimator.fit(X, y.astype(object))
    except Exception as e:
        if "Unknown label type" not in str(e):
            raise

    X[0, 0] = {'foo': 'bar'}
    msg = "argument must be a string or a number"
    assert_raises_regex(TypeError, msg, estimator.fit, X, y)


def check_complex_data(name, estimator_orig):
    # check that estimators raise an exception on providing complex data
    X = np.random.sample(10) + 1j * np.random.sample(10)
    X = X.reshape(-1, 1)
    y = np.random.sample(10) + 1j * np.random.sample(10)
    estimator = clone(estimator_orig)
    assert_raises_regex(ValueError, "Complex data not supported",
                        estimator.fit, X, y)
```
8 - /tmp/repos/scikit-learn/sklearn/utils/_joblib.py
```python
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib imports may raise DeprecationWarning on certain Python
    # versions
    import joblib
    from joblib import logger
    from joblib import dump, load
    from joblib import __version__
    from joblib import effective_n_jobs
    from joblib import hash
    from joblib import cpu_count, Parallel, Memory, delayed
    from joblib import parallel_backend, register_parallel_backend


__all__ = ["parallel_backend", "register_parallel_backend", "cpu_count",
           "Parallel", "Memory", "delayed", "effective_n_jobs", "hash",
           "logger", "dump", "load", "joblib", "__version__"]
```
**9 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class _CFSubcluster(object):
    """Each subcluster in a CFNode is called a CFSubcluster.

    A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray, shape (n_features,), optional
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """
    def __init__(self, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,
             self.centroid_, self.sq_norm_) = \
                new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False

    @property
    def radius(self):
        """Return radius of the subcluster"""
        dot_product = -2 * np.dot(self.linear_sum_, self.centroid_)
        return sqrt(
            ((self.squared_sum_ + dot_product) / self.n_samples_) +
            self.sq_norm_)


class Birch(BaseEstimator, TransformerMixin, ClusterMixin):
    
```
10 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)

    X_zero_samples = np.empty(0).reshape(0, 3)
    # The precise message can change depending on whether X or y is
    # validated first. Let us test the type of exception only:
    with assert_raises(ValueError, msg="The estimator {} does not"
                       " raise an error when an empty data is used "
                       "to train. Perhaps use "
                       "check_array in train.".format(name)):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape(3, 0)
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = multioutput_estimator_convert_y_2d(e, np.array([1, 0, 1]))
    msg = (r"0 feature\(s\) \(shape=\(3, 0\)\) while a minimum of \d* "
           "is required.")
    assert_raises_regex(ValueError, msg, e.fit, X_zero_features, y)


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
**11 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py**:
```python
while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if (self.is_data_valid is not None
                    and not self.is_data_valid(X_subset, y_subset)):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator.fit(X_subset, y_subset)
            else:
                base_estimator.fit(X_subset, y_subset,
                                   sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)

            # XXX: Deprecation: Remove this if block in 0.20
            if self.residual_metric is not None:
                diff = y_pred - y
                if diff.ndim == 1:
                    diff = diff.reshape(-1, 1)
                residuals_subset = self.residual_metric(diff)
            else:
                residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = base_estimator.score(X_inlier_subset,
                                                y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            if ((self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips):
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*).")
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*).")
        else:
            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                warnings.warn("RANSAC found a valid consensus set but exited"
                              " early due to skipping more iterations than"
                              " `max_skips`. See estimator attributes for"
                              " diagnostics (n_skips*).",
                              ConvergenceWarning)

        # estimate final model using all inliers
        base_estimator.fit(X_inlier_best, y_inlier_best)

        self.estimator_ = base_estimator
        self.inlier_mask_ = inlier_mask_best
        return self

    
```
**12 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
```python
"""
Python implementation of the fast ICA algorithms.

Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.
"""

# Authors: Pierre Lafaye de Micheaux, Stefan van der Walt, Gael Varoquaux,
#          Bertrand Thirion, Alexandre Gramfort, Denis A. Engemann
# License: BSD 3 clause

import warnings

import numpy as np
from scipy import linalg

from ..base import BaseEstimator, TransformerMixin
from ..exceptions import ConvergenceWarning
from ..externals import six
from ..externals.six import moves
from ..externals.six import string_types
from ..utils import check_array, as_float_array, check_random_state
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES

__all__ = ['fastica', 'FastICA']


def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W

    Parameters
    ----------
    w : ndarray of shape(n)
        Array to be orthogonalized

    W : ndarray of shape(p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.dot(np.dot(w, W[:j].T), W[:j])
    return w


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())

        for i in moves.xrange(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)

            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            _gs_decorrelation(w1, W, j)

            w1 /= np.sqrt((w1 ** 2).sum())

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in moves.xrange(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_
                                - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge. Consider increasing '
                      'tolerance or the maximum number of iterations.',
                      ConvergenceWarning)

    return W, ii + 1


# Some standard non-linear functions.
# XXX: these should be optimized, as they can be a bottleneck.

```
13 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_fit_returns_self(name, estimator_orig):
    """Check if self is returned when calling fit"""
    X, y = make_blobs(random_state=0, n_samples=9, n_features=4)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    assert_true(estimator.fit(X, y) is estimator)


@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise either AttributeError or ValueError.
    The specific exception type NotFittedError inherits from both and can
    therefore be adequately raised for that purpose.
    """

    # Common test for Regressors, Classifiers and Outlier detection estimators
    X, y = _boston_subset()

    est = clone(estimator_orig)

    msg = "fit"
    if hasattr(est, 'predict'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict, X)

    if hasattr(est, 'decision_function'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.decision_function, X)

    if hasattr(est, 'predict_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_proba, X)

    if hasattr(est, 'predict_log_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_log_proba, X)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_2d(name, estimator_orig):
    if "MultiTask" in name:
        # These only work on 2d, so this test makes no sense
        return
    rnd = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)), estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % (
        ", ".join([str(w_x) for w_x in w]))
    if name not in MULTI_OUTPUT:
        # check that we warned if we don't support multi-output
        assert_greater(len(w), 0, msg)
        assert_true("DataConversionWarning('A column-vector y"
                    " was passed when a 1d array was expected" in msg)
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())



```
14 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_train(name, regressor_orig):
    X, y = _boston_subset()
    X = pairwise_estimator_convert_X(X, regressor_orig)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))  # X is already scaled
    y = y.ravel()
    regressor = clone(regressor_orig)
    y = multioutput_estimator_convert_y_2d(regressor, y)
    rnd = np.random.RandomState(0)
    if not hasattr(regressor, 'alphas') and hasattr(regressor, 'alpha'):
        # linear regressors need to set alpha, but not generalized CV ones
        regressor.alpha = 0.01
    if name == 'PassiveAggressiveRegressor':
        regressor.C = 0.01

    # raises error on malformed input for fit
    with assert_raises(ValueError, msg="The classifier {} does not"
                       " raise an error when incorrect/malformed input "
                       "data for fit is passed. The number of training "
                       "examples is not the same as the number of "
                       "labels. Perhaps use check_X_y in fit.".format(name)):
        regressor.fit(X, y[:-1])
    # fit
    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y
    set_random_state(regressor)
    regressor.fit(X, y_)
    regressor.fit(X.tolist(), y_.tolist())
    y_pred = regressor.predict(X)
    assert_equal(y_pred.shape, y_.shape)

    # TODO: find out why PLS and CCA fail. RANSAC is random
    # and furthermore assumes the presence of outliers, hence
    # skipped
    if name not in ('PLSCanonical', 'CCA', 'RANSACRegressor'):
        assert_greater(regressor.score(X, y_), 0.5)


@ignore_warnings
def check_regressors_no_decision_function(name, regressor_orig):
    # checks whether regressors have decision_function or predict_proba
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 4))
    regressor = clone(regressor_orig)
    y = multioutput_estimator_convert_y_2d(regressor, X[:, 0])

    if hasattr(regressor, "n_components"):
        # FIXME CCA, PLS is not robust to rank 1 effects
        regressor.n_components = 1

    regressor.fit(X, y)
    funcs = ["decision_function", "predict_proba", "predict_log_proba"]
    for func_name in funcs:
        func = getattr(regressor, func_name, None)
        if func is None:
            # doesn't have function
            continue
        # has function. Should raise deprecation warning
        msg = func_name
        assert_warns_message(DeprecationWarning, msg, func, X)



```
15 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _apply_func(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features))
                       for batch in X]
    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    return np.ravel(result_full), np.ravel(result_by_batch)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini bathes or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "score_samples", "predict_proba"]:

        msg = ("{method} of {name} is not invariant when applied "
               "to a subset.").format(method=method, name=name)
        # TODO remove cases when corrected
        if (name, method) in [('SVC', 'decision_function'),
                              ('SparsePCA', 'transform'),
                              ('MiniBatchSparsePCA', 'transform'),
                              ('BernoulliRBM', 'score_samples')]:
            raise SkipTest(msg)

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_func(
                getattr(estimator, method), X)
            assert_allclose(result_full, result_by_batch,
                            atol=1e-7, err_msg=msg)


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    msgs = ["1 sample", "n_samples = 1", "n_samples=1", "one sample",
            "1 class", "one class"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == 'RandomizedLogisticRegression':
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == 'RANSACRegressor':
        estimator.residual_threshold = 0.5

    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator, 1)

    msgs = ["1 feature(s)", "n_features = 1", "n_features=1"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e



```
16 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    assert_raises(ValueError, estimator.fit, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_general(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    _check_transformer(name, transformer, X, y)
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - .1
    this_X = NotAnArray(X)
    this_y = NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformers_unfitted(name, transformer):
    X, y = _boston_subset()

    transformer = clone(transformer)
    with assert_raises((AttributeError, ValueError), msg="The unfitted "
                       "transformer {} does not raise an error when "
                       "transform is called. Perhaps use "
                       "check_is_fitted in transform.".format(name)):
        transformer.transform(X)



```
17 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = ['Ridge', 'SVR', 'NuSVR', 'NuSVC',
                            'RidgeClassifier', 'SVC', 'RandomizedLasso',
                            'LogisticRegressionCV', 'LinearSVC',
                            'LogisticRegression']

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = enforce_estimator_tags_y(estimator, y_)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
            estimator.fit(X, y_)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                               random_state=0, n_features=2, cluster_std=0.1)
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert iter_ >= 1
        else:
            assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert all(item in deep_params.items() for item in
               shallow_params.items())



```
18 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        count = center_mask.sum()

        if count > 0:
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            centers[center_idx] *= counts[center_idx]

            # inplace sum with new points members of this cluster
            centers[center_idx] += np.sum(X[center_mask], axis=0)

            # update the count statistics for this center
            counts[center_idx] += count

            # inplace rescale to compute mean of all points (old and new)
            # Note: numpy >= 1.10 does not support '/=' for the following
            # expression for a mixture of int and float (see numpy issue #6464)
            centers[center_idx] = centers[center_idx] / counts[center_idx]

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return inertia, squared_diff


def _mini_batch_convergence(model, iteration_idx, n_iter, tol,
                            n_samples, centers_squared_diff, batch_inertia,
                            context, verbose=0):
    """Helper function to encapsulate the early stopping logic"""
    # Normalize inertia to be able to compare values when
    # batch_size changes
    batch_inertia /= model.batch_size
    centers_squared_diff /= model.batch_size

    # Compute an Exponentially Weighted Average of the squared
    # diff to monitor the convergence while discarding
    # minibatch-local stochastic variability:
    # https://en.wikipedia.org/wiki/Moving_average
    ewa_diff = context.get('ewa_diff')
    ewa_inertia = context.get('ewa_inertia')
    if ewa_diff is None:
        ewa_diff = centers_squared_diff
        ewa_inertia = batch_inertia
    else:
        alpha = float(model.batch_size) * 2.0 / (n_samples + 1)
        alpha = 1.0 if alpha > 1.0 else alpha
        ewa_diff = ewa_diff * (1 - alpha) + centers_squared_diff * alpha
        ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

    # Log progress to be able to monitor convergence
    if verbose:
        progress_msg = (
            'Minibatch iteration %d/%d:'
            ' mean batch inertia: %f, ewa inertia: %f ' % (
                iteration_idx + 1, n_iter, batch_inertia,
                ewa_inertia))
        print(progress_msg)

    # Early stopping based on absolute tolerance on squared change of
    # centers position (using EWA smoothing)
    if tol > 0.0 and ewa_diff <= tol:
        if verbose:
            print('Converged (small centers change) at iteration %d/%d'
                  % (iteration_idx + 1, n_iter))
        return True

    # Early stopping heuristic due to lack of improvement on smoothed inertia
    ewa_inertia_min = context.get('ewa_inertia_min')
    no_improvement = context.get('no_improvement', 0)
    if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
        no_improvement = 0
        ewa_inertia_min = ewa_inertia
    else:
        no_improvement += 1

    if (model.max_no_improvement is not None
            and no_improvement >= model.max_no_improvement):
        if verbose:
            print('Converged (lack of improvement in inertia)'
                  ' at iteration %d/%d'
                  % (iteration_idx + 1, n_iter))
        return True

    # update the convergence context to maintain state across successive calls:
    context['ewa_diff'] = ewa_diff
    context['ewa_inertia'] = ewa_inertia
    context['ewa_inertia_min'] = ewa_inertia_min
    context['no_improvement'] = no_improvement
    return False


class MiniBatchKMeans(KMeans):
    
```
19 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.array(array, dtype=dtype, order=order, copy=copy)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happend, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of strings will be "
                "interpreted as decimal numbers if parameter 'dtype' is "
                "'numeric'. It is recommended that you convert the array to "
                "type np.float64 before passing it to check_array.",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    
```
20 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
    assert set(orig_params.keys()) == set(curr_params.keys()), msg
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
                    assert (set(params_before_exception.keys()) ==
                                 set(curr_params.keys()))
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                curr_params = estimator.get_params(deep=False)
                assert (set(test_params.keys()) ==
                        set(curr_params.keys())), msg
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg
        test_params[param_name] = default_value


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    boston = load_boston()
    X, y = boston.data, boston.target
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    if not _safe_tags(e, "no_validation"):
        assert_raises_regex(ValueError, msg, e.fit, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(n_samples=100, random_state=0, n_features=4,
                      centers=centers, cluster_std=1.0, shuffle=True)
    X_test = np.random.randn(20, 2) + 4
    estimator = clone(estimator_orig)

    if (hasattr(estimator, "decision_function") and
            hasattr(estimator, "predict_proba")):

        estimator.fit(X, y)
        # Since the link function from decision_function() to predict_proba()
        # is sometimes not precise enough (typically expit), we round to the
        # 10th decimal to avoid numerical issues.
        a = estimator.predict_proba(X_test)[:, 1].round(decimals=10)
        b = estimator.decision_function(X_test).round(decimals=10)
        assert_array_equal(rankdata(a), rankdata(b))



```
21 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert_true(args[1] in ["y", "Y"],
                        "Expected y or Y as second argument for method "
                        "%s of %s. Got arguments: %r."
                        % (func_name, type(estimator).__name__, args))


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
22 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import gc
import time

from sklearn.externals.joblib import Memory
from sklearn.linear_model import (LogisticRegression, SGDClassifier)
from sklearn.datasets import fetch_rcv1
from sklearn.linear_model.sag import get_auto_step_size

try:
    import lightning.classification as lightning_clf
except ImportError:
    lightning_clf = None

m = Memory(cachedir='.', verbose=0)


# compute logistic loss
def get_loss(w, intercept, myX, myy, C):
    n_samples = myX.shape[0]
    w = w.ravel()
    p = np.mean(np.log(1. + np.exp(-myy * (myX.dot(w) + intercept))))
    print("%f + %f" % (p, w.dot(w) / 2. / C / n_samples))
    p += w.dot(w) / 2. / C / n_samples
    return p


# We use joblib to cache individual fits. Note that we do not pass the dataset
# as argument as the hashing would be too slow, so we assume that the dataset
# never changes.
@m.cache()
def bench_one(name, clf_type, clf_params, n_iter):
    clf = clf_type(**clf_params)
    try:
        clf.set_params(max_iter=n_iter, random_state=42)
    except:
        clf.set_params(n_iter=n_iter, random_state=42)

    st = time.time()
    clf.fit(X, y)
    end = time.time()

    try:
        C = 1.0 / clf.alpha / n_samples
    except:
        C = clf.C

    try:
        intercept = clf.intercept_
    except:
        intercept = 0.

    train_loss = get_loss(clf.coef_, intercept, X, y, C)
    train_score = clf.score(X, y)
    test_score = clf.score(X_test, y_test)
    duration = end - st

    return train_loss, train_score, test_score, duration


def bench(clfs):
    for (name, clf, iter_range, train_losses, train_scores,
         test_scores, durations) in clfs:
        print("training %s" % name)
        clf_type = type(clf)
        clf_params = clf.get_params()

        for n_iter in iter_range:
            gc.collect()

            train_loss, train_score, test_score, duration = bench_one(
                name, clf_type, clf_params, n_iter)

            train_losses.append(train_loss)
            train_scores.append(train_score)
            test_scores.append(test_score)
            durations.append(duration)
            print("classifier: %s" % name)
            print("train_loss: %.8f" % train_loss)
            print("train_score: %.8f" % train_score)
            print("test_score: %.8f" % test_score)
            print("time for fit: %.8f seconds" % duration)
            print("")

        print("")
    return clfs


def plot_train_losses(clfs):
    plt.figure()
    for (name, _, _, train_losses, _, _, durations) in clfs:
        plt.plot(durations, train_losses, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train loss")


def plot_train_scores(clfs):
    plt.figure()
    for (name, _, _, _, train_scores, _, durations) in clfs:
        plt.plot(durations, train_scores, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train score")
        plt.ylim((0.92, 0.96))


def plot_test_scores(clfs):
    plt.figure()
    for (name, _, _, _, _, test_scores, durations) in clfs:
        plt.plot(durations, test_scores, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("test score")
        plt.ylim((0.92, 0.96))



```
23 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(n_samples=100, random_state=0, n_features=4,
                      centers=centers, cluster_std=1.0, shuffle=True)
    X_test = np.random.randn(20, 2) + 4
    estimator = clone(estimator_orig)

    if (hasattr(estimator, "decision_function") and
            hasattr(estimator, "predict_proba")):

        estimator.fit(X, y)
        a = estimator.predict_proba(X_test)[:, 1]
        b = estimator.decision_function(X_test)
        assert_array_equal(rankdata(a), rankdata(b))


def check_outliers_fit_predict(name, estimator_orig):
    # Check fit_predict for outlier detectors.

    X, _ = make_blobs(n_samples=300, random_state=0)
    X = shuffle(X, random_state=7)
    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)

    set_random_state(estimator)

    y_pred = estimator.fit_predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    # check fit_predict = fit.predict when possible
    if hasattr(estimator, 'predict'):
        y_pred_2 = estimator.fit(X).predict(X)
        assert_array_equal(y_pred, y_pred_2)

    if hasattr(estimator, "contamination"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'
        contamination = 0.1
        estimator.set_params(contamination=contamination)
        y_pred = estimator.fit_predict(X)
        assert_almost_equal(np.mean(y_pred != 1), contamination)

        # raises error when contamination is a scalar and not in [0,1]
        for contamination in [-0.5, 2.3]:
            estimator.set_params(contamination=contamination)
            assert_raises(ValueError, estimator.fit_predict, X)

```
24 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_no_attributes_set_in_init(name, estimator):
    """Check setting during init. """

    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    init_params = _get_args(type(estimator).__init__)
    parents_init_params = [param for params_parent in
                           (_get_args(parent) for parent in
                            type(estimator).__mro__)
                           for param in params_parent]

    # Test for no setting apart from parameters during init
    invalid_attr = (set(vars(estimator)) - set(init_params)
                    - set(parents_init_params))
    assert_false(invalid_attr,
                 "Estimator %s should not set any attribute apart"
                 " from parameters during init. Found attributes %s."
                 % (name, sorted(invalid_attr)))
    # Ensure that each parameter is set in init
    invalid_attr = (set(init_params) - set(vars(estimator))
                    - set(["self"]))
    assert_false(invalid_attr,
                 "Estimator %s should store all parameters"
                 " as an attribute during init. Did not find "
                 "attributes %s." % (name, sorted(invalid_attr)))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sparsify_coefficients(name, estimator_orig):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
                  [-1, -2], [2, 2], [-2, -2]])
    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    est = clone(estimator_orig)

    est.fit(X, y)
    pred_orig = est.predict(X)

    # test sparsify with dense inputs
    est.sparsify()
    assert_true(sparse.issparse(est.coef_))
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)

    # pickle and unpickle with sparse coef_
    est = pickle.loads(pickle.dumps(est))
    assert_true(sparse.issparse(est.coef_))
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)


@ignore_warnings(category=DeprecationWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array([[3, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 1]])
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = [1, 1, 1, 2, 2, 2]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)



```
25 - /tmp/repos/scikit-learn/examples/cluster/plot_kmeans_assumptions.py
```python
"""
====================================
Demonstration of k-means assumptions
====================================

This example is meant to illustrate situations where k-means will produce
unintuitive and possibly unexpected clusters. In the first three plots, the
input data does not conform to some implicit assumption that k-means makes and
undesirable clusters are produced as a result. In the last plot, k-means
returns intuitive clusters despite unevenly sized blobs.
"""
print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
```
26 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _boston_subset(n_samples=50)
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_data_not_an_array(name, estimator_orig, X, y):
    if name in CROSS_DECOMPOSITION:
        raise SkipTest("Skipping check_estimators_data_not_an_array "
                       "for cross decomposition module as estimators "
                       "are not deterministic.")
    # separate estimators to control random seeds
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    y_ = NotAnArray(np.asarray(y))
    X_ = NotAnArray(np.asarray(X))

    # fit
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


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
                assert_equal(param_value, init_param.default, init_param.name)


def multioutput_estimator_convert_y_2d(estimator, y):
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if "MultiTask" in estimator.__class__.__name__:
        return np.reshape(y, (-1, 1))
    return y
```
27 - /tmp/repos/scikit-learn/examples/cluster/plot_linkage_comparison.py
```python
"""
================================================================
Comparing different hierarchical linkage methods on toy datasets
================================================================

This example shows characteristics of different linkage
methods for hierarchical clustering on datasets that are
"interesting" but still in 2D.

The main observations to make are:

- single linkage is fast, and can perform well on
  non-globular data, but it performs poorly in the
  presence of noise.
- average and complete linkage perform well on
  cleanly separated globular clusters, but have mixed
  results otherwise.
- Ward is the most effective method for noisy data.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

######################################################################
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

######################################################################
# Run the clustering and plot

# Set up cluster parameters
plt.figure(figsize=(9 * 1.3 + 2, 14.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
    (noisy_circles, {'n_clusters': 2}),
    (noisy_moons, {'n_clusters': 2}),
    (varied, {'n_neighbors': 2}),
    (aniso, {'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]


```
28 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.ascontiguousarray(covariance_[1:, 1:])
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == 'cd':
                        # Use coordinate descent
                        coefs = -(precision_[indices != idx, idx]
                                  / (precision_[idx, idx] + 1000 * eps))
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs, alpha, 0, sub_covariance,
                            row, row, max_iter, enet_tol,
                            check_random_state(None), False)
                    else:
                        # Use LARS
                        _, _, coefs = lars_path(
                            sub_covariance, row, Xy=row, Gram=sub_covariance,
                            alpha_min=alpha / (n_features - 1), copy_Gram=True,
                            eps=eps, method='lars', return_path=False)
                # Update the precision matrix
                precision_[idx, idx] = (
                    1. / (covariance_[idx, idx]
                          - np.dot(covariance_[indices != idx, idx], coefs)))
                precision_[indices != idx, idx] = (- precision_[idx, idx]
                                                   * coefs)
                precision_[idx, indices != idx] = (- precision_[idx, idx]
                                                   * coefs)
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)
            if verbose:
                print('[graphical_lasso] Iteration '
                      '% 3i, cost % 3.2e, dual gap %.3e'
                      % (i, cost, d_gap))
            if return_costs:
                costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError('Non SPD result: the system is '
                                         'too ill-conditioned for this solver')
        else:
            warnings.warn('graphical_lasso: did not converge after '
                          '%i iteration: dual gap: %.3e'
                          % (max_iter, d_gap), ConvergenceWarning)
    except FloatingPointError as e:
        e.args = (e.args[0]
                  + '. The system is too ill-conditioned for this solver',)
        raise e

    if return_costs:
        if return_n_iter:
            return covariance_, precision_, costs, i + 1
        else:
            return covariance_, precision_, costs
    else:
        if return_n_iter:
            return covariance_, precision_, i + 1
        else:
            return covariance_, precision_



```
29 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
        n_samples, n_features = X.shape
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


def choose_check_classifiers_labels(name, y, y_names):
    return y if name in ["LabelPropagation", "LabelSpreading"] else y_names

def check_classifiers_classes(name, classifier_orig):
    X_multiclass, y_multiclass = make_blobs(n_samples=30, random_state=0,
                                            cluster_std=0.1)
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass,
                                         random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X_multiclass -= X_multiclass.min() - .1

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = pairwise_estimator_convert_X(X_multiclass, classifier_orig)
    X_binary = pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    for X, y, y_names in [(X_multiclass, y_multiclass, y_names_multiclass),
                          (X_binary, y_binary, y_names_binary)]:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)
```
30 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def check_outliers_train(name, estimator_orig):
    X, _ = make_blobs(n_samples=300, random_state=0)
    X = shuffle(X, random_state=7)
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
**31 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
# Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause
from __future__ import division

import warnings
import numpy as np
from scipy import sparse
from math import sqrt

from ..metrics.pairwise import euclidean_distances
from ..base import TransformerMixin, ClusterMixin, BaseEstimator
from ..externals.six.moves import xrange
from ..utils import check_array
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..exceptions import NotFittedError, ConvergenceWarning
from .hierarchical import AgglomerativeClustering


def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples = X.shape[0]
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in xrange(n_samples):
        row = np.zeros(X.shape[1])
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row


def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_node2 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]

    farthest_idx = np.unravel_index(
        dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[[farthest_idx]]

    node1_closer = node1_dist < node2_dist
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _CFNode(object):
    
```
32 - /tmp/repos/scikit-learn/examples/cluster/plot_linkage_comparison.py
```python
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # ============
    # Create cluster objects
    # ============
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward')
    complete = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='complete')
    average = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='average')
    single = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='single')

    clustering_algorithms = (
        ('Single Linkage', single),
        ('Average Linkage', average),
        ('Complete Linkage', complete),
        ('Ward Linkage', ward),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

```
**33 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
"""Implements the Birch clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    Parameters
    ----------
    threshold : float, default 0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default 50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model, default 3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - `sklearn.cluster` Estimator : If a model is provided, the model is
          fit treating the subclusters as new samples and the initial data is
          mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default True
        Whether or not to compute labels for each fit.

    copy : bool, default True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray,
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray,
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,
    ... compute_labels=True)
    >>> brc.fit(X)
    Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,
       threshold=0.5)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.
    """

    
```
34 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _yield_outliers_checks(name, estimator):

    # checks for all outlier detectors
    yield check_outliers_fit_predict

    # checks for estimators that can be used on a test set
    if hasattr(estimator, 'predict'):
        yield check_outliers_train
        # test outlier detectors can handle non-array data
        yield check_classifier_data_not_an_array
        # test if NotFittedError is raised
        yield check_estimators_unfitted


def _yield_all_checks(name, estimator):
    for check in _yield_non_meta_checks(name, estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(name, estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(name, estimator):
            yield check
    if hasattr(estimator, 'transform'):
        for check in _yield_transformer_checks(name, estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(name, estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(name, estimator):
            yield check
    yield check_fit2d_predict1d
    yield check_methods_subset_invariance
    if name != 'GaussianProcess':  # FIXME
        # XXX GaussianProcess deprecated in 0.20
        yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_fit1d
    yield check_get_params_invariance
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters


def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions.

    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    This test can be applied to classes or instances.
    Classes currently have some additional tests that related to construction,
    while passing instances allows the testing of multiple options.

    Parameters
    ----------
    estimator : estimator object or class
        Estimator to check. Estimator is a class object or instance.

    """
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        try:
            check(name, estimator)
        except SkipTest as message:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(message, SkipTestWarning)


def _boston_subset(n_samples=200):
    global BOSTON
    if BOSTON is None:
        boston = load_boston()
        X, y = boston.data, boston.target
        X, y = shuffle(X, y, random_state=0)
        X, y = X[:n_samples], y[:n_samples]
        X = StandardScaler().fit_transform(X)
        BOSTON = X, y
    return BOSTON



```
35 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python



# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
deprecation_msg = ('Scoring method mean_squared_error was renamed to '
                   'neg_mean_squared_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_squared_error_scorer = make_scorer(mean_squared_error,
                                        greater_is_better=False)
mean_squared_error_scorer._deprecation_msg = deprecation_msg
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)
deprecation_msg = ('Scoring method mean_absolute_error was renamed to '
                   'neg_mean_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                         greater_is_better=False)
mean_absolute_error_scorer._deprecation_msg = deprecation_msg
neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)
deprecation_msg = ('Scoring method median_absolute_error was renamed to '
                   'neg_median_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
median_absolute_error_scorer = make_scorer(median_absolute_error,
                                           greater_is_better=False)
median_absolute_error_scorer._deprecation_msg = deprecation_msg


# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
deprecation_msg = ('Scoring method log_loss was renamed to '
                   'neg_log_loss in version 0.18 and will be removed in 0.20.')
log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                              needs_proba=True)
log_loss_scorer._deprecation_msg = deprecation_msg
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)



```
36 - /tmp/repos/scikit-learn/examples/cluster/plot_cluster_comparison.py
```python
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

```
37 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    # this estimator raises
    # ValueError: Found array with 0 feature(s) (shape=(23, 0))
    # while a minimum of 1 is required.
    # error
    if name in ['SpectralCoclustering']:
        return
    rnd = np.random.RandomState(0)
    if name in ['RANSACRegressor']:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))

    X = pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function",
                   "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert_dict_equal(estimator.__dict__, dict_before,
                              'Estimator changes __dict__ during %s' % method)


def is_public_parameter(attr):
    return not (attr.startswith('_') or attr.endswith('_'))


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


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(ValueError, "Reshape your data",
                                 getattr(estimator, method), X[0])



```
**38 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3,
                 compute_labels=True, copy=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy

    def fit(self, X, y=None):
        """
        Build a CF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        y : Ignored

        """
        self.fit_, self.partial_fit_ = True, False
        return self._fit(X)

    def _fit(self, X):
        X = check_array(X, accept_sparse='csr', copy=self.copy)
        threshold = self.threshold
        branching_factor = self.branching_factor

        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")
        n_samples, n_features = X.shape

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        partial_fit = getattr(self, 'partial_fit_')
        has_root = getattr(self, 'root_', None)
        if getattr(self, 'fit_') or (partial_fit and not has_root):
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _CFNode(threshold, branching_factor, is_leaf=True,
                                 n_features=n_features)

            # To enable getting back subclusters.
            self.dummy_leaf_ = _CFNode(threshold, branching_factor,
                                       is_leaf=True, n_features=n_features)
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold, branching_factor,
                                     is_leaf=False,
                                     n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        centroids = np.concatenate([
            leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids

        self._global_clustering(X)
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the CF Node.

        Returns
        -------
        leaves : array-like
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def partial_fit(self, X=None, y=None):
        """
        Online learning. Prevents rebuilding of CFTree from scratch.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), None
            Input data. If X is not provided, only the global clustering
            step is done.

        y : Ignored

        """
        self.partial_fit_, self.fit_ = True, False
        if X is None:
            # Perform just the final global clustering step.
            self._global_clustering()
            return self
        else:
            self._check_fit(X)
            return self._fit(X)

    def _check_fit(self, X):
        is_fitted = hasattr(self, 'subcluster_centers_')

        # Called by partial_fit, before fitting.
        has_partial_fit = hasattr(self, 'partial_fit_')

        # Should raise an error if one does not fit before predicting.
        if not (is_fitted or has_partial_fit):
            raise NotFittedError("Fit training data before predicting")

        if is_fitted and X.shape[1] != self.subcluster_centers_.shape[1]:
            raise ValueError(
                "Training data and predicted data do "
                "not have same number of features.")

    
```
39 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python



@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = ['Ridge', 'SVR', 'NuSVR', 'NuSVC',
                            'RidgeClassifier', 'SVC', 'RandomizedLasso',
                            'LogisticRegressionCV', 'LinearSVC',
                            'LogisticRegression']

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = multioutput_estimator_convert_y_2d(estimator, y_)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
            estimator.fit(X, y_)

        # HuberRegressor depends on scipy.optimize.fmin_l_bfgs_b
        # which doesn't return a n_iter for old versions of SciPy.
        if not (name == 'HuberRegressor' and estimator.n_iter_ is None):
            assert_greater_equal(estimator.n_iter_, 1)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                               random_state=0, n_features=2, cluster_std=0.1)
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert_greater_equal(iter_, 1)
        else:
            assert_greater_equal(estimator.n_iter_, 1)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    class T(BaseEstimator):
        """Mock classifier
        """

        def __init__(self):
            pass

        def fit(self, X, y):
            return self

        def transform(self, X):
            return X

    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert_true(all(item in deep_params.items() for item in
                    shallow_params.items()))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    boston = load_boston()
    X, y = boston.data, boston.target
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    assert_raises_regex(ValueError, msg, e.fit, X, y)



```
40 - /tmp/repos/scikit-learn/examples/covariance/plot_outlier_detection.py
```python
for _, offset in enumerate(clusters_separation):
    np.random.seed(SEED)
    # Data generation
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.concatenate([X1, X2], axis=0)
    # Add outliers
    X = np.concatenate([X, np.random.uniform(low=-6, high=6,
                       size=(n_outliers, 2))], axis=0)

    # Fit the model
    plt.figure(figsize=(9, 7))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_
        else:
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points
        if clf_name == "Local Outlier Factor":
            # decision_function is private for LOF
            Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(2, 2, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                         cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[0],
                            linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[0, Z.max()],
                         colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    plt.suptitle("Outlier detection")

plt.show()

```
41 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            if positive:
                sign_active[n_active] = np.ones_like(C_)
            else:
                sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added.

                # Note: this case is very rare. It is no longer triggered by
                # the test suite. The `equality_tolerance` margin added in 0.16
                # to get early stopping to work consistently on all versions of
                # Python including 32 bit Python under Windows seems to make it
                # very difficult to trigger the 'drop for good' strategy.
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e.'
                              ' Reduce max_iter or increase eps parameters.'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)

                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        
```
**42 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
```python
fun_args = {} if fun_args is None else fun_args
    # make interface compatible with other decompositions
    # a copy is required only for non whitened data
    X = check_array(X, copy=whiten, dtype=FLOAT_DTYPES,
                    ensure_min_samples=2).T

    alpha = fun_args.get('alpha', 1.0)
    if not 1 <= alpha <= 2:
        raise ValueError('alpha must be in [1,2]')

    if fun == 'logcosh':
        g = _logcosh
    elif fun == 'exp':
        g = _exp
    elif fun == 'cube':
        g = _cube
    elif callable(fun):
        def g(x, fun_args):
            return fun(x, **fun_args)
    else:
        exc = ValueError if isinstance(fun, six.string_types) else TypeError
        raise exc("Unknown function %r;"
                  " should be one of 'logcosh', 'exp', 'cube' or callable"
                  % fun)

    n, p = X.shape

    if not whiten and n_components is not None:
        n_components = None
        warnings.warn('Ignoring n_components with whiten=False.')

    if n_components is None:
        n_components = min(n, p)
    if (n_components > min(n, p)):
        n_components = min(n, p)
        warnings.warn('n_components is too large: it will be set to %s' % n_components)

    if whiten:
        # Centering the columns (ie the variables)
        X_mean = X.mean(axis=-1)
        X -= X_mean[:, np.newaxis]

        # Whitening and preprocessing by PCA
        u, d, _ = linalg.svd(X, full_matrices=False)

        del _
        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, X)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(p)
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = as_float_array(X, copy=False)  # copy has been taken care of

    if w_init is None:
        w_init = np.asarray(random_state.normal(size=(n_components,
                            n_components)), dtype=X1.dtype)

    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_components, n_components):
            raise ValueError('w_init has invalid shape -- should be %(shape)s'
                             % {'shape': (n_components, n_components)})

    kwargs = {'tol': tol,
              'g': g,
              'fun_args': fun_args,
              'max_iter': max_iter,
              'w_init': w_init}

    if algorithm == 'parallel':
        W, n_iter = _ica_par(X1, **kwargs)
    elif algorithm == 'deflation':
        W, n_iter = _ica_def(X1, **kwargs)
    else:
        raise ValueError('Invalid algorithm: must be either `parallel` or'
                         ' `deflation`.')
    del X1

    if whiten:
        if compute_sources:
            S = np.dot(np.dot(W, K), X).T
        else:
            S = None
        if return_X_mean:
            if return_n_iter:
                return K, W, S, X_mean, n_iter
            else:
                return K, W, S, X_mean
        else:
            if return_n_iter:
                return K, W, S, n_iter
            else:
                return K, W, S

    else:
        if compute_sources:
            S = np.dot(W, X).T
        else:
            S = None
        if return_X_mean:
            if return_n_iter:
                return None, W, S, None, n_iter
            else:
                return None, W, S, None
        else:
            if return_n_iter:
                return None, W, S, n_iter
            else:
                return None, W, S


class FastICA(BaseEstimator, TransformerMixin):
    
```
43 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
for i in range(n_refinements):
            with warnings.catch_warnings():
                # No need to see the convergence warnings on this grid:
                # they will always be points that will not converge
                # during the cross-validation
                warnings.simplefilter('ignore', ConvergenceWarning)
                # Compute the cross-validated loss on the current grid

                # NOTE: Warm-restarting graphical_lasso_path has been tried,
                # and this did not allow to gain anything
                # (same execution time with or without).
                this_path = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )(delayed(graphical_lasso_path)(X[train], alphas=alphas,
                                                X_test=X[test], mode=self.mode,
                                                tol=self.tol,
                                                enet_tol=self.enet_tol,
                                                max_iter=int(.1 *
                                                             self.max_iter),
                                                verbose=inner_verbose)
                  for train, test in cv.split(X, y))

            # Little danse to transform the list in what we need
            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using built in 'max' function to
            # have a fully-reproducible selection of the smallest alpha
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (alpha, scores, _) in enumerate(path):
                this_score = np.mean(scores)
                if this_score >= .1 / np.finfo(np.float64).eps:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # Refine the grid
            if best_index == 0:
                # We do not need to go back: we have chosen
                # the highest value of alpha for which there are
                # non-zero coefficients
                alpha_1 = path[0][0]
                alpha_0 = path[1][0]
            elif (best_index == last_finite_idx
                    and not best_index == len(path) - 1):
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                alpha_1 = path[best_index][0]
                alpha_0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                alpha_1 = path[best_index][0]
                alpha_0 = 0.01 * path[best_index][0]
            else:
                alpha_1 = path[best_index - 1][0]
                alpha_0 = path[best_index + 1][0]

            if not isinstance(n_alphas, collections.Sequence):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0),
                                     n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print('[GraphicalLassoCV] Done refinement % 2i out of'
                      ' %i: % 3is' % (i + 1, n_refinements, time.time() - t0))

        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas = list(path[0])
        # Finally, compute the score with alpha = 0
        alphas.append(0)
        grid_scores.append(cross_val_score(EmpiricalCovariance(), X,
                                           cv=cv, n_jobs=self.n_jobs,
                                           verbose=inner_verbose))
        self.grid_scores_ = np.array(grid_scores)
        best_alpha = alphas[best_index]
        self.alpha_ = best_alpha
        self.cv_alphas_ = alphas

        # Finally fit the model with the selected alpha
        self.covariance_, self.precision_, self.n_iter_ = graphical_lasso(
            emp_cov, alpha=best_alpha, mode=self.mode, tol=self.tol,
            enet_tol=self.enet_tol, max_iter=self.max_iter,
            verbose=inner_verbose, return_n_iter=True)
        return self
```
44 - /tmp/repos/scikit-learn/examples/applications/plot_outlier_detection_housing.py
```python
"""
====================================
Outlier detection on a real data set
====================================

This example illustrates the need for robust covariance estimation
on a real data set. It is useful both for outlier detection and for
a better understanding of the data structure.

We selected two sets of two variables from the Boston housing data set
as an illustration of what kind of analysis can be done with several
outlier detection tools. For the purpose of visualization, we are working
with two-dimensional examples, but one should be aware that things are
not so trivial in high-dimension, as it will be pointed out.

In both examples below, the main result is that the empirical covariance
estimate, as a non-robust one, is highly influenced by the heterogeneous
structure of the observations. Although the robust covariance estimate is
able to focus on the main mode of the data distribution, it sticks to the
assumption that the data should be Gaussian distributed, yielding some biased
estimation of the data structure, but yet accurate to some extent.
The One-Class SVM does not assume any parametric form of the data distribution
and can therefore model the complex shape of the data much better.

First example
-------------
The first example illustrates how robust covariance estimation can help
concentrating on a relevant cluster when another one exists. Here, many
observations are confounded into one and break down the empirical covariance
estimation.
Of course, some screening tools would have pointed out the presence of two
clusters (Support Vector Machines, Gaussian Mixture Models, univariate
outlier detection, ...). But had it been a high-dimensional example, none
of these could be applied that easily.

Second example
--------------
The second example shows the ability of the Minimum Covariance Determinant
robust estimator of covariance to concentrate on the main mode of the data
distribution: the location seems to be well estimated, although the covariance
is hard to estimate due to the banana-shaped distribution. Anyway, we can
get rid of some outlying observations.
The One-Class SVM is able to capture the real data structure, but the
difficulty is to adjust its kernel bandwidth parameter so as to obtain
a good compromise between the shape of the data scatter matrix and the
risk of over-fitting the data.

"""
print(__doc__)

# Author: Virgile Fritsch <virgile.fritsch@inria.fr>
# License: BSD 3 clause

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston

# Get data
X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

# Define "classifiers" to be used
classifiers = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1.,
                                             contamination=0.261),
    "Robust Covariance (Minimum Covariance Determinant)":
    EllipticEnvelope(contamination=0.261),
    "OCSVM": OneClassSVM(nu=0.261, gamma=0.05)}
colors = ['m', 'g', 'b']
legend1 = {}
legend2 = {}

# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(-8, 28, 500), np.linspace(3, 40, 500))
xx2, yy2 = np.meshgrid(np.linspace(3, 10, 500), np.linspace(-5, 45, 500))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1)
    clf.fit(X1)
    Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])
    plt.figure(2)
    clf.fit(X2)
    Z2 = clf.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
    Z2 = Z2.reshape(xx2.shape)
    legend2[clf_name] = plt.contour(
        xx2, yy2, Z2, levels=[0], linewidths=2, colors=colors[i])

legend1_values_list = list(legend1.values())
legend1_keys_list = list(legend1.keys())

# Plot the results (= shape of the data points cloud)
plt.figure(1)  # two clusters
plt.title("Outlier detection on a real data set (boston housing)")

```
**45 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py**:
```python
e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn("Affinity propagation did not converge, this model "
                      "will not have any cluster centers.", ConvergenceWarning)
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


###############################################################################

class AffinityPropagation(BaseEstimator, ClusterMixin):
    
```
46 - /tmp/repos/scikit-learn/examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py
```python
"""
==============================================
Feature agglomeration vs. univariate selection
==============================================

This example compares 2 dimensionality reduction strategies:

- univariate feature selection with Anova

- feature agglomeration with Ward hierarchical clustering

Both methods are compared in a regression problem using
a BayesianRidge as supervised estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

print(__doc__)

import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# #############################################################################
# Generate data
n_samples = 200
size = 40  # image size
roi_size = 15
snr = 5.
np.random.seed(0)
mask = np.ones([size, size], dtype=np.bool)

coef = np.zeros((size, size))
coef[0:roi_size, 0:roi_size] = -1.
coef[-roi_size:, -roi_size:] = 1.

X = np.random.randn(n_samples, size ** 2)
for x in X:  # smooth data
    x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()
X -= X.mean(axis=0)
X /= X.std(axis=0)

y = np.dot(X, coef.ravel())
noise = np.random.randn(y.shape[0])
noise_coef = (linalg.norm(y, 2) / np.exp(snr / 20.)) / linalg.norm(noise, 2)
y += noise_coef * noise  # add noise

# #############################################################################
# Compute the coefs of a Bayesian Ridge with GridSearch
cv = KFold(2)  # cross-validation generator for model selection
ridge = BayesianRidge()
cachedir = tempfile.mkdtemp()
mem = Memory(cachedir=cachedir, verbose=1)

# Ward agglomeration followed by BayesianRidge
connectivity = grid_to_graph(n_x=size, n_y=size)
ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity,
                            memory=mem)
clf = Pipeline([('ward', ward), ('ridge', ridge)])
# Select the optimal number of parcels with grid search
clf = GridSearchCV(clf, {'ward__n_clusters': [10, 20, 30]}, n_jobs=1, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)
coef_agglomeration_ = coef_.reshape(size, size)

# Anova univariate feature selection followed by BayesianRidge
f_regression = mem.cache(feature_selection.f_regression)  # caching function
anova = feature_selection.SelectPercentile(f_regression)
clf = Pipeline([('anova', anova), ('ridge', ridge)])
# Select the optimal percentage of features with grid search
clf = GridSearchCV(clf, {'anova__percentile': [5, 10, 20]}, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_.reshape(1, -1))
coef_selection_ = coef_.reshape(size, size)

# #############################################################################
# Inverse the transformation to plot the results on an image
plt.close('all')
plt.figure(figsize=(7.3, 2.7))
plt.subplot(1, 3, 1)
plt.imshow(coef, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("True weights")
plt.subplot(1, 3, 2)
plt.imshow(coef_selection_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Selection")
plt.subplot(1, 3, 3)
plt.imshow(coef_agglomeration_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Agglomeration")
plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.16, 0.26)
plt.show()

# Attempt to remove the temporary cachedir, but don't worry if it fails
shutil.rmtree(cachedir, ignore_errors=True)
```
47 - /tmp/repos/scikit-learn/sklearn/utils/deprecation.py
```python
import sys
import warnings
import functools

__all__ = ["deprecated", "DeprecationDict"]


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from sklearn.utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <sklearn.utils.deprecation.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass

    Parameters
    ----------
    extra : string
          to be added to the deprecation messages
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):
        self.extra = extra

    def __call__(self, obj):
        """Call method

        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


def _is_deprecated(func):
    """Helper to check if func is wraped by our deprecated decorator"""
    if sys.version_info < (3, 5):
        raise NotImplementedError("This is only available for python3.5 "
                                  "or above")
    closures = getattr(func, '__closure__', [])
    if closures is None:
        closures = []
    is_deprecated = ('deprecated' in ''.join([c.cell_contents
                                              for c in closures
                     if isinstance(c.cell_contents, str)]))
    return is_deprecated


class DeprecationDict(dict):
    """A dict which raises a warning when some keys are looked up

    Note, this does not raise a warning for __contains__ and iteration.

    It also will raise a warning even after the key has been manually set by
    the user.
    """
    def __init__(self, *args, **kwargs):
        self._deprecations = {}
        super(DeprecationDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self._deprecations:
            warn_args, warn_kwargs = self._deprecations[key]
            warnings.warn(*warn_args, **warn_kwargs)
        return super(DeprecationDict, self).__getitem__(key)

    def get(self, key, default=None):
        # dict does not implement it like this, hence it needs to be overridden
        try:
            return self[key]
        except KeyError:
            return default

    def add_warning(self, key, *args, **kwargs):
        """Add a warning to be triggered when the specified key is read"""
        self._deprecations[key] = (args, kwargs)

```
48 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Eric Martin <eric@ericmart.in>
#          Giorgio Patrini <giorgio.patrini@anu.edu.au>
#          Eric Chang <ericchang2017@u.northwestern.edu>
# License: BSD 3 clause

from __future__ import division

from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse
from scipy import stats

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six import string_types
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.extmath import _incremental_mean_and_var
from ..utils.fixes import _argmax
from ..utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
from ..utils.sparsefuncs import (inplace_column_scale,
                                 mean_variance_axis, incr_mean_variance_axis,
                                 min_max_axis)
from ..utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)
from .label import LabelEncoder


BOUNDS_THRESHOLD = 1e-7


zip = six.moves.zip
map = six.moves.map
range = six.moves.range

__all__ = [
    'Binarizer',
    'KernelCenterer',
    'MinMaxScaler',
    'MaxAbsScaler',
    'Normalizer',
    'OneHotEncoder',
    'RobustScaler',
    'StandardScaler',
    'QuantileTransformer',
    'PowerTransformer',
    'add_dummy_feature',
    'binarize',
    'normalize',
    'scale',
    'robust_scale',
    'maxabs_scale',
    'minmax_scale',
    'quantile_transform',
    'power_transform',
]


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    
```
**49 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
"""Each node in a CFTree is called a CFNode.

    The CFNode can have a maximum of branching_factor
    number of CFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a CFSubcluster.

    branching_factor : int
        Maximum number of CF subclusters in each node.

    is_leaf : bool
        We need to know if the CFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : array-like
        list of subclusters for a particular CFNode.

    prev_leaf_ : _CFNode
        prev_leaf. Useful only if is_leaf is True.

    next_leaf_ : _CFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray, shape (branching_factor + 1, n_features)
        manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    init_sq_norm_ : ndarray, shape (branching_factor + 1,)
        manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.

    centroids_ : ndarray
        view of ``init_centroids_``.

    squared_norm_ : ndarray
        view of ``init_sq_norm_``.

    """
    def __init__(self, threshold, branching_factor, is_leaf, n_features):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        self.init_sq_norm_ = np.zeros((branching_factor + 1))
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_

        # Keep centroids and squared norm as views. In this way
        # if we change init_centroids and init_sq_norm_, it is
        # sufficient,
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]

    def update_split_subclusters(self, subcluster,
                                 new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        self.append_subcluster(new_subcluster2)

    
```
50 - /tmp/repos/scikit-learn/sklearn/setup.py
```python
import os
from os.path import join
import warnings

from sklearn._build_utils import maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sklearn', parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('covariance')
    config.add_subpackage('covariance/tests')
    config.add_subpackage('cross_decomposition')
    config.add_subpackage('cross_decomposition/tests')
    config.add_subpackage('feature_selection')
    config.add_subpackage('feature_selection/tests')
    config.add_subpackage('gaussian_process')
    config.add_subpackage('gaussian_process/tests')
    config.add_subpackage('mixture')
    config.add_subpackage('mixture/tests')
    config.add_subpackage('model_selection')
    config.add_subpackage('model_selection/tests')
    config.add_subpackage('neural_network')
    config.add_subpackage('neural_network/tests')
    config.add_subpackage('preprocessing')
    config.add_subpackage('preprocessing/tests')
    config.add_subpackage('semi_supervised')
    config.add_subpackage('semi_supervised/tests')

    # submodules which have their own setup.py
    # leave out "linear_model" and "utils" for now; add them after cblas below
    config.add_subpackage('cluster')
    config.add_subpackage('datasets')
    config.add_subpackage('decomposition')
    config.add_subpackage('ensemble')
    config.add_subpackage('externals')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('manifold')
    config.add_subpackage('metrics')
    config.add_subpackage('metrics/cluster')
    config.add_subpackage('neighbors')
    config.add_subpackage('tree')
    config.add_subpackage('svm')

    # add cython extension module for isotonic regression
    config.add_extension('_isotonic',
                         sources=['_isotonic.pyx'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         )

    # some libs needs cblas, fortran-compiled BLAS will not be sufficient
    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or (
            ('NO_ATLAS_INFO', 1) in blas_info.get('define_macros', [])):
        config.add_library('cblas',
                           sources=[join('src', 'cblas', '*.c')])
        warnings.warn(BlasNotFoundError.__doc__)

    # the following packages depend on cblas, so they have to be build
    # after the above.
    config.add_subpackage('linear_model')
    config.add_subpackage('utils')

    # add the test directory
    config.add_subpackage('tests')

    maybe_cythonize_extensions(top_path, config)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

```
51 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def assert_no_warnings(func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        if len(w) > 0:
            raise AssertionError("Got warnings when calling %s: [%s]"
                                 % (func.__name__,
                                    ', '.join(str(warning) for warning in w)))
    return result


def ignore_warnings(obj=None, category=Warning):
    """Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    category : warning class, defaults to Warning.
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...    warnings.warn('buhuhuhu')
    ...    print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    """
    if callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        return _IgnoreWarnings(category=category)


class _IgnoreWarnings(object):
    """Improved and simplified Python warnings context manager and decorator.

    This class allows the user to ignore the warnings raised by a function.
    Copied from Python 2.7.5 and modified as required.

    Parameters
    ----------
    category : tuple of warning class, default to Warning
        The category to filter. By default, all the categories will be muted.

    """

    def __init__(self, category):
        self._record = True
        self._module = sys.modules['warnings']
        self._entered = False
        self.log = []
        self.category = category

    def __call__(self, fn):
        """Decorator to catch and hide warnings without visual nesting."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # very important to avoid uncontrolled state propagation
            clean_warning_registry()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", self.category)
                return fn(*args, **kwargs)

        return wrapper

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules['warnings']:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        clean_warning_registry()  # be safe and not propagate state + chaos
        warnings.simplefilter("ignore", self.category)
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning
        self.log[:] = []
        clean_warning_registry()  # be safe and not propagate state + chaos


assert_less = _dummy.assertLess
assert_greater = _dummy.assertGreater

assert_allclose = np.testing.assert_allclose


```
52 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python


rcv1 = fetch_rcv1()
X = rcv1.data
n_samples, n_features = X.shape

# consider the binary classification problem 'CCAT' vs the rest
ccat_idx = rcv1.target_names.tolist().index('CCAT')
y = rcv1.target.tocsc()[:, ccat_idx].toarray().ravel().astype(np.float64)
y[y == 0] = -1

# parameters
C = 1.
fit_intercept = True
tol = 1.0e-14

# max_iter range
sgd_iter_range = list(range(1, 121, 10))
newton_iter_range = list(range(1, 25, 3))
lbfgs_iter_range = list(range(1, 242, 12))
liblinear_iter_range = list(range(1, 37, 3))
liblinear_dual_iter_range = list(range(1, 85, 6))
sag_iter_range = list(range(1, 37, 3))

clfs = [
    ("LR-liblinear",
     LogisticRegression(C=C, tol=tol,
                        solver="liblinear", fit_intercept=fit_intercept,
                        intercept_scaling=1),
     liblinear_iter_range, [], [], [], []),
    ("LR-liblinear-dual",
     LogisticRegression(C=C, tol=tol, dual=True,
                        solver="liblinear", fit_intercept=fit_intercept,
                        intercept_scaling=1),
     liblinear_dual_iter_range, [], [], [], []),
    ("LR-SAG",
     LogisticRegression(C=C, tol=tol,
                        solver="sag", fit_intercept=fit_intercept),
     sag_iter_range, [], [], [], []),
    ("LR-newton-cg",
     LogisticRegression(C=C, tol=tol, solver="newton-cg",
                        fit_intercept=fit_intercept),
     newton_iter_range, [], [], [], []),
    ("LR-lbfgs",
     LogisticRegression(C=C, tol=tol,
                        solver="lbfgs", fit_intercept=fit_intercept),
     lbfgs_iter_range, [], [], [], []),
    ("SGD",
     SGDClassifier(alpha=1.0 / C / n_samples, penalty='l2', loss='log',
                   fit_intercept=fit_intercept, verbose=0),
     sgd_iter_range, [], [], [], [])]


if lightning_clf is not None and not fit_intercept:
    alpha = 1. / C / n_samples
    # compute the same step_size than in LR-sag
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log",
                                   fit_intercept)

    clfs.append(
        ("Lightning-SVRG",
         lightning_clf.SVRGClassifier(alpha=alpha, eta=step_size,
                                      tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))
    clfs.append(
        ("Lightning-SAG",
         lightning_clf.SAGClassifier(alpha=alpha, eta=step_size,
                                     tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))

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
53 - /tmp/repos/scikit-learn/examples/covariance/plot_outlier_detection.py
```python
"""
==========================================
Outlier detection with several methods.
==========================================

When the amount of contamination is known, this example illustrates three
different ways of performing :ref:`outlier_detection`:

- based on a robust estimator of covariance, which is assuming that the
  data are Gaussian distributed and performs better than the One-Class SVM
  in that case.

- using the One-Class SVM and its ability to capture the shape of the
  data set, hence performing better when the data is strongly
  non-Gaussian, i.e. with two well-separated clusters;

- using the Isolation Forest algorithm, which is based on random forests and
  hence more adapted to large-dimensional settings, even if it performs
  quite well in the examples below.

- using the Local Outlier Factor to measure the local deviation of a given
  data point with respect to its neighbors by comparing their local density.

The ground truth about inliers and outliers is given by the points colors
while the orange-filled area indicates which points are reported as inliers
by each method.

Here, we assume that we know the fraction of outliers in the datasets.
Thus rather than using the 'predict' method of the objects, we set the
threshold on the decision_function to separate out the corresponding
fraction.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

SEED = 42
GRID_PRECISION = 100

rng = np.random.RandomState(SEED)

# Example settings
n_samples = 200
outliers_fraction = 0.25
clusters_separation = (0, 1, 2)

# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                     kernel="rbf", gamma=0.1),
    "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
    "Isolation Forest": IsolationForest(max_samples=n_samples,
                                        contamination=outliers_fraction,
                                        random_state=rng),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=35,
        contamination=outliers_fraction)}

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, GRID_PRECISION),
                     np.linspace(-7, 7, GRID_PRECISION))
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = -1

# Fit the problem with varying cluster separation

```
54 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

from __future__ import division

import numpy as np
import scipy as sp
import warnings
from warnings import warn
from sklearn.utils.fixes import euler_gamma

from scipy.sparse import issparse

import numbers
from ..externals import six
from ..tree import ExtraTreeRegressor
from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..base import OutlierMixin

from .bagging import BaseBagging

__all__ = ["IsolationForest"]

INTEGER_TYPES = (numbers.Integral, np.integer)


class IsolationForest(BaseBagging, OutlierMixin):
    
```
55 - /tmp/repos/scikit-learn/examples/decomposition/plot_pca_vs_fa_model_selection.py
```python



for X, title in [(X_homo, 'Homoscedastic Noise'),
                 (X_hetero, 'Heteroscedastic Noise')]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()

```
56 - /tmp/repos/scikit-learn/examples/preprocessing/plot_discretization_classification.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
Feature discretization
======================

A demonstration of feature discretization on synthetic classification datasets.
Feature discretization decomposes each feature into a set of bins, here equally
distributed in width. The discrete values are then one-hot encoded, and given
to a linear classifier. This preprocessing enables a non-linear behavior even
though the classifier is linear.

On this example, the first two rows represent linearly non-separable datasets
(moons and concentric circles) while the third is approximately linearly
separable. On the two linearly non-separable datasets, feature discretization
largely increases the performance of linear classifiers. On the linearly
separable dataset, feature discretization decreases the performance of linear
classifiers. Two non-linear classifiers are also shown for comparison.

This example should be taken with a grain of salt, as the intuition conveyed
does not necessarily carry over to real datasets. Particularly in
high-dimensional spaces, data can more easily be separated linearly. Moreover,
using feature discretization and one-hot encoding increases the number of
features, which easily lead to overfitting when the number of samples is small.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

print(__doc__)

h = .02  # step size in the mesh


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == 'Pipeline':
        name = [get_name(est[1]) for est in estimator.steps]
        name = ' + '.join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
classifiers = [
    (LogisticRegression(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (LinearSVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'),
        LogisticRegression(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'logisticregression__C': np.logspace(-2, 7, 10),
        }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'), LinearSVC(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'linearsvc__C': np.logspace(-2, 7, 10),
        }),
    (GradientBoostingClassifier(n_estimators=50, random_state=0), {
        'learning_rate': np.logspace(-4, 0, 10)
    }),
    (SVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
]

names = [get_name(e) for e, g in classifiers]

n_samples = 100
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                        n_informative=2, random_state=2,
                        n_clusters_per_class=1)
]

fig, axes = plt.subplots(nrows=len(datasets), ncols=len(classifiers) + 1,
                         figsize=(21, 9))

cm = plt.cm.PiYG
cm_bright = ListedColormap(['#b30065', '#178000'])

# iterate over datasets

```
57 - /tmp/repos/scikit-learn/examples/cluster/plot_cluster_comparison.py
```python
"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]


```
58 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            X -= X.min()
        X = pairwise_estimator_convert_X(X, classifier_orig)
        set_random_state(classifier)
        # raises error on malformed input for fit
        with assert_raises(ValueError, msg="The classifier {} does not"
                           " raise an error when incorrect/malformed input "
                           "data for fit is passed. The number of training "
                           "examples is not the same as the number of labels."
                           " Perhaps use check_X_y in fit.".format(name)):
            classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert_true(hasattr(classifier, "classes_"))
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        if name not in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        if _is_pairwise(classifier):
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when shape of X"
                               "in predict is not equal to (n_test_samples,"
                               "n_training_samples)".format(name)):
                classifier.predict(X.reshape(-1, 1))
        else:
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when the number of features "
                               "in predict is different from the number of"
                               " features in fit.".format(name)):
                classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    assert_equal(decision.shape, (n_samples,))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                if (n_classes == 3 and
                        # 1on1 of LibSVM works differently
                        not isinstance(classifier, BaseLibSVM)):
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if _is_pairwise(classifier):
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the  "
                                       "shape of X in decision_function is "
                                       "not equal to (n_test_samples, "
                                       "n_training_samples) in fit."
                                       .format(name)):
                        classifier.decision_function(X.reshape(-1, 1))
                else:
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the number "
                                       "of features in decision_function is "
                                       "different from the number of features"
                                       " in fit.".format(name)):
                        classifier.decision_function(X.T)
            except NotImplementedError:
                pass
        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert_equal(y_prob.shape, (n_samples, n_classes))
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
            # raises error on malformed input for predict_proba
            if _is_pairwise(classifier_orig):
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the shape of X"
                                   "in predict_proba is not equal to "
                                   "(n_test_samples, n_training_samples)."
                                   .format(name)):
                    classifier.predict_proba(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the number of "
                                   "features in predict_proba is different "
                                   "from the number of features in fit."
                                   .format(name)):
                    classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))



```
**59 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py**:
```python
def _pre_compute(self, X, y, centered_kernel=True):
        # even if X is very sparse, K is usually very dense
        K = safe_sparse_dot(X, X.T, dense_output=True)
        # the following emulates an additional constant regressor
        # corresponding to fit_intercept=True
        # but this is done only when the features have been centered
        if centered_kernel:
            K += np.ones_like(K)
        v, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return v, Q, QT_y

    def _decomp_diag(self, v_prime, Q):
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q ** 2).sum(axis=-1)

    def _diag_dot(self, D, B):
        # compute dot(diag(D), B)
        if len(B.shape) > 1:
            # handle case where B is > 1-d
            D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
        return D * B

    def _errors_and_values_helper(self, alpha, y, v, Q, QT_y):
        """Helper function to avoid code duplication between self._errors and
        self._values.

        Notes
        -----
        We don't construct matrix G, instead compute action on y & diagonal.
        """
        w = 1. / (v + alpha)
        constant_column = np.var(Q, 0) < 1.e-12
        # detect constant columns
        w[constant_column] = 0  # cancel the regularization for the intercept

        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return (c / G_diag) ** 2, c

    def _values(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return y - (c / G_diag), c

    def _pre_compute_svd(self, X, y, centered_kernel=True):
        if sparse.issparse(X):
            raise TypeError("SVD not supported for sparse matrices")
        if centered_kernel:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # to emulate fit_intercept=True situation, add a column on ones
        # Note that by centering, the other columns are orthogonal to that one
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s ** 2
        UT_y = np.dot(U.T, y)
        return v, U, UT_y

    def _errors_and_values_svd_helper(self, alpha, y, v, U, UT_y):
        """Helper function to avoid code duplication between self._errors_svd
        and self._values_svd.
        """
        constant_column = np.var(U, 0) < 1.e-12
        # detect columns colinear to ones
        w = ((v + alpha) ** -1) - (alpha ** -1)
        w[constant_column] = - (alpha ** -1)
        # cancel the regularization for the intercept
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha ** -1) * y
        G_diag = self._decomp_diag(w, U) + (alpha ** -1)
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return (c / G_diag) ** 2, c

    def _values_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return y - (c / G_diag), c

    
```
60 - /tmp/repos/scikit-learn/examples/plot_anomaly_comparison.py
```python
"""
============================================================================
Comparing anomaly detection algorithms for outlier detection on toy datasets
============================================================================

This example shows characteristics of different anomaly detection algorithms
on 2D datasets. Datasets contain one or two modes (regions of high density)
to illustrate the ability of algorithms to cope with multimodal data.

For each dataset, 15% of samples are generated as random uniform noise. This
proportion is the value given to the nu parameter of the OneClassSVM and the
contamination parameter of the other outlier detection algorithms.
Decision boundaries between inliers and outliers are displayed in black.

Local Outlier Factor (LOF) does not show a decision boundary in black as it
has no predict method to be applied on new data.

While these examples give some intuition about the algorithms, this
intuition might not apply to very high dimensional data.

Finally, note that parameters of the models have been here handpicked but
that in practice they need to be adjusted. In the absence of labelled data,
the problem is completely unsupervised so model selection can be a challenge.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Albert Thomas <albert.thomas@telecom-paristech.fr>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)


```
61 - /tmp/repos/scikit-learn/sklearn/linear_model/coordinate_descent.py
```python
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Gael Varoquaux <gael.varoquaux@inria.fr>
#
# License: BSD 3 clause

import sys
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

from .base import LinearModel, _pre_fit
from ..base import RegressorMixin
from .base import _preprocess_data
from ..utils import check_array, check_X_y
from ..utils.validation import check_random_state
from ..model_selection import check_cv
from ..externals.joblib import Parallel, delayed
from ..externals import six
from ..externals.six.moves import xrange
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..utils.validation import column_or_1d
from ..exceptions import ConvergenceWarning

from . import cd_fast


###############################################################################
# Paths functions


```
62 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_covariances.py
```python


iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))


X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                    label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()

```
63 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
def plot_dloss(clfs):
    plt.figure()
    pobj_final = []
    for (name, _, _, train_losses, _, _, durations) in clfs:
        pobj_final.append(train_losses[-1])

    indices = np.argsort(pobj_final)
    pobj_best = pobj_final[indices[0]]

    for (name, _, _, train_losses, _, _, durations) in clfs:
        log_pobj = np.log(abs(np.array(train_losses) - pobj_best)) / np.log(10)

        plt.plot(durations, log_pobj, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("log(best - train_loss)")


def get_max_squared_sum(X):
    """Get the maximum row-wise sum of squares"""
    return np.sum(X ** 2, axis=1).max()
```
64 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def check_estimator_sparse_data(name, estimator_orig):

    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < .8] = 0
    X = pairwise_estimator_convert_X(X, estimator_orig)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.rand(40)).astype(np.int)
    # catch deprecation warnings
    with ignore_warnings(category=DeprecationWarning):
        estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    for sparse_format in ['csr', 'csc', 'dok', 'lil', 'coo', 'dia', 'bsr']:
        X = X_csr.asformat(sparse_format)
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            if name in ['Scaler', 'StandardScaler']:
                estimator = clone(estimator).set_params(with_mean=False)
            else:
                estimator = clone(estimator)
        # fit and predict
        try:
            with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
                estimator.fit(X, y)
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                assert_equal(pred.shape, (X.shape[0],))
            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(X)
                assert_equal(probs.shape, (X.shape[0], 4))
        except (TypeError, ValueError) as e:
            if 'sparse' not in repr(e).lower():
                print("Estimator %s doesn't seem to fail gracefully on "
                      "sparse data: error message state explicitly that "
                      "sparse input is not supported if this is not the case."
                      % name)
                raise
        except Exception:
            print("Estimator %s doesn't seem to fail gracefully on "
                  "sparse data: it should raise a TypeError if sparse input "
                  "is explicitly not supported." % name)
            raise
```
65 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def check_outliers_fit_predict(name, estimator_orig):
    # Check fit_predict for outlier detectors.

    n_samples = 300
    X, _ = make_blobs(n_samples=n_samples, random_state=0)
    X = shuffle(X, random_state=7)
    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)

    set_random_state(estimator)

    y_pred = estimator.fit_predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    # check fit_predict = fit.predict when the estimator has both a predict and
    # a fit_predict method. recall that it is already assumed here that the
    # estimator has a fit_predict method
    if hasattr(estimator, 'predict'):
        y_pred_2 = estimator.fit(X).predict(X)
        assert_array_equal(y_pred, y_pred_2)

    if hasattr(estimator, "contamination"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'
        expected_outliers = 30
        contamination = float(expected_outliers)/n_samples
        estimator.set_params(contamination=contamination)
        y_pred = estimator.fit_predict(X)

        num_outliers = np.sum(y_pred != 1)
        # num_outliers should be equal to expected_outliers unless
        # there are ties in the decision_function values. this can
        # only be tested for estimators with a decision_function
        # method
        if (num_outliers != expected_outliers and
                hasattr(estimator, 'decision_function')):
            decision = estimator.decision_function(X)
            check_outlier_corruption(num_outliers, expected_outliers, decision)

        # raises error when contamination is a scalar and not in [0,1]
        for contamination in [-0.5, 2.3]:
            estimator.set_params(contamination=contamination)
            assert_raises(ValueError, estimator.fit_predict, X)


def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]
    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = enforce_estimator_tags_y(estimator, y)

    train, test = next(ShuffleSplit(test_size=.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {method: getattr(estimator, method)(X_test)
              for method in check_methods
              if hasattr(estimator, method)}

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2*np.finfo(new_result.dtype).eps
            else:
                tol = 2*np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method], new_result,
                atol=max(tol, 1e-9), rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method)
            )

```
66 - /tmp/repos/scikit-learn/examples/cluster/plot_birch_vs_minibatchkmeans.py
```python
"""
=================================
Compare BIRCH and MiniBatchKMeans
=================================

This example compares the timing of Birch (with and without the global
clustering step) and MiniBatchKMeans on a synthetic dataset having
100,000 samples and 2 features generated using make_blobs.

If ``n_clusters`` is set to None, the data is reduced from 100,000
samples to a set of 158 clusters. This can be viewed as a preprocessing
step before the final (global) clustering step that further reduces these
158 clusters to 100 clusters.
"""

# Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

print(__doc__)

from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs


# Generate centers for the blobs so that it forms a 10 X 10 grid.
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))

# Generate blobs to do a comparison between MiniBatchKMeans and Birch.
X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

# Compute clustering with Birch with and without the final clustering step
# and plot.
birch_models = [Birch(threshold=1.7, n_clusters=None),
                Birch(threshold=1.7, n_clusters=100)]
final_step = ['without global clustering', 'with global clustering']

for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_model.fit(X)
    time_ = time() - t
    print("Birch %s as the final step took %0.2f seconds" % (
          info, (time() - t)))

    # Plot result
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   c='w', edgecolor=col, marker='.', alpha=0.5)
        if birch_model.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], marker='+',
                       c='k', s=25)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)

# Compute clustering with MiniBatchKMeans.
mbk = MiniBatchKMeans(init='k-means++', n_clusters=100, batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)
t0 = time()
mbk.fit(X)
t_mini_batch = time() - t0
print("Time taken to run MiniBatchKMeans %0.2f seconds" % t_mini_batch)
mbk_means_labels_unique = np.unique(mbk.labels_)

ax = fig.add_subplot(1, 3, 3)
for this_centroid, k, col in zip(mbk.cluster_centers_,
                                 range(n_clusters), colors_):
    mask = mbk.labels_ == k
    ax.scatter(X[mask, 0], X[mask, 1], marker='.',
               c='w', edgecolor=col, alpha=0.5)
    ax.scatter(this_centroid[0], this_centroid[1], marker='+',
               c='k', s=25)
ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])
ax.set_title("MiniBatchKMeans")
ax.set_autoscaley_on(False)
plt.show()
```
67 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
def _update_precisions(self, X, z):
        """Update the variational distributions for the precisions"""
        n_features = X.shape[1]
        if self.covariance_type == 'spherical':
            self.dof_ = 0.5 * n_features * np.sum(z, axis=0)
            for k in range(self.n_components):
                # could be more memory efficient ?
                sq_diff = np.sum((X - self.means_[k]) ** 2, axis=1)
                self.scale_[k] = 1.
                self.scale_[k] += 0.5 * np.sum(z.T[k] * (sq_diff + n_features))
                self.bound_prec_[k] = (
                    0.5 * n_features * (
                        digamma(self.dof_[k]) - np.log(self.scale_[k])))
            self.precs_ = np.tile(self.dof_ / self.scale_, [n_features, 1]).T

        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                self.dof_[k].fill(1. + 0.5 * np.sum(z.T[k], axis=0))
                sq_diff = (X - self.means_[k]) ** 2  # see comment above
                self.scale_[k] = np.ones(n_features) + 0.5 * np.dot(
                    z.T[k], (sq_diff + 1))
                self.precs_[k] = self.dof_[k] / self.scale_[k]
                self.bound_prec_[k] = 0.5 * np.sum(digamma(self.dof_[k])
                                                   - np.log(self.scale_[k]))
                self.bound_prec_[k] -= 0.5 * np.sum(self.precs_[k])

        elif self.covariance_type == 'tied':
            self.dof_ = 2 + X.shape[0] + n_features
            self.scale_ = (X.shape[0] + 1) * np.identity(n_features)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.scale_ += np.dot(diff.T, z[:, k:k + 1] * diff)
            self.scale_ = pinvh(self.scale_)
            self.precs_ = self.dof_ * self.scale_
            self.det_scale_ = linalg.det(self.scale_)
            self.bound_prec_ = 0.5 * wishart_log_det(
                self.dof_, self.scale_, self.det_scale_, n_features)
            self.bound_prec_ -= 0.5 * self.dof_ * np.trace(self.scale_)

        elif self.covariance_type == 'full':
            for k in range(self.n_components):
                sum_resp = np.sum(z.T[k])
                self.dof_[k] = 2 + sum_resp + n_features
                self.scale_[k] = (sum_resp + 1) * np.identity(n_features)
                diff = X - self.means_[k]
                self.scale_[k] += np.dot(diff.T, z[:, k:k + 1] * diff)
                self.scale_[k] = pinvh(self.scale_[k])
                self.precs_[k] = self.dof_[k] * self.scale_[k]
                self.det_scale_[k] = linalg.det(self.scale_[k])
                self.bound_prec_[k] = 0.5 * wishart_log_det(
                    self.dof_[k], self.scale_[k], self.det_scale_[k],
                    n_features)
                self.bound_prec_[k] -= 0.5 * self.dof_[k] * np.trace(
                    self.scale_[k])

    def _monitor(self, X, z, n, end=False):
        """Monitor the lower bound during iteration

        Debug method to help see exactly when it is failing to converge as
        expected.

        Note: this is very expensive and should not be used by default."""
        if self.verbose > 0:
            print("Bound after updating %8s: %f" % (n, self.lower_bound(X, z)))
            if end:
                print("Cluster proportions:", self.gamma_.T[1])
                print("covariance_type:", self.covariance_type)

    
```
68 - /tmp/repos/scikit-learn/examples/cluster/plot_kmeans_stability_low_dim_dense.py
```python


# Part 1: Quantitative evaluation of various init methods

plt.figure()
plots = []
legends = []

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
    (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
    (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))

    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                         n_init=n_init, **params).fit(X)
            inertia[i, run_id] = km.inertia_
    p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel('n_init')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

# Part 2: Qualitative visual inspection of the convergence

X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
km = MiniBatchKMeans(n_clusters=n_clusters, init='random', n_init=1,
                     random_state=random_state).fit(X)

plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0], X[my_members, 1], 'o', marker='.', c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(cluster_center[0], cluster_center[1], 'o',
             markerfacecolor=color, markeredgecolor='k', markersize=6)
    plt.title("Example cluster allocation with a single random init\n"
              "with MiniBatchKMeans")

plt.show()

```
69 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def build_clfs(cd_iters, pg_iters, mu_iters):
    clfs = [("Coordinate Descent", NMF, cd_iters, {'solver': 'cd'}),
            ("Projected Gradient", _PGNMF, pg_iters, {'solver': 'pg'}),
            ("Multiplicative Update", NMF, mu_iters, {'solver': 'mu'}),
            ]
    return clfs


if __name__ == '__main__':
    alpha = 0.
    l1_ratio = 0.5
    n_components = 10
    tol = 1e-15

    # first benchmark on 20 newsgroup dataset: sparse, shape(11314, 39116)
    plot_name = "20 Newsgroups sparse dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 6)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_20news = load_20news()
    run_bench(X_20news, clfs, plot_name, n_components, tol, alpha, l1_ratio)

    # second benchmark on Olivetti faces dataset: dense, shape(400, 4096)
    plot_name = "Olivetti Faces dense dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 12)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_faces = load_faces()
    run_bench(X_faces, clfs, plot_name, n_components, tol, alpha, l1_ratio,)

    plt.show()

```
70 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError("Intercept scaling is %r but needs to be greater than 0."
                             " To disable fitting an intercept,"
                             " set fit_intercept=False." % intercept_scaling)
        else:
            bias = intercept_scaling

    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # Liblinear doesn't support 64bit sparse matrix indices yet
    if sp.issparse(X):
        _check_large_sparse(X)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    y_ind = np.require(y_ind, requirements="W")
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    else:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(sample_weight, X)

    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X, y_ind, sp.isspmatrix(X), solver_type, tol, bias, C,
        class_weight_, max_iter, rnd.randint(np.iinfo('i').max),
        epsilon, sample_weight)
    # Regarding rnd.randint(..) in the above signature:
    # seed for srand in range [0..INT_MAX); due to limitations in Numpy
    # on 32-bit platforms, we can't get to the UINT_MAX limit that
    # srand supports
    n_iter_ = max(n_iter_)
    if n_iter_ >= max_iter:
        warnings.warn("Liblinear failed to converge, increase "
                      "the number of iterations.", ConvergenceWarning)

    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.

    return coef_, intercept_, n_iter_

```
71 - /tmp/repos/scikit-learn/examples/decomposition/plot_ica_vs_pca.py
```python
"""
==========================
FastICA on 2D point clouds
==========================

This example illustrates visually in the feature space a comparison by
results using two different component analysis techniques.

:ref:`ICA` vs :ref:`PCA`.

Representing ICA in the feature space gives the view of 'geometric ICA':
ICA is an algorithm that finds directions in the feature space
corresponding to projections with high non-Gaussianity. These directions
need not be orthogonal in the original feature space, but they are
orthogonal in the whitened feature space, in which all directions
correspond to the same variance.

PCA, on the other hand, finds orthogonal directions in the raw feature
space that correspond to directions accounting for maximum variance.

Here we simulate independent sources using a highly non-Gaussian
process, 2 student T with a low number of degrees of freedom (top left
figure). We mix them to create observations (top right figure).
In this raw observation space, directions identified by PCA are
represented by orange vectors. We represent the signal in the PCA space,
after whitening by the variance corresponding to the PCA vectors (lower
left). Running ICA corresponds to finding a rotation in this space to
identify the directions of largest non-Gaussianity (lower right).
"""
print(__doc__)

# Authors: Alexandre Gramfort, Gael Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA

# #############################################################################
# Generate sample data
rng = np.random.RandomState(42)
S = rng.standard_t(1.5, size=(20000, 2))
S[:, 0] *= 2.

# Mix data
A = np.array([[1, 1], [0, 2]])  # Mixing matrix

X = np.dot(S, A.T)  # Generate observations

pca = PCA()
S_pca_ = pca.fit(X).transform(X)

ica = FastICA(random_state=rng)
S_ica_ = ica.fit(X).transform(X)  # Estimate the sources

S_ica_ /= S_ica_.std(axis=0)


# #############################################################################
# Plot results

def plot_samples(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    if axis_list is not None:
        colors = ['orange', 'red']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color=color)

    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('y')

plt.figure()
plt.subplot(2, 2, 1)
plot_samples(S / S.std())
plt.title('True Independent Sources')

axis_list = [pca.components_.T, ica.mixing_]
plt.subplot(2, 2, 2)
plot_samples(X / np.std(X), axis_list=axis_list)
legend = plt.legend(['PCA', 'ICA'], loc='upper right')
legend.set_zorder(100)

plt.title('Observations')

plt.subplot(2, 2, 3)
plot_samples(S_pca_ / np.std(S_pca_, axis=0))
plt.title('PCA recovered signals')

plt.subplot(2, 2, 4)
plot_samples(S_ica_ / np.std(S_ica_))
plt.title('ICA recovered signals')

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
plt.show()

```
72 - /tmp/repos/scikit-learn/examples/cluster/plot_inductive_clustering.py
```python
"""
==============================================
Inductive Clustering
==============================================

Clustering can be expensive, especially when our dataset contains millions
of datapoints. Many clustering algorithms are not :term:`inductive` and so
cannot be directly applied to new data samples without recomputing the
clustering, which may be intractable. Instead, we can use clustering to then
learn an inductive model with a classifier, which has several benefits:

- it allows the clusters to scale and apply to new data
- unlike re-fitting the clusters to new samples, it makes sure the labelling
  procedure is consistent over time
- it allows us to use the inferential capabilities of the classifier to
  describe or explain the clusters

This example illustrates a generic implementation of a meta-estimator which
extends clustering by inducing a classifier from the cluster labels.
"""
# Authors: Chirag Nagpal
#          Christos Aridas
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import if_delegate_has_method


N_SAMPLES = 5000
RANDOM_STATE = 42


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def decision_function(self, X):
        return self.classifier_.decision_function(X)


def plot_scatter(X,  color, alpha=0.5):
    return plt.scatter(X[:, 0],
                       X[:, 1],
                       c=color,
                       alpha=alpha,
                       edgecolor='k')


# Generate some training data from clustering
X, y = make_blobs(n_samples=N_SAMPLES,
                  cluster_std=[1.0, 1.0, 0.5],
                  centers=[(-5, -5), (0, 0), (5, 5)],
                  random_state=RANDOM_STATE)


# Train a clustering algorithm on the training data and get the cluster labels
clusterer = AgglomerativeClustering(n_clusters=3)
cluster_labels = clusterer.fit_predict(X)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plot_scatter(X, cluster_labels)
plt.title("Ward Linkage")


# Generate new samples and plot them along with the original dataset
X_new, y_new = make_blobs(n_samples=10,
                          centers=[(-7, -1), (-2, 4), (3, 6)],
                          random_state=RANDOM_STATE)

plt.subplot(132)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, 'black', 1)
plt.title("Unknown instances")


# Declare the inductive learning model that it will be used to
# predict cluster membership for unknown instances
classifier = RandomForestClassifier(random_state=RANDOM_STATE)
inductive_learner = InductiveClusterer(clusterer, classifier).fit(X)

probable_clusters = inductive_learner.predict(X_new)


plt.subplot(133)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, probable_clusters)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = inductive_learner.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.title("Classify unknown instances")

plt.show()

```
73 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def _fit_projected_gradient(X, W, H, tol, max_iter, nls_max_iter, alpha,
                            l1_ratio):
    gradW = (np.dot(W, np.dot(H, H.T)) -
             safe_sparse_dot(X, H.T, dense_output=True))
    gradH = (np.dot(np.dot(W.T, W), H) -
             safe_sparse_dot(W.T, X, dense_output=True))

    init_grad = squared_norm(gradW) + squared_norm(gradH.T)
    # max(0.001, tol) to force alternating minimizations of W and H
    tolW = max(0.001, tol) * np.sqrt(init_grad)
    tolH = tolW

    for n_iter in range(1, max_iter + 1):
        # stopping condition as discussed in paper
        proj_grad_W = squared_norm(gradW * np.logical_or(gradW < 0, W > 0))
        proj_grad_H = squared_norm(gradH * np.logical_or(gradH < 0, H > 0))

        if (proj_grad_W + proj_grad_H) / init_grad < tol ** 2:
            break

        # update W
        Wt, gradWt, iterW = _nls_subproblem(X.T, H.T, W.T, tolW, nls_max_iter,
                                            alpha=alpha, l1_ratio=l1_ratio)
        W, gradW = Wt.T, gradWt.T

        if iterW == 1:
            tolW = 0.1 * tolW

        # update H
        H, gradH, iterH = _nls_subproblem(X, W, H, tolH, nls_max_iter,
                                          alpha=alpha, l1_ratio=l1_ratio)
        if iterH == 1:
            tolH = 0.1 * tolH

    H[H == 0] = 0   # fix up negative zeros

    if n_iter == max_iter:
        Wt, _, _ = _nls_subproblem(X.T, H.T, W.T, tolW, nls_max_iter,
                                   alpha=alpha, l1_ratio=l1_ratio)
        W = Wt.T

    return W, H, n_iter



```
74 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators"""
    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]

    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)

    # some estimators can't do features less than 0
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)

    estimator = clone(estimator_orig)

    # some estimators only take multioutputs
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)
    estimator.fit(X, y)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    if estimator.__module__.startswith('sklearn.'):
        assert_true(b"version" in pickled_estimator)
    unpickled_estimator = pickle.loads(pickled_estimator)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # check if number of features changes between calls to partial_fit.
    if not hasattr(estimator_orig, 'partial_fit'):
        return
    estimator = clone(estimator_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X -= X.min()

    try:
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return

    with assert_raises(ValueError,
                       msg="The estimator {} does not raise an"
                           " error when the number of features"
                           " changes between calls to "
                           "partial_fit.".format(name)):
        estimator.partial_fit(X[:, :-1], y)



```
75 - /tmp/repos/scikit-learn/examples/preprocessing/plot_scaling_importance.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=========================================================
Importance of Feature Scaling
=========================================================

Feature scaling through standardization (or Z-score normalization)
can be an important preprocessing step for many machine learning
algorithms. Standardization involves rescaling the features such
that they have the properties of a standard normal distribution
with a mean of zero and a standard deviation of one.

While many algorithms (such as SVM, K-nearest neighbors, and logistic
regression) require features to be normalized, intuitively we can
think of Principle Component Analysis (PCA) as being a prime example
of when normalization is important. In PCA we are interested in the
components that maximize the variance. If one component (e.g. human
height) varies less than another (e.g. weight) because of their
respective scales (meters vs. kilos), PCA might determine that the
direction of maximal variance more closely corresponds with the
'weight' axis, if those features are not scaled. As a change in
height of one meter can be considered much more important than the
change in weight of one kilogram, this is clearly incorrect.

To illustrate this, PCA is performed comparing the use of data with
:class:`StandardScaler <sklearn.preprocessing.StandardScaler>` applied,
to unscaled data. The results are visualized and a clear difference noted.
The 1st principal component in the unscaled set can be seen. It can be seen
that feature #13 dominates the direction, being a whole two orders of
magnitude above the other features. This is contrasted when observing
the principal component for the scaled version of the data. In the scaled
version, the orders of magnitude are roughly the same across all the features.

The dataset used is the Wine Dataset available at UCI. This dataset
has continuous features that are heterogeneous in scale due to differing
properties that they measure (i.e alcohol content, and malic acid).

The transformed data is then used to train a naive Bayes classifier, and a
clear difference in prediction accuracies is observed wherein the dataset
which is scaled before PCA vastly outperforms the unscaled version.

"""
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

# Code source: Tyler Lanigan <tylerlanigan@gmail.com>
#              Sebastian Raschka <mail@sebastianraschka.com>

# License: BSD 3 clause

RANDOM_STATE = 42
FIG_SIZE = (10, 7)


features, target = load_wine(return_X_y=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)

# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal componenets
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Scale and use PCA on X_train data for visualization.
scaler = std_clf.named_steps['standardscaler']
X_train_std = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)



```
76 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
```python
"""Utilities to evaluate models with respect to a variable
"""
# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import warnings

import numpy as np

from .base import is_classifier, clone
from .cross_validation import check_cv
from .externals.joblib import Parallel, delayed
from .cross_validation import _safe_split, _score, _fit_and_score
from .metrics.scorer import check_scoring
from .utils import indexable


warnings.warn("This module was deprecated in version 0.18 in favor of the "
              "model_selection module into which all the functions are moved."
              " This module will be removed in 0.20",
              DeprecationWarning)


__all__ = ['learning_curve', 'validation_curve']


def learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
                   cv=None, scoring=None, exploit_incremental_learning=False,
                   n_jobs=1, pre_dispatch="all", verbose=0,
                   error_score='raise'):
    
```
77 - /tmp/repos/scikit-learn/examples/calibration/plot_calibration_multiclass.py
```python
plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
             xy=(.5, .0), xytext=(.5, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
             xy=(.0, .5), xytext=(.1, .5), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
             xy=(.5, .5), xytext=(.6, .6), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $0$, $1$)',
             xy=(0, 0), xytext=(.1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($1$, $0$, $0$)',
             xy=(1, 0), xytext=(1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $1$, $0$)',
             xy=(0, 1), xytext=(.1, 1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
# Add grid
plt.grid("off")
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Change of predicted probabilities after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(loc="best")

print("Log-loss of")
print(" * uncalibrated classifier trained on 800 datapoints: %.3f "
      % score)
print(" * classifier trained on 600 datapoints and calibrated on "
      "200 datapoint: %.3f" % sig_score)

# Illustrate calibrator
plt.figure(1)
# generate grid over 2-simplex
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

calibrated_classifier = sig_clf.calibrated_classifiers_[0]
prediction = np.vstack([calibrator.predict(this_p)
                        for calibrator, this_p in
                        zip(calibrated_classifier.calibrators_, p.T)]).T
prediction /= prediction.sum(axis=1)[:, None]

# Plot modifications of calibrator
for i in range(prediction.shape[0]):
    plt.arrow(p[i, 0], p[i, 1],
              prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
              head_width=1e-2, color=colors[np.argmax(p[i])])
# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

plt.grid("off")

```
78 - /tmp/repos/scikit-learn/sklearn/utils/__init__.py
```python
def is_scalar_nan(x):
    """Tests if x is NaN

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not np.float('nan').

    Parameters
    ----------
    x : any type

    Returns
    -------
    boolean

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    # convert from numpy.bool_ to python bool to ensure that testing
    # is_scalar_nan(x) is True does not fail.
    return bool(isinstance(x, numbers.Real) and np.isnan(x))


def _approximate_mode(class_counts, n_draws, rng):
    """Computes approximate mode of multivariate hypergeometric.

    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.

    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.

    Parameters
    ----------
    class_counts : ndarray of int
        Population per class.
    n_draws : int
        Number of draws (samples to draw) from the overall population.
    rng : random state
        Used to break ties.

    Returns
    -------
    sampled_classes : ndarray of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils import _approximate_mode
    >>> _approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
    array([2, 1])
    >>> _approximate_mode(class_counts=np.array([5, 2]), n_draws=4, rng=0)
    array([3, 1])
    >>> _approximate_mode(class_counts=np.array([2, 2, 2, 1]),
    ...                   n_draws=2, rng=0)
    array([0, 1, 1, 0])
    >>> _approximate_mode(class_counts=np.array([2, 2, 2, 1]),
    ...                   n_draws=2, rng=42)
    array([1, 1, 0, 0])
    """
    rng = check_random_state(rng)
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = n_draws * class_counts / class_counts.sum()
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = np.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            inds, = np.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int)


def check_matplotlib_support(caller_name):
    """Raise ImportError with detailed error message if mpl is not installed.

    Plot utilities like :func:`plot_partial_dependence` should lazily import
    matplotlib and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires matplotlib.
    """
    try:
        import matplotlib  # noqa
    except ImportError as e:
        raise ImportError(
            "{} requires matplotlib. You can install matplotlib with "
            "`pip install matplotlib`".format(caller_name)
        ) from e

```
79 - /tmp/repos/scikit-learn/examples/text/plot_document_classification_20newsgroups.py
```python



results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty,
                                           max_iter=5)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet",
                                       max_iter=5)))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

```
80 - /tmp/repos/scikit-learn/examples/model_selection/plot_nested_cross_validation_iris.py
```python
"""
=========================================
Nested versus non-nested cross-validation
=========================================

This example compares non-nested and nested cross-validation strategies on a
classifier of the iris data set. Nested cross-validation (CV) is often used to
train a model in which hyperparameters also need to be optimized. Nested CV
estimates the generalization error of the underlying model and its
(hyper)parameter search. Choosing the parameters that maximize non-nested CV
biases the model to the dataset, yielding an overly-optimistic score.

Model selection without nested CV uses the same data to tune model parameters
and evaluate model performance. Information may thus "leak" into the model
and overfit the data. The magnitude of this effect is primarily dependent on
the size of the dataset and the stability of the model. See Cawley and Talbot
[1]_ for an analysis of these issues.

To avoid this problem, nested CV effectively uses a series of
train/validation/test set splits. In the inner loop (here executed by
:class:`GridSearchCV <sklearn.model_selection.GridSearchCV>`), the score is
approximately maximized by fitting a model to each training set, and then
directly maximized in selecting (hyper)parameters over the validation set. In
the outer loop (here in :func:`cross_val_score
<sklearn.model_selection.cross_val_score>`), generalization error is estimated
by averaging test set scores over several dataset splits.

The example below uses a support vector classifier with a non-linear kernel to
build a model with optimized hyperparameters by grid search. We compare the
performance of non-nested and nested CV strategies by taking the difference
between their scores.

.. topic:: See Also:

    - :ref:`cross_validation`
    - :ref:`grid_search`

.. topic:: References:

    .. [1] `Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and
     subsequent selection bias in performance evaluation.
     J. Mach. Learn. Res 2010,11, 2079-2107.
     <http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf>`_

"""
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np

print(__doc__)

# Number of random trials
NUM_TRIALS = 30

# Load the dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf")

# Arrays to store scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
    clf.fit(X_iris, y_iris)
    non_nested_scores[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores

print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(score_difference.mean(), score_difference.std()))

# Plot scores on each trial for nested and non-nested CV
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
nested_line, = plt.plot(nested_scores, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation on Iris Dataset",
          x=.5, y=1.1, fontsize="15")


```
81 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        if self.min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated and"
                          " will be removed in version 0.21. "
                          "Use the min_impurity_decrease parameter instead.",
                          DeprecationWarning)
            min_impurity_split = self.min_impurity_split
        else:
            min_impurity_split = 1e-7

        if min_impurity_split < 0.:
            raise ValueError("min_impurity_split must be greater than "
                             "or equal to 0")

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        allowed_presort = ('auto', True, False)
        if self.presort not in allowed_presort:
            raise ValueError("'presort' should be in {}. Got {!r} instead."
                             .format(allowed_presort, self.presort))

        if self.presort is True and issparse(X):
            raise ValueError("Presorting is not supported for sparse "
                             "matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto':
            presort = not issparse(X)

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        
```
82 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _check_transformer(name, transformer_orig, X, y):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # on numpy & scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[y, y]
        y_[::2, 1] *= 2
    else:
        y_ = y

    transformer.fit(X, y_)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y_)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert_equal(x_pred.shape[0], n_samples)
    else:
        # check for consistent n_samples
        assert_equal(X_pred.shape[0], n_samples)

    if hasattr(transformer, 'transform'):
        if name in CROSS_DECOMPOSITION:
            X_pred2 = transformer.transform(X, y_)
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            X_pred2 = transformer.transform(X)
            X_pred3 = transformer.fit_transform(X, y=y_)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred, x_pred2, atol=1e-2,
                    err_msg="fit_transform and transform outcomes "
                            "not consistent in %s"
                    % transformer)
                assert_allclose_dense_sparse(
                    x_pred, x_pred3, atol=1e-2,
                    err_msg="consecutive fit_transform outcomes "
                            "not consistent in %s"
                    % transformer)
        else:
            assert_allclose_dense_sparse(
                X_pred, X_pred2,
                err_msg="fit_transform and transform outcomes "
                        "not consistent in %s"
                % transformer, atol=1e-2)
            assert_allclose_dense_sparse(
                X_pred, X_pred3, atol=1e-2,
                err_msg="consecutive fit_transform outcomes "
                        "not consistent in %s"
                % transformer)
            assert_equal(_num_samples(X_pred2), n_samples)
            assert_equal(_num_samples(X_pred3), n_samples)

        # raises error on malformed input for transform
        if hasattr(X, 'T'):
            # If it's not an array, it does not have a 'T' property
            with assert_raises(ValueError, msg="The transformer {} does "
                               "not raise an error when the number of "
                               "features in transform is different from"
                               " the number of features in "
                               "fit.".format(name)):
                transformer.transform(X.T)
```
83 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values."""
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


@deprecated("The function distribute_covar_matrix_to_match_covariance_type"
            "is deprecated in 0.18 and will be removed in 0.20.")
def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template."""
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv


def _covar_mstep_diag(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Perform the covariance M step for diagonal cases."""
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = gmm.means_ ** 2
    avg_X_means = gmm.means_ * weighted_X_sum * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _covar_mstep_spherical(*args):
    """Perform the covariance M step for spherical cases."""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))


def _covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Perform the covariance M step for full cases."""
    # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        with np.errstate(under='ignore'):
            # Underflow Errors in doing post * X.T are  not important
            avg_cv = np.dot(post * diff.T, diff) / (post.sum() + 10 * EPS)
        cv[c] = avg_cv + min_covar * np.eye(n_features)
    return cv



```
84 - /tmp/repos/scikit-learn/sklearn/metrics/classification.py
```python
with np.errstate(divide='ignore', invalid='ignore'):
        # Divide, and on zero-division, set scores to 0 and warn:

        # Oddly, we may get an "invalid" rather than a "divide" error
        # here.
        precision = _prf_divide(tp_sum, pred_sum,
                                'precision', 'predicted', average, warn_for)
        recall = _prf_divide(tp_sum, true_sum,
                             'recall', 'true', average, warn_for)
        # Don't need to warn for F: either P or R warned, or tp == 0 where pos
        # and true are nonzero, in which case, F is well-defined and zero
        f_score = ((1 + beta2) * precision * recall /
                   (beta2 * precision + recall))
        f_score[tp_sum == 0] = 0.0

    # Average the results

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum



```
85 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError("Intercept scaling is %r but needs to be greater than 0."
                             " To disable fitting an intercept,"
                             " set fit_intercept=False." % intercept_scaling)
        else:
            bias = intercept_scaling

    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    else:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(sample_weight, X)

    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X, y_ind, sp.isspmatrix(X), solver_type, tol, bias, C,
        class_weight_, max_iter, rnd.randint(np.iinfo('i').max),
        epsilon, sample_weight)
    # Regarding rnd.randint(..) in the above signature:
    # seed for srand in range [0..INT_MAX); due to limitations in Numpy
    # on 32-bit platforms, we can't get to the UINT_MAX limit that
    # srand supports
    n_iter_ = max(n_iter_)
    if n_iter_ >= max_iter and verbose > 0:
        warnings.warn("Liblinear failed to converge, increase "
                      "the number of iterations.", ConvergenceWarning)

    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.

    return coef_, intercept_, n_iter_

```
86 - /tmp/repos/scikit-learn/examples/linear_model/plot_lasso_model_selection.py
```python


plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)

# #############################################################################
# LassoCV: coordinate descent

# Compute paths
print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = LassoCV(cv=20).fit(X, y)
t_lasso_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.alphas_)

plt.figure()
ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

# #############################################################################
# LassoLarsCV: least angle regression

# Compute paths
print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv=20).fit(X, y)
t_lasso_lars_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

plt.show()

```
87 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
"""
Benchmarks of Non-Negative Matrix Factorization
"""
# Authors: Tom Dupre la Tour (benchmark)
#          Chih-Jen Linn (original projected gradient NMF implementation)
#          Anthony Di Franco (projected gradient, Python and NumPy port)
# License: BSD 3 clause

from __future__ import print_function
from time import time
import sys
import warnings
import numbers

import numpy as np
import matplotlib.pyplot as plt
import pandas

from sklearn.utils.testing import ignore_warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.nmf import NMF
from sklearn.decomposition.nmf import _initialize_nmf
from sklearn.decomposition.nmf import _beta_divergence
from sklearn.decomposition.nmf import INTEGER_TYPES, _check_init
from sklearn.externals.joblib import Memory
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_non_negative


mem = Memory(cachedir='.', verbose=0)

###################
# Start of _PGNMF #
###################
# This class implements a projected gradient solver for the NMF.
# The projected gradient solver was removed from scikit-learn in version 0.19,
# and a simplified copy is used here for comparison purpose only.
# It is not tested, and it may change or disappear without notice.


def _norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return np.sqrt(squared_norm(x))



```
88 - /tmp/repos/scikit-learn/benchmarks/bench_lof.py
```python
"""
============================
LocalOutlierFactor benchmark
============================

A test of LocalOutlierFactor on classical anomaly detection datasets.

Note that LocalOutlierFactor is not meant to predict on a test set and its
performance is assessed in an outlier detection context:
1. The model is trained on the whole dataset which is assumed to contain
outliers.
2. The ROC curve is computed on the same dataset using the knowledge of the
labels.
In this context there is no need to shuffle the dataset because the model
is trained and tested on the whole dataset. The randomness of this benchmark
is only caused by the random selection of anomalies in the SA dataset.

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.preprocessing import LabelBinarizer

print(__doc__)

random_state = 2  # to control the random selection of anomalies in SA

# datasets available: ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']

plt.figure()
for dataset_name in datasets:
    # loading and vectorization
    print('loading data')
    if dataset_name in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dataset_name, percent10=True,
                                 random_state=random_state)
        X = dataset.data
        y = dataset.target

    if dataset_name == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dataset_name == 'forestcover':
        dataset = fetch_covtype()
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dataset_name == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'http' or dataset_name == 'smtp':
        y = (y != b'normal.').astype(int)

    X = X.astype(float)

    print('LocalOutlierFactor processing...')
    model = LocalOutlierFactor(n_neighbors=20)
    tstart = time()
    model.fit(X)
    fit_time = time() - tstart
    scoring = -model.negative_outlier_factor_  # the lower, the more normal
    fpr, tpr, thresholds = roc_curve(y, scoring)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1,
             label=('ROC for %s (area = %0.3f, train-time: %0.2fs)'
                    % (dataset_name, AUC, fit_time)))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
```
89 - /tmp/repos/scikit-learn/sklearn/preprocessing/_discretization.py
```python
# -*- coding: utf-8 -*-

# Author: Henry Lin <hlin117@gmail.com>
#         Tom Dupré la Tour

# License: BSD


import numbers
import numpy as np
import warnings

from . import OneHotEncoder

from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES


class KBinsDiscretizer(BaseEstimator, TransformerMixin):
    
```
90 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_covariances.py
```python
"""
===============
GMM covariances
===============

Demonstration of several covariances types for Gaussian mixture models.

See :ref:`gmm` for more information on the estimator.

Although GMM are often used for clustering, we can compare the obtained
clusters with the actual classes from the dataset. We initialize the means
of the Gaussians with the means of the classes from the training set to make
this comparison valid.

We plot predicted labels on both training and held out test data using a
variety of GMM covariance types on the iris dataset.
We compare GMMs with spherical, diagonal, full, and tied covariance
matrices in increasing order of performance. Although one would
expect full covariance to perform best in general, it is prone to
overfitting on small datasets and does not generalize well to held out
test data.

On the plots, train data is shown as dots, while test data is shown as
crosses. The iris dataset is four-dimensional. Only the first two
dimensions are shown here, and thus some points are separated in other
dimensions.
"""

# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

print(__doc__)

colors = ['navy', 'turquoise', 'darkorange']


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
```
91 - /tmp/repos/scikit-learn/examples/calibration/plot_compare_calibration.py
```python
"""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of the predict_proba method can be directly interpreted as a confidence level.
For instance a well calibrated (binary) classifier should classify the samples
such that among the samples to which it gave a predict_proba value close to
0.8, approx. 80% actually belong to the positive class.

LogisticRegression returns well calibrated predictions as it directly
optimizes log-loss. In contrast, the other methods return biased probabilities,
with different biases per method:

* GaussianNaiveBayes tends to push probabilities to 0 or 1 (note the counts in
  the histograms). This is mainly because it makes the assumption that features
  are conditionally independent given the class, which is not the case in this
  dataset which contains 2 redundant features.

* RandomForestClassifier shows the opposite behavior: the histograms show
  peaks at approx. 0.2 and 0.9 probability, while probabilities close to 0 or 1
  are very rare. An explanation for this is given by Niculescu-Mizil and Caruana
  [1]: "Methods such as bagging and random forests that average predictions from
  a base set of models can have difficulty making predictions near 0 and 1
  because variance in the underlying base models will bias predictions that
  should be near zero or one away from these values. Because predictions are
  restricted to the interval [0,1], errors caused by variance tend to be one-
  sided near zero and one. For example, if a model should predict p = 0 for a
  case, the only way bagging can achieve this is if all bagged trees predict
  zero. If we add noise to the trees that bagging is averaging over, this noise
  will cause some trees to predict values larger than 0 for this case, thus
  moving the average prediction of the bagged ensemble away from 0. We observe
  this effect most strongly with random forests because the base-level trees
  trained with random forests have relatively high variance due to feature
  subsetting." As a result, the calibration curve shows a characteristic
  sigmoid shape, indicating that the classifier could trust its "intuition"
  more and return probabilities closer to 0 or 1 typically.

* Support Vector Classification (SVC) shows an even more sigmoid curve as
  the  RandomForestClassifier, which is typical for maximum-margin methods
  (compare Niculescu-Mizil and Caruana [1]), which focus on hard samples
  that are close to the decision boundary (the support vectors).

.. topic:: References:

    .. [1] Predicting Good Probabilities with Supervised Learning,
          A. Niculescu-Mizil & R. Caruana, ICML 2005
"""
print(__doc__)

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train = X[:train_samples]
X_test = X[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

```
92 - /tmp/repos/scikit-learn/examples/cluster/plot_agglomerative_clustering_metrics.py
```python


X = list()
y = list()
for i, (phi, a) in enumerate([(.5, .15), (.5, .6), (.3, .2)]):
    for _ in range(30):
        phase_noise = .01 * np.random.normal()
        amplitude_noise = .04 * np.random.normal()
        additional_noise = 1 - 2 * np.random.rand(n_features)
        # Make the noise sparse
        additional_noise[np.abs(additional_noise) < .997] = 0

        X.append(12 * ((a + amplitude_noise)
                 * (sqr(6 * (t + phi + phase_noise)))
                 + additional_noise))
        y.append(i)

X = np.array(X)
y = np.array(y)

n_clusters = 3

labels = ('Waveform 1', 'Waveform 2', 'Waveform 3')

# Plot the ground-truth labelling
plt.figure()
plt.axes([0, 0, 1, 1])
for l, c, n in zip(range(n_clusters), 'rgb',
                   labels):
    lines = plt.plot(X[y == l].T, c=c, alpha=.5)
    lines[0].set_label(n)

plt.legend(loc='best')

plt.axis('tight')
plt.axis('off')
plt.suptitle("Ground truth", size=20)


# Plot the distances
for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
    avg_dist = np.zeros((n_clusters, n_clusters))
    plt.figure(figsize=(5, 4.5))
    for i in range(n_clusters):
        for j in range(n_clusters):
            avg_dist[i, j] = pairwise_distances(X[y == i], X[y == j],
                                                metric=metric).mean()
    avg_dist /= avg_dist.max()
    for i in range(n_clusters):
        for j in range(n_clusters):
            plt.text(i, j, '%5.3f' % avg_dist[i, j],
                     verticalalignment='center',
                     horizontalalignment='center')

    plt.imshow(avg_dist, interpolation='nearest', cmap=plt.cm.gnuplot2,
               vmin=0)
    plt.xticks(range(n_clusters), labels, rotation=45)
    plt.yticks(range(n_clusters), labels)
    plt.colorbar()
    plt.suptitle("Interclass %s distances" % metric, size=18)
    plt.tight_layout()


# Plot clustering results
for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage="average", affinity=metric)
    model.fit(X)
    plt.figure()
    plt.axes([0, 0, 1, 1])
    for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
        plt.plot(X[model.labels_ == l].T, c=c, alpha=.5)
    plt.axis('tight')
    plt.axis('off')
    plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)


plt.show()

```
93 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
if not (self._is_fitted() and self.warm_start):
            # Clear random state and score attributes
            self._clear_state()

            # initialize raw_predictions: those are the accumulated values
            # predicted by the trees for the training data. raw_predictions has
            # shape (n_trees_per_iteration, n_samples) where
            # n_trees_per_iterations is n_classes in multiclass classification,
            # else 1.
            self._baseline_prediction = self.loss_.get_baseline_prediction(
                y_train, self.n_trees_per_iteration_
            )
            raw_predictions = np.zeros(
                shape=(self.n_trees_per_iteration_, n_samples),
                dtype=self._baseline_prediction.dtype
            )
            raw_predictions += self._baseline_prediction

            # initialize gradients and hessians (empty arrays).
            # shape = (n_trees_per_iteration, n_samples).
            gradients, hessians = self.loss_.init_gradients_and_hessians(
                n_samples=n_samples,
                prediction_dim=self.n_trees_per_iteration_
            )

            # predictors is a matrix (list of lists) of TreePredictor objects
            # with shape (n_iter_, n_trees_per_iteration)
            self._predictors = predictors = []

            # Initialize structures and attributes related to early stopping
            self.scorer_ = None  # set if scoring != loss
            raw_predictions_val = None  # set if scoring == loss and use val
            self.train_score_ = []
            self.validation_score_ = []

            if self.do_early_stopping_:
                # populate train_score and validation_score with the
                # predictions of the initial model (before the first tree)

                if self.scoring == 'loss':
                    # we're going to compute scoring w.r.t the loss. As losses
                    # take raw predictions as input (unlike the scorers), we
                    # can optimize a bit and avoid repeating computing the
                    # predictions of the previous trees. We'll re-use
                    # raw_predictions (as it's needed for training anyway) for
                    # evaluating the training loss, and create
                    # raw_predictions_val for storing the raw predictions of
                    # the validation data.

                    if self._use_validation_data:
                        raw_predictions_val = np.zeros(
                            shape=(self.n_trees_per_iteration_,
                                   X_binned_val.shape[0]),
                            dtype=self._baseline_prediction.dtype
                        )

                        raw_predictions_val += self._baseline_prediction

                    self._check_early_stopping_loss(raw_predictions, y_train,
                                                    raw_predictions_val, y_val)
                else:
                    self.scorer_ = check_scoring(self, self.scoring)
                    # scorer_ is a callable with signature (est, X, y) and
                    # calls est.predict() or est.predict_proba() depending on
                    # its nature.
                    # Unfortunately, each call to scorer_() will compute
                    # the predictions of all the trees. So we use a subset of
                    # the training set to compute train scores.

                    # Save the seed for the small trainset generator
                    self._small_trainset_seed = rng.randint(1024)

                    # Compute the subsample set
                    (X_binned_small_train,
                     y_small_train) = self._get_small_trainset(
                        X_binned_train, y_train, self._small_trainset_seed)

                    self._check_early_stopping_scorer(
                        X_binned_small_train, y_small_train,
                        X_binned_val, y_val,
                    )
            begin_at_stage = 0

        # warm start: this is not the first time fit was called
        
```
94 - /tmp/repos/scikit-learn/sklearn/model_selection/__init__.py
```python
from ._split import BaseCrossValidator
from ._split import KFold
from ._split import GroupKFold
from ._split import StratifiedKFold
from ._split import TimeSeriesSplit
from ._split import LeaveOneGroupOut
from ._split import LeaveOneOut
from ._split import LeavePGroupsOut
from ._split import LeavePOut
from ._split import RepeatedKFold
from ._split import RepeatedStratifiedKFold
from ._split import ShuffleSplit
from ._split import GroupShuffleSplit
from ._split import StratifiedShuffleSplit
from ._split import PredefinedSplit
from ._split import train_test_split
from ._split import check_cv

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score
from ._validation import validation_curve

from ._search import GridSearchCV
from ._search import RandomizedSearchCV
from ._search import ParameterGrid
from ._search import ParameterSampler
from ._search import fit_grid_point

__all__ = ('BaseCrossValidator',
           'GridSearchCV',
           'TimeSeriesSplit',
           'KFold',
           'GroupKFold',
           'GroupShuffleSplit',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedKFold',
           'RepeatedStratifiedKFold',
           'ParameterGrid',
           'ParameterSampler',
           'PredefinedSplit',
           'RandomizedSearchCV',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'check_cv',
           'cross_val_predict',
           'cross_val_score',
           'cross_validate',
           'fit_grid_point',
           'learning_curve',
           'permutation_test_score',
           'train_test_split',
           'validation_curve')
```
95 - /tmp/repos/scikit-learn/examples/model_selection/plot_learning_curve.py
```python
axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

digits = load_digits()
X, y = digits.data, digits.target


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()

```
96 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python

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
```
97 - /tmp/repos/scikit-learn/examples/svm/plot_oneclass.py
```python
"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
```
98 - /tmp/repos/scikit-learn/examples/classification/plot_classifier_comparison.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets

```
99 - /tmp/repos/scikit-learn/examples/ensemble/plot_gradient_boosting_early_stopping.py
```python
"""
===================================
Early stopping of Gradient Boosting
===================================

Gradient boosting is an ensembling technique where several weak learners
(regression trees) are combined to yield a powerful single model, in an
iterative fashion.

Early stopping support in Gradient Boosting enables us to find the least number
of iterations which is sufficient to build a model that generalizes well to
unseen data.

The concept of early stopping is simple. We specify a ``validation_fraction``
which denotes the fraction of the whole dataset that will be kept aside from
training to assess the validation loss of the model. The gradient boosting
model is trained using the training set and evaluated using the validation set.
When each additional stage of regression tree is added, the validation set is
used to score the model.  This is continued until the scores of the model in
the last ``n_iter_no_change`` stages do not improve by atleast `tol`. After
that the model is considered to have converged and further addition of stages
is "stopped early".

The number of stages of the final model is available at the attribute
``n_estimators_``.

This example illustrates how the early stopping can used in the
:class:`sklearn.ensemble.GradientBoostingClassifier` model to achieve
almost the same accuracy as compared to a model built without early stopping
using many fewer estimators. This can significantly reduce training time,
memory usage and prediction latency.
"""

# Authors: Vighnesh Birodkar <vighneshbirodkar@nyu.edu>
#          Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.model_selection import train_test_split

print(__doc__)

data_list = [datasets.load_iris(), datasets.load_digits()]
data_list = [(d.data, d.target) for d in data_list]
data_list += [datasets.make_hastie_10_2()]
names = ['Iris Data', 'Digits Data', 'Hastie Data']

n_gb = []
score_gb = []
time_gb = []
n_gbes = []
score_gbes = []
time_gbes = []

n_estimators = 500

for X, y in data_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    # We specify that if the scores don't improve by atleast 0.01 for the last
    # 10 stages, stop fitting additional stages
    gbes = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                               validation_fraction=0.2,
                                               n_iter_no_change=5, tol=0.01,
                                               random_state=0)
    gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                             random_state=0)
    start = time.time()
    gb.fit(X_train, y_train)
    time_gb.append(time.time() - start)

    start = time.time()
    gbes.fit(X_train, y_train)
    time_gbes.append(time.time() - start)

    score_gb.append(gb.score(X_test, y_test))
    score_gbes.append(gbes.score(X_test, y_test))

    n_gb.append(gb.n_estimators_)
    n_gbes.append(gbes.n_estimators_)

bar_width = 0.2
n = len(data_list)
index = np.arange(0, n * bar_width, bar_width) * 2.5
index = index[0:n]

#######################################################################
# Compare scores with and without early stopping
# ----------------------------------------------

plt.figure(figsize=(9, 5))

bar1 = plt.bar(index, score_gb, bar_width, label='Without early stopping',
               color='crimson')
bar2 = plt.bar(index + bar_width, score_gbes, bar_width,
               label='With early stopping', color='coral')

plt.xticks(index + bar_width, names)
plt.yticks(np.arange(0, 1.3, 0.1))


def autolabel(rects, n_estimators):
    """
    Attach a text label above each bar displaying n_estimators of each model
    """
    for i, rect in enumerate(rects):
        plt.text(rect.get_x() + rect.get_width() / 2.,
                 1.05 * rect.get_height(), 'n_est=%d' % n_estimators[i],
                 ha='center', va='bottom')
```
100 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError("An estimator must support the partial_fit interface "
                         "to exploit incremental learning")
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)

    if shuffle:
        rng = check_random_state(random_state)
        cv_iter = ((rng.permutation(train), test) for train, test in cv_iter)

    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel(delayed(_incremental_fit_estimator)(
            clone(estimator), X, y, classes, train, test, train_sizes_abs,
            scorer, verbose) for train, test in cv_iter)
    else:
        train_test_proportions = []
        for train, test in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))

        out = parallel(delayed(_fit_and_score)(
            clone(estimator), X, y, scorer, train, test,
            verbose, parameters=None, fit_params=None, return_train_score=True)
            for train, test in train_test_proportions)
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)

    out = np.asarray(out).transpose((2, 1, 0))

    return train_sizes_abs, out[0], out[1]



```
101 - /tmp/repos/scikit-learn/sklearn/covariance/robust_covariance.py
```python
if (n_samples > 500) and (n_features > 1):
        # 1. Find candidate supports on subsets
        # a. split the set in subsets of size ~ 300
        n_subsets = n_samples // 300
        n_samples_subsets = n_samples // n_subsets
        samples_shuffle = random_state.permutation(n_samples)
        h_subset = int(np.ceil(n_samples_subsets *
                       (n_support / float(n_samples))))
        # b. perform a total of 500 trials
        n_trials_tot = 500
        # c. select 10 best (location, covariance) for each subset
        n_best_sub = 10
        n_trials = max(10, n_trials_tot // n_subsets)
        n_best_tot = n_subsets * n_best_sub
        all_best_locations = np.zeros((n_best_tot, n_features))
        try:
            all_best_covariances = np.zeros((n_best_tot, n_features,
                                             n_features))
        except MemoryError:
            # The above is too big. Let's try with something much small
            # (and less optimal)
            all_best_covariances = np.zeros((n_best_tot, n_features,
                                             n_features))
            n_best_tot = 10
            n_best_sub = 2
        for i in range(n_subsets):
            low_bound = i * n_samples_subsets
            high_bound = low_bound + n_samples_subsets
            current_subset = X[samples_shuffle[low_bound:high_bound]]
            best_locations_sub, best_covariances_sub, _, _ = select_candidates(
                current_subset, h_subset, n_trials,
                select=n_best_sub, n_iter=2,
                cov_computation_method=cov_computation_method,
                random_state=random_state)
            subset_slice = np.arange(i * n_best_sub, (i + 1) * n_best_sub)
            all_best_locations[subset_slice] = best_locations_sub
            all_best_covariances[subset_slice] = best_covariances_sub
        # 2. Pool the candidate supports into a merged set
        # (possibly the full dataset)
        n_samples_merged = min(1500, n_samples)
        h_merged = int(np.ceil(n_samples_merged *
                       (n_support / float(n_samples))))
        if n_samples > 1500:
            n_best_merged = 10
        else:
            n_best_merged = 1
        # find the best couples (location, covariance) on the merged set
        selection = random_state.permutation(n_samples)[:n_samples_merged]
        locations_merged, covariances_merged, supports_merged, d = \
            select_candidates(
                X[selection], h_merged,
                n_trials=(all_best_locations, all_best_covariances),
                select=n_best_merged,
                cov_computation_method=cov_computation_method,
                random_state=random_state)
        # 3. Finally get the overall best (locations, covariance) couple
        if n_samples < 1500:
            # directly get the best couple (location, covariance)
            location = locations_merged[0]
            covariance = covariances_merged[0]
            support = np.zeros(n_samples, dtype=bool)
            dist = np.zeros(n_samples)
            support[selection] = supports_merged[0]
            dist[selection] = d[0]
        else:
            # select the best couple on the full dataset
            locations_full, covariances_full, supports_full, d = \
                select_candidates(
                    X, n_support,
                    n_trials=(locations_merged, covariances_merged),
                    select=1,
                    cov_computation_method=cov_computation_method,
                    random_state=random_state)
            location = locations_full[0]
            covariance = covariances_full[0]
            support = supports_full[0]
            dist = d[0]
    
```
102 - /tmp/repos/scikit-learn/benchmarks/bench_mnist.py
```python
ESTIMATORS = {
    "dummy": DummyClassifier(),
    'CART': DecisionTreeClassifier(),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'Nystroem-SVM': make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'SampledRBF-SVM': make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'LogisticRegression-SAG': LogisticRegression(solver='sag', tol=1e-1,
                                                 C=1e4),
    'LogisticRegression-SAGA': LogisticRegression(solver='saga', tol=1e-1,
                                                  C=1e4),
    'MultilayerPerceptron': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='sgd', learning_rate_init=0.2, momentum=0.9, verbose=1,
        tol=1e-4, random_state=1),
    'MLP-adam': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='adam', learning_rate_init=0.001, verbose=1,
        tol=1e-4, random_state=1)
}



```
103 - /tmp/repos/scikit-learn/examples/cluster/plot_cluster_iris.py
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()
```
104 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
"""Testing utilities."""

# Copyright (c) 2011, 2012
# Authors: Pietro Berkes,
#          Andreas Muller
#          Mathieu Blondel
#          Olivier Grisel
#          Arnaud Joly
#          Denis Engemann
#          Giorgio Patrini
#          Thierry Guillemot
# License: BSD 3 clause
import os
import inspect
import pkgutil
import warnings
import sys
import struct

import scipy as sp
import scipy.io
from functools import wraps
from operator import itemgetter
try:
    # Python 2
    from urllib2 import urlopen
    from urllib2 import HTTPError
except ImportError:
    # Python 3+
    from urllib.request import urlopen
    from urllib.error import HTTPError

import tempfile
import shutil
import os.path as op
import atexit
import unittest

# WindowsError only exist on Windows
try:
    WindowsError
except NameError:
    WindowsError = None

import sklearn
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.externals.funcsigs import signature
from sklearn.utils import deprecated

additional_names_in_all = []
try:
    from nose.tools import raises as _nose_raises
    deprecation_message = (
        'sklearn.utils.testing.raises has been deprecated in version 0.20 '
        'and will be removed in 0.22. Please use '
        'sklearn.utils.testing.assert_raises instead.')
    raises = deprecated(deprecation_message)(_nose_raises)
    additional_names_in_all.append('raises')
except ImportError:
    pass

try:
    from nose.tools import with_setup as _with_setup
    deprecation_message = (
        'sklearn.utils.testing.with_setup has been deprecated in version 0.20 '
        'and will be removed in 0.22.'
        'If your code relies on with_setup, please use'
        ' nose.tools.with_setup instead.')
    with_setup = deprecated(deprecation_message)(_with_setup)
    additional_names_in_all.append('with_setup')
except ImportError:
    pass

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less
from numpy.testing import assert_approx_equal
import numpy as np

from sklearn.base import (ClassifierMixin, RegressorMixin, TransformerMixin,
                          ClusterMixin)
from sklearn.utils._unittest_backport import TestCase

__all__ = ["assert_equal", "assert_not_equal", "assert_raises",
           "assert_raises_regexp", "assert_true",
           "assert_false", "assert_almost_equal", "assert_array_equal",
           "assert_array_almost_equal", "assert_array_less",
           "assert_less", "assert_less_equal",
           "assert_greater", "assert_greater_equal",
           "assert_approx_equal", "SkipTest"]
__all__.extend(additional_names_in_all)

_dummy = TestCase('__init__')
assert_equal = _dummy.assertEqual
assert_not_equal = _dummy.assertNotEqual
assert_true = _dummy.assertTrue
assert_false = _dummy.assertFalse
assert_raises = _dummy.assertRaises
SkipTest = unittest.case.SkipTest
assert_dict_equal = _dummy.assertDictEqual
assert_in = _dummy.assertIn
assert_not_in = _dummy.assertNotIn
assert_less = _dummy.assertLess
assert_greater = _dummy.assertGreater
assert_less_equal = _dummy.assertLessEqual
assert_greater_equal = _dummy.assertGreaterEqual

assert_raises_regex = _dummy.assertRaisesRegex
# assert_raises_regexp is deprecated in Python 3.4 in favor of
# assert_raises_regex but lets keep the backward compat in scikit-learn with
# the old name for now
assert_raises_regexp = assert_raises_regex



```
105 - /tmp/repos/scikit-learn/examples/cluster/plot_agglomerative_clustering_metrics.py
```python
"""
Agglomerative clustering with different metrics
===============================================

Demonstrates the effect of different metrics on the hierarchical clustering.

The example is engineered to show the effect of the choice of different
metrics. It is applied to waveforms, which can be seen as
high-dimensional vector. Indeed, the difference between metrics is
usually more pronounced in high dimension (in particular for euclidean
and cityblock).

We generate data from three groups of waveforms. Two of the waveforms
(waveform 1 and waveform 2) are proportional one to the other. The cosine
distance is invariant to a scaling of the data, as a result, it cannot
distinguish these two waveforms. Thus even with no noise, clustering
using this distance will not separate out waveform 1 and 2.

We add observation noise to these waveforms. We generate very sparse
noise: only 6% of the time points contain noise. As a result, the
l1 norm of this noise (ie "cityblock" distance) is much smaller than it's
l2 norm ("euclidean" distance). This can be seen on the inter-class
distance matrices: the values on the diagonal, that characterize the
spread of the class, are much bigger for the Euclidean distance than for
the cityblock distance.

When we apply clustering to the data, we find that the clustering
reflects what was in the distance matrices. Indeed, for the Euclidean
distance, the classes are ill-separated because of the noise, and thus
the clustering does not separate the waveforms. For the cityblock
distance, the separation is good and the waveform classes are recovered.
Finally, the cosine distance does not separate at all waveform 1 and 2,
thus the clustering puts them in the same cluster.
"""
# Author: Gael Varoquaux
# License: BSD 3-Clause or CC-0

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

np.random.seed(0)

# Generate waveform data
n_features = 2000
t = np.pi * np.linspace(0, 1, n_features)


def sqr(x):
    return np.sign(np.cos(x))
```
106 - /tmp/repos/scikit-learn/examples/calibration/plot_calibration.py
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
print(__doc__)

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = \
    train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)

# Gaussian Naive-Bayes with no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(X_train, y_train, sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, y_train, sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier scores: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

# #############################################################################
# Plot the data and the predicted probabilities
plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))

```
**107 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py**:
```python
"""Affinity Propagation clustering algorithm."""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause

import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
from ..base import BaseEstimator, ClusterMixin
from ..utils import as_float_array, check_array
from ..utils.validation import check_is_fitted
from ..metrics import euclidean_distances
from ..metrics import pairwise_distances_argmin


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    
```
108 - /tmp/repos/scikit-learn/examples/cluster/plot_adjusted_for_chance_measures.py
```python


score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    ami_score,
    metrics.mutual_info_score,
]

# 2 independent random clusterings with equal cluster number

n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(np.int)

plt.figure(1)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, np.median(scores, axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for 2 random uniform labelings\n"
          "with equal number of clusters")
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.legend(plots, names)
plt.ylim(bottom=-0.05, top=1.05)


# Random labeling with varying n_clusters against ground class labels
# with fixed number of clusters

n_samples = 1000
n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
n_classes = 10

plt.figure(2)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                                      fixed_n_classes=n_classes)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for random uniform labeling\n"
          "against reference assignment with %d classes" % n_classes)
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names)
plt.show()

```
**109 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
def predict(self, X):
        """
        Predict data using the ``centroids_`` of subclusters.

        Avoid computation of the row norms of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray, shape(n_samples)
            Labelled data.
        """
        X = check_array(X, accept_sparse='csr')
        self._check_fit(X)
        reduced_distance = safe_sparse_dot(X, self.subcluster_centers_.T)
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]

    def transform(self, X):
        """
        Transform X into subcluster centroids dimension.

        Each dimension represents the distance from the sample point to each
        cluster centroid.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_trans : {array-like, sparse matrix}, shape (n_samples, n_clusters)
            Transformed data.
        """
        check_is_fitted(self, 'subcluster_centers_')
        return euclidean_distances(X, self.subcluster_centers_)

    def _global_clustering(self, X=None):
        """
        Global clustering for the subclusters obtained after fitting
        """
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = (X is not None) and self.compute_labels

        # Preprocessing for the global clustering.
        not_enough_centroids = False
        if isinstance(clusterer, int):
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters)
            # There is no need to perform the global clustering step.
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        elif (clusterer is not None and not
              hasattr(clusterer, 'fit_predict')):
            raise ValueError("n_clusters should be an instance of "
                             "ClusterMixin or an int")

        # To use in predict to avoid recalculation.
        self._subcluster_norms = row_norms(
            self.subcluster_centers_, squared=True)

        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn(
                    "Number of subclusters found (%d) by Birch is less "
                    "than (%d). Decrease the threshold."
                    % (len(centroids), self.n_clusters), ConvergenceWarning)
        else:
            # The global clustering step that clusters the subclusters of
            # the leaves. It assumes the centroids of the subclusters as
            # samples and finds the final centroids.
            self.subcluster_labels_ = clusterer.fit_predict(
                self.subcluster_centers_)

        if compute_labels:
            self.labels_ = self.predict(X)

```
110 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
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
            return_times=True, return_estimator=return_estimator)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret



```
111 - /tmp/repos/scikit-learn/sklearn/cluster/spectral.py
```python
while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components))

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ((abs(ncut_value - last_objective_value) < eps) or
                    (n_iter > n_iter_max)):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')
    return labels


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    
```
112 - /tmp/repos/scikit-learn/sklearn/__init__.py
```python
"""
Machine learning module for Python
==================================

sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.org for complete documentation.
"""
import sys
import re
import warnings
import logging

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.20.dev0'


try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKLEARN_SETUP__
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write('Partial import of sklearn during the build process.\n')
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import __check_build
    from .base import clone
    __check_build  # avoid flakes unused variable error

    __all__ = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
               'cross_validation', 'datasets', 'decomposition', 'dummy',
               'ensemble', 'exceptions', 'externals', 'feature_extraction',
               'feature_selection', 'gaussian_process', 'grid_search',
               'isotonic', 'kernel_approximation', 'kernel_ridge',
               'learning_curve', 'linear_model', 'manifold', 'metrics',
               'mixture', 'model_selection', 'multiclass', 'multioutput',
               'naive_bayes', 'neighbors', 'neural_network', 'pipeline',
               'preprocessing', 'random_projection', 'semi_supervised',
               'svm', 'tree', 'discriminant_analysis', 'impute',
               # Non-modules:
               'clone', 'get_config', 'set_config', 'config_context']


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import numpy as np
    import random

    # It could have been provided in the environment
    _random_seed = os.environ.get('SKLEARN_SEED', None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2 ** 31 - 1)
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)

```
113 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
"""

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        super(BayesianGaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if (self.weight_concentration_prior_type not in
                ['dirichlet_process', 'dirichlet_distribution']):
            raise ValueError(
                "Invalid value for 'weight_concentration_prior_type': %s "
                "'weight_concentration_prior_type' should be in "
                "['dirichlet_process', 'dirichlet_distribution']"
                % self.weight_concentration_prior_type)

        self._check_weights_parameters()
        self._check_means_parameters(X)
        self._check_precision_parameters(X)
        self._checkcovariance_prior_parameter(X)

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1. / self.n_components
        elif self.weight_concentration_prior > 0.:
            self.weight_concentration_prior_ = (
                self.weight_concentration_prior)
        else:
            raise ValueError("The parameter 'weight_concentration_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.weight_concentration_prior)

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.
        elif self.mean_precision_prior > 0.:
            self.mean_precision_prior_ = self.mean_precision_prior
        else:
            raise ValueError("The parameter 'mean_precision_prior' should be "
                             "greater than 0., but got %.3f."
                             % self.mean_precision_prior)

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(self.mean_prior,
                                           dtype=[np.float64, np.float32],
                                           ensure_2d=False)
            _check_shape(self.mean_prior_, (n_features, ), 'means')

    def _check_precision_parameters(self, X):
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.degrees_of_freedom_prior > n_features - 1.:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            raise ValueError("The parameter 'degrees_of_freedom_prior' "
                             "should be greater than %d, but got %.3f."
                             % (n_features - 1, self.degrees_of_freedom_prior))

    
```
114 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
else:
            # Check that the maximum number of iterations is not smaller
            # than the number of iterations from the previous fit
            if self.max_iter < self.n_iter_:
                raise ValueError(
                    'max_iter=%d must be larger than or equal to '
                    'n_iter_=%d when warm_start==True'
                    % (self.max_iter, self.n_iter_)
                )

            # Convert array attributes to lists
            self.train_score_ = self.train_score_.tolist()
            self.validation_score_ = self.validation_score_.tolist()

            # Compute raw predictions
            raw_predictions = self._raw_predict(X_binned_train)

            if self.do_early_stopping_ and self.scoring != 'loss':
                # Compute the subsample set
                X_binned_small_train, y_small_train = self._get_small_trainset(
                    X_binned_train, y_train, self._small_trainset_seed)

            # Initialize the gradients and hessians
            gradients, hessians = self.loss_.init_gradients_and_hessians(
                n_samples=n_samples,
                prediction_dim=self.n_trees_per_iteration_
            )

            # Get the predictors from the previous fit
            predictors = self._predictors

            begin_at_stage = self.n_iter_
```
115 - /tmp/repos/scikit-learn/examples/plot_anomaly_comparison.py
```python
for i_dataset, X in enumerate(datasets):
    # Add outliers
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                       size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

```
116 - /tmp/repos/scikit-learn/examples/svm/plot_svm_scale_c.py
```python
"""
==============================================
Scaling the regularization parameter for SVCs
==============================================

The following example illustrates the effect of scaling the
regularization parameter when using :ref:`svm` for
:ref:`classification <svm_classification>`.
For SVC classification, we are interested in a risk minimization for the
equation:


.. math::

    C \sum_{i=1, n} \mathcal{L} (f(x_i), y_i) + \Omega (w)

where

    - :math:`C` is used to set the amount of regularization
    - :math:`\mathcal{L}` is a `loss` function of our samples
      and our model parameters.
    - :math:`\Omega` is a `penalty` function of our model parameters

If we consider the loss function to be the individual error per
sample, then the data-fit term, or the sum of the error for each sample, will
increase as we add more samples. The penalization term, however, will not
increase.

When using, for example, :ref:`cross validation <cross_validation>`, to
set the amount of regularization with `C`, there will be a
different amount of samples between the main problem and the smaller problems
within the folds of the cross validation.

Since our loss function is dependent on the amount of samples, the latter
will influence the selected value of `C`.
The question that arises is `How do we optimally adjust C to
account for the different amount of training samples?`

The figures below are used to illustrate the effect of scaling our
`C` to compensate for the change in the number of samples, in the
case of using an `l1` penalty, as well as the `l2` penalty.

l1-penalty case
-----------------
In the `l1` case, theory says that prediction consistency
(i.e. that under given hypothesis, the estimator
learned predicts as well as a model knowing the true distribution)
is not possible because of the bias of the `l1`. It does say, however,
that model consistency, in terms of finding the right set of non-zero
parameters as well as their signs, can be achieved by scaling
`C1`.

l2-penalty case
-----------------
The theory says that in order to achieve prediction consistency, the
penalty parameter should be kept constant
as the number of samples grow.

Simulations
------------

The two figures below plot the values of `C` on the `x-axis` and the
corresponding cross-validation scores on the `y-axis`, for several different
fractions of a generated data-set.

In the `l1` penalty case, the cross-validation-error correlates best with
the test-error, when scaling our `C` with the number of samples, `n`,
which can be seen in the first figure.

For the `l2` penalty case, the best result comes from the case where `C`
is not scaled.

.. topic:: Note:

    Two separate datasets are used for the two different plots. The reason
    behind this is the `l1` case works better on sparse data, while `l2`
    is better suited to the non-sparse case.
"""
print(__doc__)


# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Jaques Grobler <jaques.grobler@inria.fr>
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn import datasets

rnd = check_random_state(1)

# set up dataset
n_samples = 100
n_features = 300

# l1 data (only 5 informative features)
X_1, y_1 = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features, n_informative=5,
                                        random_state=1)

# l2 data: non sparse, but less features
y_2 = np.sign(.5 - rnd.rand(n_samples))
X_2 = rnd.randn(n_samples, n_features // 5) + y_2[:, np.newaxis]
X_2 += 5 * rnd.randn(n_samples, n_features // 5)


```
117 - /tmp/repos/scikit-learn/examples/covariance/plot_robust_vs_empirical_covariance.py
```python
for i, n_outliers in enumerate(range_n_outliers):
    for j in range(repeat):

        rng = np.random.RandomState(i * j)

        # generate data
        X = rng.randn(n_samples, n_features)
        # add some outliers
        outliers_index = rng.permutation(n_samples)[:n_outliers]
        outliers_offset = 10. * \
            (np.random.randint(2, size=(n_outliers, n_features)) - 0.5)
        X[outliers_index] += outliers_offset
        inliers_mask = np.ones(n_samples).astype(bool)
        inliers_mask[outliers_index] = False

        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        mcd = MinCovDet().fit(X)
        # compare raw robust estimates with the true location and covariance
        err_loc_mcd[i, j] = np.sum(mcd.location_ ** 2)
        err_cov_mcd[i, j] = mcd.error_norm(np.eye(n_features))

        # compare estimators learned from the full data set with true
        # parameters
        err_loc_emp_full[i, j] = np.sum(X.mean(0) ** 2)
        err_cov_emp_full[i, j] = EmpiricalCovariance().fit(X).error_norm(
            np.eye(n_features))

        # compare with an empirical covariance learned from a pure data set
        # (i.e. "perfect" mcd)
        pure_X = X[inliers_mask]
        pure_location = pure_X.mean(0)
        pure_emp_cov = EmpiricalCovariance().fit(pure_X)
        err_loc_emp_pure[i, j] = np.sum(pure_location ** 2)
        err_cov_emp_pure[i, j] = pure_emp_cov.error_norm(np.eye(n_features))

# Display results
font_prop = matplotlib.font_manager.FontProperties(size=11)
plt.subplot(2, 1, 1)
lw = 2
plt.errorbar(range_n_outliers, err_loc_mcd.mean(1),
             yerr=err_loc_mcd.std(1) / np.sqrt(repeat),
             label="Robust location", lw=lw, color='m')
plt.errorbar(range_n_outliers, err_loc_emp_full.mean(1),
             yerr=err_loc_emp_full.std(1) / np.sqrt(repeat),
             label="Full data set mean", lw=lw, color='green')
plt.errorbar(range_n_outliers, err_loc_emp_pure.mean(1),
             yerr=err_loc_emp_pure.std(1) / np.sqrt(repeat),
             label="Pure data set mean", lw=lw, color='black')
plt.title("Influence of outliers on the location estimation")
plt.ylabel(r"Error ($||\mu - \hat{\mu}||_2^2$)")
plt.legend(loc="upper left", prop=font_prop)

plt.subplot(2, 1, 2)
x_size = range_n_outliers.size
plt.errorbar(range_n_outliers, err_cov_mcd.mean(1),
             yerr=err_cov_mcd.std(1),
             label="Robust covariance (mcd)", color='m')
plt.errorbar(range_n_outliers[:(x_size // 5 + 1)],
             err_cov_emp_full.mean(1)[:(x_size // 5 + 1)],
             yerr=err_cov_emp_full.std(1)[:(x_size // 5 + 1)],
             label="Full data set empirical covariance", color='green')
plt.plot(range_n_outliers[(x_size // 5):(x_size // 2 - 1)],
         err_cov_emp_full.mean(1)[(x_size // 5):(x_size // 2 - 1)],
         color='green', ls='--')
plt.errorbar(range_n_outliers, err_cov_emp_pure.mean(1),
             yerr=err_cov_emp_pure.std(1),
             label="Pure data set empirical covariance", color='black')
plt.title("Influence of outliers on the covariance estimation")
plt.xlabel("Amount of contamination (%)")
plt.ylabel("RMSE")
plt.legend(loc="upper center", prop=font_prop)

plt.show()

```
118 - /tmp/repos/scikit-learn/benchmarks/bench_saga.py
```python
def exp(solvers, penalties, single_target, n_samples=30000, max_iter=20,
        dataset='rcv1', n_jobs=1, skip_slow=False):
    mem = Memory(cachedir=expanduser('~/cache'), verbose=0)

    if dataset == 'rcv1':
        rcv1 = fetch_rcv1()

        lbin = LabelBinarizer()
        lbin.fit(rcv1.target_names)

        X = rcv1.data
        y = rcv1.target
        y = lbin.inverse_transform(y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        if single_target:
            y_n = y.copy()
            y_n[y > 16] = 1
            y_n[y <= 16] = 0
            y = y_n

    elif dataset == 'digits':
        digits = load_digits()
        X, y = digits.data, digits.target
        if single_target:
            y_n = y.copy()
            y_n[y < 5] = 1
            y_n[y >= 5] = 0
            y = y_n
    elif dataset == 'iris':
        iris = load_iris()
        X, y = iris.data, iris.target
    elif dataset == '20newspaper':
        ng = fetch_20newsgroups_vectorized()
        X = ng.data
        y = ng.target
        if single_target:
            y_n = y.copy()
            y_n[y > 4] = 1
            y_n[y <= 16] = 0
            y = y_n

    X = X[:n_samples]
    y = y[:n_samples]

    cached_fit = mem.cache(fit_single)
    out = Parallel(n_jobs=n_jobs, mmap_mode=None)(
        delayed(cached_fit)(solver, X, y,
                            penalty=penalty, single_target=single_target,
                            C=1, max_iter=max_iter, skip_slow=skip_slow)
        for solver in solvers
        for penalty in penalties)

    res = []
    idx = 0
    for solver in solvers:
        for penalty in penalties:
            if not (skip_slow and solver == 'lightning' and penalty == 'l1'):
                lr, times, train_scores, test_scores, accuracies = out[idx]
                this_res = dict(solver=solver, penalty=penalty,
                                single_target=single_target,
                                times=times, train_scores=train_scores,
                                test_scores=test_scores,
                                accuracies=accuracies)
                res.append(this_res)
            idx += 1

    with open('bench_saga.json', 'w+') as f:
        json.dump(res, f)



```
119 - /tmp/repos/scikit-learn/examples/covariance/plot_lw_vs_oas.py
```python
"""
=============================
Ledoit-Wolf vs OAS estimation
=============================

The usual covariance maximum likelihood estimate can be regularized
using shrinkage. Ledoit and Wolf proposed a close formula to compute
the asymptotically optimal shrinkage parameter (minimizing a MSE
criterion), yielding the Ledoit-Wolf covariance estimate.

Chen et al. proposed an improvement of the Ledoit-Wolf shrinkage
parameter, the OAS coefficient, whose convergence is significantly
better under the assumption that the data are Gaussian.

This example, inspired from Chen's publication [1], shows a comparison
of the estimated MSE of the LW and OAS methods, using Gaussian
distributed data.

[1] "Shrinkage Algorithms for MMSE Covariance Estimation"
Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky

from sklearn.covariance import LedoitWolf, OAS

np.random.seed(0)
###############################################################################
n_features = 100
# simulation covariance matrix (AR(1) process)
r = 0.1
real_cov = toeplitz(r ** np.arange(n_features))
coloring_matrix = cholesky(real_cov)

n_samples_range = np.arange(6, 31, 1)
repeat = 100
lw_mse = np.zeros((n_samples_range.size, repeat))
oa_mse = np.zeros((n_samples_range.size, repeat))
lw_shrinkage = np.zeros((n_samples_range.size, repeat))
oa_shrinkage = np.zeros((n_samples_range.size, repeat))
for i, n_samples in enumerate(n_samples_range):
    for j in range(repeat):
        X = np.dot(
            np.random.normal(size=(n_samples, n_features)), coloring_matrix.T)

        lw = LedoitWolf(store_precision=False, assume_centered=True)
        lw.fit(X)
        lw_mse[i, j] = lw.error_norm(real_cov, scaling=False)
        lw_shrinkage[i, j] = lw.shrinkage_

        oa = OAS(store_precision=False, assume_centered=True)
        oa.fit(X)
        oa_mse[i, j] = oa.error_norm(real_cov, scaling=False)
        oa_shrinkage[i, j] = oa.shrinkage_

# plot MSE
plt.subplot(2, 1, 1)
plt.errorbar(n_samples_range, lw_mse.mean(1), yerr=lw_mse.std(1),
             label='Ledoit-Wolf', color='navy', lw=2)
plt.errorbar(n_samples_range, oa_mse.mean(1), yerr=oa_mse.std(1),
             label='OAS', color='darkorange', lw=2)
plt.ylabel("Squared error")
plt.legend(loc="upper right")
plt.title("Comparison of covariance estimators")
plt.xlim(5, 31)

# plot shrinkage coefficient
plt.subplot(2, 1, 2)
plt.errorbar(n_samples_range, lw_shrinkage.mean(1), yerr=lw_shrinkage.std(1),
             label='Ledoit-Wolf', color='navy', lw=2)
plt.errorbar(n_samples_range, oa_shrinkage.mean(1), yerr=oa_shrinkage.std(1),
             label='OAS', color='darkorange', lw=2)
plt.xlabel("n_samples")
plt.ylabel("Shrinkage")
plt.legend(loc="lower right")
plt.ylim(plt.ylim()[0], 1. + (plt.ylim()[1] - plt.ylim()[0]) / 10.)
plt.xlim(5, 31)

plt.show()
```
120 - /tmp/repos/scikit-learn/examples/ensemble/plot_adaboost_multiclass.py
```python
"""
=====================================
Multi-class AdaBoosted Decision Trees
=====================================

This example reproduces Figure 1 of Zhu et al [1]_ and shows how boosting can
improve prediction accuracy on a multi-class problem. The classification
dataset is constructed by taking a ten-dimensional standard normal distribution
and defining three classes separated by nested concentric ten-dimensional
spheres such that roughly equal numbers of samples are in each class (quantiles
of the :math:`\chi^2` distribution).

The performance of the SAMME and SAMME.R [1]_ algorithms are compared. SAMME.R
uses the probability estimates to update the additive model, while SAMME  uses
the classifications only. As the example illustrates, the SAMME.R algorithm
typically converges faster than SAMME, achieving a lower test error with fewer
boosting iterations. The error of each algorithm on the test set after each
boosting iteration is shown on the left, the classification error on the test
set of each tree is shown in the middle, and the boost weight of each tree is
shown on the right. All trees have a weight of one in the SAMME.R algorithm and
therefore are not shown.

.. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

"""
print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()

```
121 - /tmp/repos/scikit-learn/benchmarks/bench_plot_randomized_svd.py
```python
"""
Benchmarks on the power iterations phase in randomized SVD.

We test on various synthetic and real datasets the effect of increasing
the number of power iterations in terms of quality of approximation
and running time. A number greater than 0 should help with noisy matrices,
which are characterized by a slow spectral decay.

We test several policy for normalizing the power iterations. Normalization
is crucial to avoid numerical issues.

The quality of the approximation is measured by the spectral norm discrepancy
between the original input matrix and the reconstructed one (by multiplying
the randomized_svd's outputs). The spectral norm is always equivalent to the
largest singular value of a matrix. (3) justifies this choice. However, one can
notice in these experiments that Frobenius and spectral norms behave
very similarly in a qualitative sense. Therefore, we suggest to run these
benchmarks with `enable_spectral_norm = False`, as Frobenius' is MUCH faster to
compute.

The benchmarks follow.

(a) plot: time vs norm, varying number of power iterations
    data: many datasets
    goal: compare normalization policies and study how the number of power
    iterations affect time and norm

(b) plot: n_iter vs norm, varying rank of data and number of components for
    randomized_SVD
    data: low-rank matrices on which we control the rank
    goal: study whether the rank of the matrix and the number of components
    extracted by randomized SVD affect "the optimal" number of power iterations

(c) plot: time vs norm, varing datasets
    data: many datasets
    goal: compare default configurations

We compare the following algorithms:
-   randomized_svd(..., power_iteration_normalizer='none')
-   randomized_svd(..., power_iteration_normalizer='LU')
-   randomized_svd(..., power_iteration_normalizer='QR')
-   randomized_svd(..., power_iteration_normalizer='auto')
-   fbpca.pca() from https://github.com/facebook/fbpca (if installed)

Conclusion
----------
- n_iter=2 appears to be a good default value
- power_iteration_normalizer='none' is OK if n_iter is small, otherwise LU
  gives similar errors to QR but is cheaper. That's what 'auto' implements.

References
----------
(1) Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

(2) A randomized algorithm for the decomposition of matrices
    Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

(3) An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
"""

# Author: Giorgio Patrini

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import gc
import pickle
from time import time
from collections import defaultdict
import os.path

from sklearn.utils import gen_batches
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets.samples_generator import (make_low_rank_matrix,
                                                make_sparse_uncorrelated)
from sklearn.datasets import (fetch_lfw_people,
                              fetch_mldata,
                              fetch_20newsgroups_vectorized,
                              fetch_olivetti_faces,
                              fetch_rcv1)

try:
    import fbpca
    fbpca_available = True
except ImportError:
    fbpca_available = False

# If this is enabled, tests are much slower and will crash with the large data
enable_spectral_norm = False

# TODO: compute approximate spectral norms with the power method as in
# Estimating the largest eigenvalues by the power and Lanczos methods with
# a random start, Jacek Kuczynski and Henryk Wozniakowski, SIAM Journal on
# Matrix Analysis and Applications, 13 (4): 1094-1122, 1992.
# This approximation is a very fast estimate of the spectral norm, but depends
# on starting random vectors.

# Determine when to switch to batch computation for matrix norms,
# in case the reconstructed (dense) matrix is too large
MAX_MEMORY = np.int(2e9)

# The following datasets can be dowloaded manually from:
# CIFAR 10: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# SVHN: http://ufldl.stanford.edu/housenumbers/train_32x32.mat
CIFAR_FOLDER = "./cifar-10-batches-py/"
SVHN_FOLDER = "./SVHN/"


```
122 - /tmp/repos/scikit-learn/examples/decomposition/plot_pca_vs_fa_model_selection.py
```python
"""
===============================================================
Model selection with Probabilistic PCA and Factor Analysis (FA)
===============================================================

Probabilistic PCA and Factor Analysis are probabilistic models.
The consequence is that the likelihood of new data can be used
for model selection and covariance estimation.
Here we compare PCA and FA with cross-validation on low rank data corrupted
with homoscedastic noise (noise variance
is the same for each feature) or heteroscedastic noise (noise variance
is the different for each feature). In a second step we compare the model
likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed
in recovering the size of the low rank subspace. The likelihood with PCA
is higher than FA in this case. However PCA fails and overestimates
the rank when heteroscedastic noise is present. Under appropriate
circumstances the low rank models are more likely than shrinkage models.

The automatic estimation from
Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
by Thomas P. Minka is also compared.

"""

# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print(__doc__)

# #############################################################################
# Create the data

n_samples, n_features, rank = 1000, 50, 10
sigma = 1.
rng = np.random.RandomState(42)
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

# #############################################################################
# Fit the models

n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))
```
123 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
```python
"""
    if warm_start_mem is None:
        warm_start_mem = {}
    # Ridge default max_iter is None
    if max_iter is None:
        max_iter = 1000

    if check_input:
        X = check_array(X, dtype=np.float64, accept_sparse='csr', order='C')
        y = check_array(y, dtype=np.float64, ensure_2d=False, order='C')

    n_samples, n_features = X.shape[0], X.shape[1]
    # As in SGD, the alpha is scaled by n_samples.
    alpha_scaled = float(alpha) / n_samples
    beta_scaled = float(beta) / n_samples

    # if loss == 'multinomial', y should be label encoded.
    n_classes = int(y.max()) + 1 if loss == 'multinomial' else 1

    # initialization
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64, order='C')

    if 'coef' in warm_start_mem.keys():
        coef_init = warm_start_mem['coef']
    else:
        # assume fit_intercept is False
        coef_init = np.zeros((n_features, n_classes), dtype=np.float64,
                             order='C')

    # coef_init contains possibly the intercept_init at the end.
    # Note that Ridge centers the data before fitting, so fit_intercept=False.
    fit_intercept = coef_init.shape[0] == (n_features + 1)
    if fit_intercept:
        intercept_init = coef_init[-1, :]
        coef_init = coef_init[:-1, :]
    else:
        intercept_init = np.zeros(n_classes, dtype=np.float64)

    if 'intercept_sum_gradient' in warm_start_mem.keys():
        intercept_sum_gradient = warm_start_mem['intercept_sum_gradient']
    else:
        intercept_sum_gradient = np.zeros(n_classes, dtype=np.float64)

    if 'gradient_memory' in warm_start_mem.keys():
        gradient_memory_init = warm_start_mem['gradient_memory']
    else:
        gradient_memory_init = np.zeros((n_samples, n_classes),
                                        dtype=np.float64, order='C')
    if 'sum_gradient' in warm_start_mem.keys():
        sum_gradient_init = warm_start_mem['sum_gradient']
    else:
        sum_gradient_init = np.zeros((n_features, n_classes),
                                     dtype=np.float64, order='C')

    if 'seen' in warm_start_mem.keys():
        seen_init = warm_start_mem['seen']
    else:
        seen_init = np.zeros(n_samples, dtype=np.int32, order='C')

    if 'num_seen' in warm_start_mem.keys():
        num_seen_init = warm_start_mem['num_seen']
    else:
        num_seen_init = 0

    dataset, intercept_decay = make_dataset(X, y, sample_weight, random_state)

    if max_squared_sum is None:
        max_squared_sum = row_norms(X, squared=True).max()
    step_size = get_auto_step_size(max_squared_sum, alpha_scaled, loss,
                                   fit_intercept, n_samples=n_samples,
                                   is_saga=is_saga)
    if step_size * alpha_scaled == 1:
        raise ZeroDivisionError("Current sag implementation does not handle "
                                "the case step_size * alpha_scaled == 1")

    num_seen, n_iter_ = sag(dataset, coef_init,
                            intercept_init, n_samples,
                            n_features, n_classes, tol,
                            max_iter,
                            loss,
                            step_size, alpha_scaled,
                            beta_scaled,
                            sum_gradient_init,
                            gradient_memory_init,
                            seen_init,
                            num_seen_init,
                            fit_intercept,
                            intercept_sum_gradient,
                            intercept_decay,
                            is_saga,
                            verbose)
    if n_iter_ == max_iter:
        warnings.warn("The max_iter was reached which means "
                      "the coef_ did not converge", ConvergenceWarning)

    if fit_intercept:
        coef_init = np.vstack((coef_init, intercept_init))

    warm_start_mem = {'coef': coef_init, 'sum_gradient': sum_gradient_init,
                      'intercept_sum_gradient': intercept_sum_gradient,
                      'gradient_memory': gradient_memory_init,
                      'seen': seen_init, 'num_seen': num_seen}

    if loss == 'multinomial':
        coef_ = coef_init.T
    else:
        coef_ = coef_init[:, 0]

    return coef_, n_iter_, warm_start_mem

```
124 - /tmp/repos/scikit-learn/examples/applications/plot_model_complexity_influence.py
```python
def _count_nonzero_coefficients(estimator):
    a = estimator.coef_.toarray()
    return np.count_nonzero(a)

# #############################################################################
# Main code
regression_data = generate_data('regression')
classification_data = generate_data('classification', sparse=True)
configurations = [
    {'estimator': SGDClassifier,
     'tuned_params': {'penalty': 'elasticnet', 'alpha': 0.001, 'loss':
                      'modified_huber', 'fit_intercept': True, 'tol': 1e-3},
     'changing_param': 'l1_ratio',
     'changing_param_values': [0.25, 0.5, 0.75, 0.9],
     'complexity_label': 'non_zero coefficients',
     'complexity_computer': _count_nonzero_coefficients,
     'prediction_performance_computer': hamming_loss,
     'prediction_performance_label': 'Hamming Loss (Misclassification Ratio)',
     'postfit_hook': lambda x: x.sparsify(),
     'data': classification_data,
     'n_samples': 30},
    {'estimator': NuSVR,
     'tuned_params': {'C': 1e3, 'gamma': 2 ** -15},
     'changing_param': 'nu',
     'changing_param_values': [0.1, 0.25, 0.5, 0.75, 0.9],
     'complexity_label': 'n_support_vectors',
     'complexity_computer': lambda x: len(x.support_vectors_),
     'data': regression_data,
     'postfit_hook': lambda x: x,
     'prediction_performance_computer': mean_squared_error,
     'prediction_performance_label': 'MSE',
     'n_samples': 30},
    {'estimator': GradientBoostingRegressor,
     'tuned_params': {'loss': 'ls'},
     'changing_param': 'n_estimators',
     'changing_param_values': [10, 50, 100, 200, 500],
     'complexity_label': 'n_trees',
     'complexity_computer': lambda x: x.n_estimators,
     'data': regression_data,
     'postfit_hook': lambda x: x,
     'prediction_performance_computer': mean_squared_error,
     'prediction_performance_label': 'MSE',
     'n_samples': 30},
]
for conf in configurations:
    prediction_performances, prediction_times, complexities = \
        benchmark_influence(conf)
    plot_influence(conf, prediction_performances, prediction_times,
                   complexities)

```
125 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def _init_decision_function(self, X):
        """Check input and compute prediction of ``init``. """
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        score = self.init_.predict(X).astype(np.float64)
        return score

    def _decision_function(self, X):
        # for use in inner loop, not raveling the output in single-class case,
        # not doing input validation.
        score = self._init_decision_function(X)
        predict_stages(self.estimators_, X, self.learning_rate, score)
        return score


    def _staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        score = self._init_decision_function(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, score)
            yield score.copy()

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        self._check_initialized()

        total_sum = np.zeros((self.n_features_, ), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.feature_importances_
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def _validate_y(self, y, sample_weight):
        # 'sample_weight' is not utilised but is used for
        # consistency with similar method _validate_y of GBC
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        # Default implementation
        return y

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples, n_estimators, n_classes]
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

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

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

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

    init : estimator, optional
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
126 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python



n_samples_list = [1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]
n_samples_list = [n_samples for n_samples in n_samples_list
                  if n_samples <= args.n_samples_max]

sklearn_scores = []
sklearn_fit_durations = []
sklearn_score_durations = []
lightgbm_scores = []
lightgbm_fit_durations = []
lightgbm_score_durations = []
xgb_scores = []
xgb_fit_durations = []
xgb_score_durations = []
cat_scores = []
cat_fit_durations = []
cat_score_durations = []

for n_samples in n_samples_list:
    (sklearn_score,
     sklearn_fit_duration,
     sklearn_score_duration,
     lightgbm_score,
     lightgbm_fit_duration,
     lightgbm_score_duration,
     xgb_score,
     xgb_fit_duration,
     xgb_score_duration,
     cat_score,
     cat_fit_duration,
     cat_score_duration) = one_run(n_samples)

    for scores, score in (
            (sklearn_scores, sklearn_score),
            (sklearn_fit_durations, sklearn_fit_duration),
            (sklearn_score_durations, sklearn_score_duration),
            (lightgbm_scores, lightgbm_score),
            (lightgbm_fit_durations, lightgbm_fit_duration),
            (lightgbm_score_durations, lightgbm_score_duration),
            (xgb_scores, xgb_score),
            (xgb_fit_durations, xgb_fit_duration),
            (xgb_score_durations, xgb_score_duration),
            (cat_scores, cat_score),
            (cat_fit_durations, cat_fit_duration),
            (cat_score_durations, cat_score_duration)):
        scores.append(score)

fig, axs = plt.subplots(3, sharex=True)

axs[0].plot(n_samples_list, sklearn_scores, label='sklearn')
axs[1].plot(n_samples_list, sklearn_fit_durations, label='sklearn')
axs[2].plot(n_samples_list, sklearn_score_durations, label='sklearn')

if args.lightgbm:
    axs[0].plot(n_samples_list, lightgbm_scores, label='lightgbm')
    axs[1].plot(n_samples_list, lightgbm_fit_durations, label='lightgbm')
    axs[2].plot(n_samples_list, lightgbm_score_durations, label='lightgbm')

if args.xgboost:
    axs[0].plot(n_samples_list, xgb_scores, label='XGBoost')
    axs[1].plot(n_samples_list, xgb_fit_durations, label='XGBoost')
    axs[2].plot(n_samples_list, xgb_score_durations, label='XGBoost')

if args.catboost:
    axs[0].plot(n_samples_list, cat_scores, label='CatBoost')
    axs[1].plot(n_samples_list, cat_fit_durations, label='CatBoost')
    axs[2].plot(n_samples_list, cat_score_durations, label='CatBoost')

for ax in axs:
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.set_xlabel('n_samples')

axs[0].set_title('scores')
axs[1].set_title('fit duration (s)')
axs[2].set_title('score duration (s)')

title = args.problem
if args.problem == 'classification':
    title += ' n_classes = {}'.format(args.n_classes)
fig.suptitle(title)


plt.tight_layout()
plt.show()

```
**127 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py**:
```python
# coding: utf-8

# Author: Johannes Schönberger
#
# License: BSD 3 clause

import numpy as np
import warnings

from ..base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from ..utils import check_random_state, check_array, check_consistent_length
from ..utils.random import sample_without_replacement
from ..utils.validation import check_is_fitted
from .base import LinearRegression
from ..utils.validation import has_fit_parameter
from ..exceptions import ConvergenceWarning

_EPSILON = np.spacing(1)


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.

    n_samples : int
        Total number of samples in the data.

    min_samples : int
        Minimum number of samples chosen randomly from original data.

    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.

    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float('inf')
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


class RANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    A detailed description of the algorithm can be found in the documentation
    of the ``linear_model`` sub-package.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the following methods:

         * `fit(X, y)`: Fit model to given training data and target values.
         * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.

        If `base_estimator` is None, then
        ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), optional
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0]`) for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `base_estimator`. By default a
        ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``.

    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`.

    is_data_valid : callable, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    max_skips : int, optional
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

        .. versionadded:: 0.19

    stop_n_inliers : int, optional
        Stop iteration if at least this number of inliers are found.

    stop_score : float, optional
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    residual_metric : callable, optional
        Metric to reduce the dimensionality of the residuals to 1 for
        multi-dimensional target values ``y.shape[1] > 1``. By default the sum
        of absolute differences is used::

            lambda dy: np.sum(np.abs(dy), axis=1)

        .. deprecated:: 0.18
           ``residual_metric`` is deprecated from 0.18 and will be removed in
           0.20. Use ``loss`` instead.

    loss : string, callable, optional, default "absolute_loss"
        String inputs, "absolute_loss" and "squared_loss" are supported which
        find the absolute loss and squared loss per sample
        respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

    random_state : int, RandomState instance or None, optional, default None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `base_estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    n_skips_no_inliers_ : int
        Number of iterations skipped due to finding zero inliers.

        .. versionadded:: 0.19

    n_skips_invalid_data_ : int
        Number of iterations skipped due to invalid data defined by
        ``is_data_valid``.

        .. versionadded:: 0.19

    n_skips_invalid_model_ : int
        Number of iterations skipped due to an invalid model defined by
        ``is_model_valid``.

        .. versionadded:: 0.19

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] https://www.sri.com/sites/default/files/publications/ransac-publication.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
    
```
128 - /tmp/repos/scikit-learn/examples/covariance/plot_covariance_estimation.py
```python
"""
=======================================================================
Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood
=======================================================================

When working with covariance estimation, the usual approach is to use
a maximum likelihood estimator, such as the
:class:`sklearn.covariance.EmpiricalCovariance`. It is unbiased, i.e. it
converges to the true (population) covariance when given many
observations. However, it can also be beneficial to regularize it, in
order to reduce its variance; this, in turn, introduces some bias. This
example illustrates the simple regularization used in
:ref:`shrunk_covariance` estimators. In particular, it focuses on how to
set the amount of regularization, i.e. how to choose the bias-variance
trade-off.

Here we compare 3 approaches:

* Setting the parameter by cross-validating the likelihood on three folds
  according to a grid of potential shrinkage parameters.

* A close formula proposed by Ledoit and Wolf to compute
  the asymptotically optimal regularization parameter (minimizing a MSE
  criterion), yielding the :class:`sklearn.covariance.LedoitWolf`
  covariance estimate.

* An improvement of the Ledoit-Wolf shrinkage, the
  :class:`sklearn.covariance.OAS`, proposed by Chen et al. Its
  convergence is significantly better under the assumption that the data
  are Gaussian, in particular for small samples.

To quantify estimation error, we plot the likelihood of unseen data for
different values of the shrinkage parameter. We also show the choices by
cross-validation, or with the LedoitWolf and OAS estimates.

Note that the maximum likelihood estimate corresponds to no shrinkage,
and thus performs poorly. The Ledoit-Wolf estimate performs really well,
as it is close to the optimal and is computational not costly. In this
example, the OAS estimate is a bit further away. Interestingly, both
approaches outperform cross-validation, which is significantly most
computationally costly.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, \
    log_likelihood, empirical_covariance
from sklearn.model_selection import GridSearchCV


# #############################################################################
# Generate sample data
n_features, n_samples = 40, 20
np.random.seed(42)
base_X_train = np.random.normal(size=(n_samples, n_features))
base_X_test = np.random.normal(size=(n_samples, n_features))

# Color samples
coloring_matrix = np.random.normal(size=(n_features, n_features))
X_train = np.dot(base_X_train, coloring_matrix)
X_test = np.dot(base_X_test, coloring_matrix)

# #############################################################################
# Compute the likelihood on test data

# spanning a range of possible shrinkage coefficient values
shrinkages = np.logspace(-2, 0, 30)
negative_logliks = [-ShrunkCovariance(shrinkage=s).fit(X_train).score(X_test)
                    for s in shrinkages]

# under the ground-truth model, which we would not have access to in real
# settings
real_cov = np.dot(coloring_matrix.T, coloring_matrix)
emp_cov = empirical_covariance(X_train)
loglik_real = -log_likelihood(emp_cov, linalg.inv(real_cov))

# #############################################################################
# Compare different approaches to setting the parameter

# GridSearch for an optimal shrinkage coefficient
tuned_parameters = [{'shrinkage': shrinkages}]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
cv.fit(X_train)

# Ledoit-Wolf optimal shrinkage coefficient estimate
lw = LedoitWolf()
loglik_lw = lw.fit(X_train).score(X_test)

# OAS coefficient estimate
oa = OAS()
loglik_oa = oa.fit(X_train).score(X_test)

# #############################################################################
# Plot results
fig = plt.figure()
plt.title("Regularized covariance: likelihood and shrinkage coefficient")
plt.xlabel('Regularization parameter: shrinkage coefficient')
plt.ylabel('Error: negative log-likelihood on test data')
# range shrinkage curve
plt.loglog(shrinkages, negative_logliks, label="Negative log-likelihood")

plt.plot(plt.xlim(), 2 * [loglik_real], '--r',
         label="Real covariance likelihood")

# adjust view
lik_max = np.amax(negative_logliks)
lik_min = np.amin(negative_logliks)
ymin = lik_min - 6. * np.log((plt.ylim()[1] - plt.ylim()[0]))
ymax = lik_max + 10. * np.log(lik_max - lik_min)
xmin = shrinkages[0]
xmax = shrinkages[-1]

```
129 - /tmp/repos/scikit-learn/sklearn/utils/_scipy_sparse_lsqr_backport.py
```python
# The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= iter_lim-10:
            prnt = True
        # if itn%10 == 0: prnt = True
        if test3 <= 2*ctol:
            prnt = True
        if test2 <= 10*atol:
            prnt = True
        if test1 <= 10*rtol:
            prnt = True
        if istop != 0:
            prnt = True

        if prnt:
            if show:
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (anorm, acond)
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
        str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
        str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
        str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

```
130 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle,
                               random_state):
    """Helper function for _fit_coordinate_descent

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...)

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)



```
131 - /tmp/repos/scikit-learn/examples/linear_model/plot_lasso_model_selection.py
```python
"""
===================================================
Lasso model selection: Cross-Validation / AIC / BIC
===================================================

Use the Akaike information criterion (AIC), the Bayes Information
criterion (BIC) and cross-validation to select an optimal value
of the regularization parameter alpha of the :ref:`lasso` estimator.

Results obtained with LassoLarsIC are based on AIC/BIC criteria.

Information-criterion based model selection is very fast, but it
relies on a proper estimation of degrees of freedom, are
derived for large samples (asymptotic results) and assume the model
is correct, i.e. that the data are actually generated by this model.
They also tend to break when the problem is badly conditioned
(more features than samples).

For cross-validation, we use 20-fold with 2 algorithms to compute the
Lasso path: coordinate descent, as implemented by the LassoCV class, and
Lars (least angle regression) as implemented by the LassoLarsCV class.
Both algorithms give roughly the same results. They differ with regards
to their execution speed and sources of numerical errors.

Lars computes a path solution only for each kink in the path. As a
result, it is very efficient when there are only of few kinks, which is
the case if there are few features or samples. Also, it is able to
compute the full path without setting any meta parameter. On the
opposite, coordinate descent compute the path points on a pre-specified
grid (here we use the default). Thus it is more efficient if the number
of grid points is smaller than the number of kinks in the path. Such a
strategy can be interesting if the number of features is really large
and there are enough samples to select a large amount. In terms of
numerical errors, for heavily correlated variables, Lars will accumulate
more errors, while the coordinate descent algorithm will only sample the
path on a grid.

Note how the optimal value of alpha varies for each fold. This
illustrates why nested-cross validation is necessary when trying to
evaluate the performance of a method for which a parameter is chosen by
cross-validation: this choice of parameter may not be optimal for unseen
data.
"""
print(__doc__)

# Author: Olivier Grisel, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]  # add some bad features

# normalize data as done by Lars to allow for comparison
X /= np.sqrt(np.sum(X ** 2, axis=0))

# #############################################################################
# LassoLarsIC: least angle regression with BIC/AIC criterion

model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')
```
132 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
def _fit_coordinate_descent(X, W, H, tol=1e-4, max_iter=200, l1_reg_W=0,
                            l1_reg_H=0, l2_reg_W=0, l2_reg_H=0, update_H=True,
                            verbose=0, shuffle=False, random_state=None):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    l1_reg_W : double, default: 0.
        L1 regularization parameter for W.

    l1_reg_H : double, default: 0.
        L1 regularization parameter for H.

    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.

    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    rng = check_random_state(random_state)

    for n_iter in range(max_iter):
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, Ht, l1_reg_W,
                                                l2_reg_W, shuffle, rng)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, l1_reg_H,
                                                    l2_reg_H, shuffle, rng)

        if n_iter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter



```
**133 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
"""
    _check_solver_option(solver, multi_class, penalty, dual)

    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y, sample_weight)

        sample_weight = sample_weight[train]

    coefs, Cs, n_iter = logistic_regression_path(
        X_train, y_train, Cs=Cs, fit_intercept=fit_intercept,
        solver=solver, max_iter=max_iter, class_weight=class_weight,
        pos_class=pos_class, multi_class=multi_class,
        tol=tol, verbose=verbose, dual=dual, penalty=penalty,
        intercept_scaling=intercept_scaling, random_state=random_state,
        check_input=False, max_squared_sum=max_squared_sum,
        sample_weight=sample_weight)

    log_reg = LogisticRegression(fit_intercept=fit_intercept)

    # The score method of Logistic Regression has a classes_ attribute.
    if multi_class == 'ovr':
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == 'multinomial':
        log_reg.classes_ = np.unique(y_train)
    else:
        raise ValueError("multi_class should be either multinomial or ovr, "
                         "got %d" % multi_class)

    if pos_class is not None:
        mask = (y_test == pos_class)
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.

    scores = list()

    if isinstance(scoring, six.string_types):
        scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == 'ovr':
            w = w[np.newaxis, :]
        if fit_intercept:
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            log_reg.coef_ = w
            log_reg.intercept_ = 0.

        if scoring is None:
            scores.append(log_reg.score(X_test, y_test))
        else:
            scores.append(scoring(log_reg, X_test, y_test))
    return coefs, Cs, np.array(scores), n_iter


class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag' and 'lbfgs' solvers. It can handle
    both dense and sparse input. Use C-ordered arrays or CSR matrices
    containing 64-bit floats for optimal performance; any other input format
    will be converted (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation. The 'liblinear' solver supports both L1 and L2
    regularization, with a dual formulation only for the L2 penalty.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'liblinear'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
            'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
            handle multinomial loss; 'liblinear' is limited to one-versus-rest
            schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
            'liblinear' and 'saga' handle L1 penalty.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for 'liblinear'
        solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    Attributes
    ----------

    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    n_iter_ : array, shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

    See also
    --------
    SGDClassifier : incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    sklearn.svm.LinearSVC : learns SVM models using the same algorithm.
    LogisticRegressionCV : Logistic regression with built-in cross validation

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
        SAGA: A Fast Incremental Gradient Method With Support
        for Non-Strongly Convex Composite Objectives
        https://arxiv.org/abs/1407.0202

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    
```
134 - /tmp/repos/scikit-learn/benchmarks/bench_plot_ward.py
```python
"""
Benchmark scikit-learn's Ward implement compared to SciPy's
"""

import time

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

ward = AgglomerativeClustering(n_clusters=3, linkage='ward')

n_samples = np.logspace(.5, 3, 9)
n_features = np.logspace(1, 3.5, 7)
N_samples, N_features = np.meshgrid(n_samples,
                                    n_features)
scikits_time = np.zeros(N_samples.shape)
scipy_time = np.zeros(N_samples.shape)

for i, n in enumerate(n_samples):
    for j, p in enumerate(n_features):
        X = np.random.normal(size=(n, p))
        t0 = time.time()
        ward.fit(X)
        scikits_time[j, i] = time.time() - t0
        t0 = time.time()
        hierarchy.ward(X)
        scipy_time[j, i] = time.time() - t0

ratio = scikits_time / scipy_time

plt.figure("scikit-learn Ward's method benchmark results")
plt.imshow(np.log(ratio), aspect='auto', origin="lower")
plt.colorbar()
plt.contour(ratio, levels=[1, ], colors='k')
plt.yticks(range(len(n_features)), n_features.astype(np.int))
plt.ylabel('N features')
plt.xticks(range(len(n_samples)), n_samples.astype(np.int))
plt.xlabel('N samples')
plt.title("Scikit's time, in units of scipy time (log)")
plt.show()
```
135 - /tmp/repos/scikit-learn/examples/calibration/plot_calibration_multiclass.py
```python
"""
==================================================
Probability Calibration for 3-class classification
==================================================

This example illustrates how sigmoid calibration changes predicted
probabilities for a 3-class classification problem. Illustrated is the
standard 2-simplex, where the three corners correspond to the three classes.
Arrows point from the probability vectors predicted by an uncalibrated
classifier to the probability vectors predicted by the same classifier after
sigmoid calibration on a hold-out validation set. Colors indicate the true
class of an instance (red: class 1, green: class 2, blue: class 3).

The base classifier is a random forest classifier with 25 base estimators
(trees). If this classifier is trained on all 800 training datapoints, it is
overly confident in its predictions and thus incurs a large log-loss.
Calibrating an identical classifier, which was trained on 600 datapoints, with
method='sigmoid' on the remaining 200 datapoints reduces the confidence of the
predictions, i.e., moves the probability vectors from the edges of the simplex
towards the center. This calibration results in a lower log-loss. Note that an
alternative would have been to increase the number of base estimators which
would have resulted in a similar decrease in log-loss.
"""
print(__doc__)

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.


import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

np.random.seed(0)

# Generate data
X, y = make_blobs(n_samples=1000, n_features=2, random_state=42,
                  cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)

# Train random forest classifier, calibrate on validation data and evaluate
# on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
sig_clf.fit(X_valid, y_valid)
sig_clf_probs = sig_clf.predict_proba(X_test)
sig_score = log_loss(y_test, sig_clf_probs)

# Plot changes in predicted probabilities via arrows
plt.figure(0)
colors = ["r", "g", "b"]
for i in range(clf_probs.shape[0]):
    plt.arrow(clf_probs[i, 0], clf_probs[i, 1],
              sig_clf_probs[i, 0] - clf_probs[i, 0],
              sig_clf_probs[i, 1] - clf_probs[i, 1],
              color=colors[y_test[i]], head_width=1e-2)

# Plot perfect predictions
plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")

# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

# Annotate points on the simplex
plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
             xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.plot([1.0/3], [1.0/3], 'ko', ms=5)

```
136 - /tmp/repos/scikit-learn/examples/linear_model/plot_sgd_early_stopping.py
```python
"""
=============================================
Early stopping of Stochastic Gradient Descent
=============================================

Stochastic Gradient Descent is an optimization technique which minimizes a loss
function in a stochastic fashion, performing a gradient descent step sample by
sample. In particular, it is a very efficient method to fit linear models.

As a stochastic method, the loss function is not necessarily decreasing at each
iteration, and convergence is only guaranteed in expectation. For this reason,
monitoring the convergence on the loss function can be difficult.

Another approach is to monitor convergence on a validation score. In this case,
the input data is split into a training set and a validation set. The model is
then fitted on the training set and the stopping criterion is based on the
prediction score computed on the validation set. This enables us to find the
least number of iterations which is sufficient to build a model that
generalizes well to unseen data and reduces the chance of over-fitting the
training data.

This early stopping strategy is activated if ``early_stopping=True``; otherwise
the stopping criterion only uses the training loss on the entire input data. To
better control the early stopping strategy, we can specify a parameter
``validation_fraction`` which set the fraction of the input dataset that we
keep aside to compute the validation score. The optimization will continue
until the validation score did not improve by at least ``tol`` during the last
``n_iter_no_change`` iterations. The actual number of iterations is available
at the attribute ``n_iter_``.

This example illustrates how the early stopping can used in the
:class:`sklearn.linear_model.SGDClassifier` model to achieve almost the same
accuracy as compared to a model built without early stopping. This can
significantly reduce training time. Note that scores differ between the
stopping criteria even from early iterations because some of the training data
is held out with the validation stopping criterion.
"""
# Authors: Tom Dupre la Tour
#
# License: BSD 3 clause
import time
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle

print(__doc__)


def load_mnist(n_samples=None, class_0='0', class_1='8'):
    """Load MNIST, select two classes, shuffle and return only n_samples."""
    # Load data from http://openml.org/d/554
    mnist = fetch_openml('mnist_784', version=1)

    # take only two classes for binary classification
    mask = np.logical_or(mnist.target == class_0, mnist.target == class_1)

    X, y = shuffle(mnist.data[mask], mnist.target[mask], random_state=42)
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    return X, y
```
137 - /tmp/repos/scikit-learn/sklearn/utils/_unittest_backport.py
```python
"""
This is a backport of assertRaises() and assertRaisesRegex from Python 3.5.4

The original copyright message is as follows

Python unit testing framework, based on Erich Gamma's JUnit and Kent Beck's
Smalltalk testing framework (used with permission).

This module contains the core framework classes that form the basis of
specific test cases and suites (TestCase, TestSuite etc.), and also a
text-based utility class for running the tests and reporting the results
 (TextTestRunner).

Simple usage:

    import unittest

    class IntegerArithmeticTestCase(unittest.TestCase):
        def testAdd(self):  # test method names begin with 'test'
            self.assertEqual((1 + 2), 3)
            self.assertEqual(0 + 1, 1)
        def testMultiply(self):
            self.assertEqual((0 * 10), 0)
            self.assertEqual((5 * 8), 40)

    if __name__ == '__main__':
        unittest.main()

Further information is available in the bundled documentation, and from

  http://docs.python.org/library/unittest.html

Copyright (c) 1999-2003 Steve Purcell
Copyright (c) 2003-2010 Python Software Foundation
This module is free software, and you may redistribute it and/or modify
it under the same terms as Python itself, so long as this copyright message
and disclaimer are retained in their original form.

IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
THIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import re
import warnings
import unittest


def _is_subtype(expected, basetype):
    if isinstance(expected, tuple):
        return all(_is_subtype(e, basetype) for e in expected)
    return isinstance(expected, type) and issubclass(expected, basetype)


class _BaseTestCaseContext:

    def __init__(self, test_case):
        self.test_case = test_case

    def _raiseFailure(self, standardMsg):
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)


class _AssertRaisesBaseContext(_BaseTestCaseContext):

    def __init__(self, expected, test_case, expected_regex=None):
        _BaseTestCaseContext.__init__(self, test_case)
        self.expected = expected
        self.test_case = test_case
        if expected_regex is not None:
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.obj_name = None
        self.msg = None

    def handle(self, name, args, kwargs):
        """
        If args is empty, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If args is not empty, call a callable passing positional and keyword
        arguments.
        """
        try:
            if not _is_subtype(self.expected, self._base_type):
                raise TypeError('%s() arg 1 must be %s' %
                                (name, self._base_type_str))
            if args and args[0] is None:
                warnings.warn("callable is None",
                              DeprecationWarning, 3)
                args = ()
            if not args:
                self.msg = kwargs.pop('msg', None)
                if kwargs:
                    warnings.warn('%r is an invalid keyword argument for '
                                  'this function' % next(iter(kwargs)),
                                  DeprecationWarning, 3)
                return self

            callable_obj, args = args[0], args[1:]
            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
            with self:
                callable_obj(*args, **kwargs)
        finally:
            # bpo-23890: manually break a reference cycle
            self = None



```
138 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def skip_if_32bit(func):
    """Test decorator that skips tests on 32bit platforms."""
    @wraps(func)
    def run_test(*args, **kwargs):
        bits = 8 * struct.calcsize("P")
        if bits == 32:
            raise SkipTest('Test skipped on 32bit platforms.')
        else:
            return func(*args, **kwargs)
    return run_test


def if_safe_multiprocessing_with_blas(func):
    """Decorator for tests involving both BLAS calls and multiprocessing.

    Under POSIX (e.g. Linux or OSX), using multiprocessing in conjunction with
    some implementation of BLAS (or other libraries that manage an internal
    posix thread pool) can cause a crash or a freeze of the Python process.

    In practice all known packaged distributions (from Linux distros or
    Anaconda) of BLAS under Linux seems to be safe. So we this problem seems to
    only impact OSX users.

    This wrapper makes it possible to skip tests that can possibly cause
    this crash under OS X with.

    Under Python 3.4+ it is possible to use the `forkserver` start method
    for multiprocessing to avoid this issue. However it can cause pickling
    errors on interactively defined functions. It therefore not enabled by
    default.
    """
    @wraps(func)
    def run_test(*args, **kwargs):
        if sys.platform == 'darwin':
            raise SkipTest(
                "Possible multi-process bug with some BLAS")
        return func(*args, **kwargs)
    return run_test


def clean_warning_registry():
    """Safe way to reset warnings."""
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


def check_skip_network():
    if int(os.environ.get('SKLEARN_SKIP_NETWORK_TESTS', 0)):
        raise SkipTest("Text tutorial requires large dataset download")


def check_skip_travis():
    """Skip test if being run on Travis."""
    if os.environ.get('TRAVIS') == "true":
        raise SkipTest("This test needs to be skipped on Travis")


def _delete_folder(folder_path, warn=False):
    """Utility function to cleanup a temporary folder if still existing.

    Copy from joblib.pool (for independence).
    """
    try:
        if os.path.exists(folder_path):
            # This can fail under windows,
            #  but will succeed when called by atexit
            shutil.rmtree(folder_path)
    except WindowsError:
        if warn:
            warnings.warn("Could not delete temporary folder %s" % folder_path)


class TempMemmap(object):
    def __init__(self, data, mmap_mode='r'):
        self.temp_folder = tempfile.mkdtemp(prefix='sklearn_testing_')
        self.mmap_mode = mmap_mode
        self.data = data

    def __enter__(self):
        fpath = op.join(self.temp_folder, 'data.pkl')
        joblib.dump(self.data, fpath)
        data_read_only = joblib.load(fpath, mmap_mode=self.mmap_mode)
        atexit.register(lambda: _delete_folder(self.temp_folder, warn=True))
        return data_read_only

    def __exit__(self, exc_type, exc_val, exc_tb):
        _delete_folder(self.temp_folder)


# Utils to test docstrings


def _get_args(function, varargs=False):
    """Helper to get function arguments"""

    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args



```
139 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    # avoid forcing order when copy_x=False
    order = "C" if copy_x else None
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32],
                    order=order, copy=copy_x)
    # verify that the number of samples given is larger than k
    if _num_samples(X) < n_clusters:
        raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            _num_samples(X), n_clusters))
    tol = _tolerance(X, tol)

    # If the distances are precomputed every job will create a matrix of shape
    # (n_clusters, n_samples). To stop KMeans from eating up memory we only
    # activate this if the created matrix is guaranteed to be under 100MB. 12
    # million entries consume a little under 100MB if they are of type double.
    if precompute_distances == 'auto':
        n_samples = X.shape[0]
        precompute_distances = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distances, bool):
        pass
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         precompute_distances)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    if n_clusters == 1:
        # elkan doesn't make sense for a single cluster, full will produce
        # the right result.
        algorithm = "full"
    if algorithm == "auto":
        algorithm = "full" if sp.issparse(X) else 'elkan'
    if algorithm == "full":
        kmeans_single = _kmeans_single_lloyd
    elif algorithm == "elkan":
        kmeans_single = _kmeans_single_elkan
    else:
        raise ValueError("Algorithm must be 'auto', 'full' or 'elkan', got"
                         " %s" % str(algorithm))
    
```
140 - /tmp/repos/scikit-learn/sklearn/cluster/hierarchical.py
```python
for k in xrange(n_samples, n_nodes):
        # identify the merge
        while True:
            edge = heappop(inertia)
            if used_node[edge.a] and used_node[edge.b]:
                break
        i = edge.a
        j = edge.b

        if return_distance:
            # store distances
            distances[k - n_samples] = edge.weight

        parent[i] = parent[j] = k
        children.append((i, j))
        # Keep track of the number of elements per cluster
        n_i = used_node[i]
        n_j = used_node[j]
        used_node[k] = n_i + n_j
        used_node[i] = used_node[j] = False

        # update the structure matrix A and the inertia matrix
        # a clever 'min', or 'max' operation between A[i] and A[j]
        coord_col = join_func(A[i], A[j], used_node, n_i, n_j)
        for l, d in coord_col:
            A[l].append(k, d)
            # Here we use the information from coord_col (containing the
            # distances) to update the heap
            heappush(inertia, _hierarchical.WeightedEdge(d, k, l))
        A[k] = coord_col
        # Clear A[i] and A[j] to save memory
        A[i] = A[j] = 0

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples

    # # return numpy array for efficient caching
    children = np.array(children)[:, ::-1]

    if return_distance:
        return children, n_components, n_leaves, parent, distances
    return children, n_components, n_leaves, parent


# Matching names to tree-building strategies
def _complete_linkage(*args, **kwargs):
    kwargs['linkage'] = 'complete'
    return linkage_tree(*args, **kwargs)


def _average_linkage(*args, **kwargs):
    kwargs['linkage'] = 'average'
    return linkage_tree(*args, **kwargs)


def _single_linkage(*args, **kwargs):
    kwargs['linkage'] = 'single'
    return linkage_tree(*args, **kwargs)


_TREE_BUILDERS = dict(
    ward=ward_tree,
    complete=_complete_linkage,
    average=_average_linkage,
    single=_single_linkage)


###############################################################################
# Functions for cutting  hierarchical clustering tree

def _hc_cut(n_clusters, children, n_leaves):
    """Function cutting the ward tree for a given number of clusters.

    Parameters
    ----------
    n_clusters : int or ndarray
        The number of clusters to form.

    children : 2D array, shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`

    n_leaves : int
        Number of leaves of the tree.

    Returns
    -------
    labels : array [n_samples]
        cluster labels for each point

    """
    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))
    # In this function, we store nodes as a heap to avoid recomputing
    # the max of the nodes: the first element is always the smallest
    # We use negated indices as heaps work on smallest elements, and we
    # are interested in largest elements
    # children[-1] is the root of the tree
    nodes = [-(max(children[-1]) + 1)]
    for i in xrange(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        label[_hierarchical._hc_get_descendent(-node, children, n_leaves)] = i
    return label


###############################################################################

class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    
```
141 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting_higgsboson.py
```python
from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import argparse

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.utils import (
    get_equivalent_estimator)


parser = argparse.ArgumentParser()
parser.add_argument('--n-leaf-nodes', type=int, default=31)
parser.add_argument('--n-trees', type=int, default=10)
parser.add_argument('--lightgbm', action="store_true", default=False)
parser.add_argument('--xgboost', action="store_true", default=False)
parser.add_argument('--catboost', action="store_true", default=False)
parser.add_argument('--learning-rate', type=float, default=1.)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--max-bins', type=int, default=255)
args = parser.parse_args()

HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')

n_leaf_nodes = args.n_leaf_nodes
n_trees = args.n_trees
subsample = args.subsample
lr = args.learning_rate
max_bins = args.max_bins


@m.cache
def load_data():
    filename = os.path.join(HERE, URL.rsplit('/', 1)[-1])
    if not os.path.exists(filename):
        print(f"Downloading {URL} to {filename} (2.6 GB)...")
        urlretrieve(URL, filename)
        print("done.")

    print(f"Parsing {filename}...")
    tic = time()
    with GzipFile(filename) as f:
        df = pd.read_csv(f, header=None, dtype=np.float32)
    toc = time()
    print(f"Loaded {df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")
    return df


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


```
142 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.")
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.

    whom : string
        Who passed X to this function.
    """
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)

```
143 - /tmp/repos/scikit-learn/examples/svm/plot_svm_scale_c.py
```python
clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                       tol=1e-3),
             np.logspace(-2.3, -1.3, 10), X_1, y_1),
            (LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
                       tol=1e-4),
             np.logspace(-4.5, -2, 10), X_2, y_2)]

colors = ['navy', 'cyan', 'darkorange']
lw = 2

for fignum, (clf, cs, X, y) in enumerate(clf_sets):
    # set up the plot for each regressor
    plt.figure(fignum, figsize=(9, 10))

    for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]):
        param_grid = dict(C=cs)
        # To get nice curve, we need a large number of iterations to
        # reduce the variance
        grid = GridSearchCV(clf, refit=False, param_grid=param_grid,
                            cv=ShuffleSplit(train_size=train_size,
                                            n_splits=250, random_state=1))
        grid.fit(X, y)
        scores = grid.cv_results_['mean_test_score']

        scales = [(1, 'No scaling'),
                  ((n_samples * train_size), '1/n_samples'),
                  ]

        for subplotnum, (scaler, name) in enumerate(scales):
            plt.subplot(2, 1, subplotnum + 1)
            plt.xlabel('C')
            plt.ylabel('CV Score')
            grid_cs = cs * float(scaler)  # scale the C's
            plt.semilogx(grid_cs, scores, label="fraction %.2f" %
                         train_size, color=colors[k], lw=lw)
            plt.title('scaling=%s, penalty=%s, loss=%s' %
                      (name, clf.penalty, clf.loss))

    plt.legend(loc="best")
plt.show()

```
144 - /tmp/repos/scikit-learn/sklearn/__check_build/__init__.py
```python
""" Module to give helpful messages to the user that did not
compile scikit-learn properly.
"""
import os

INPLACE_MSG = """
It appears that you are importing a local scikit-learn source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""


def raise_build_error(e):
    # Raise a comprehensible error and list the contents of the
    # directory to help debugging on the mailing list.
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == "sklearn/__check_build":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = INPLACE_MSG
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if ((i + 1) % 3):
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + '\n')
    raise ImportError("""%s
___________________________________________________________________________
Contents of %s:
%s
___________________________________________________________________________
It seems that scikit-learn has not been built correctly.

If you have installed scikit-learn from source, please do not forget
to build the package before using it: run `python setup.py install` or
`make` in the source directory.
%s""" % (e, local_dir, ''.join(dir_content).strip(), msg))

try:
    from ._check_build import check_build  # noqa
except ImportError as e:
    raise_build_error(e)

```
145 - /tmp/repos/scikit-learn/examples/cluster/plot_adjusted_for_chance_measures.py
```python
"""
==========================================================
Adjustment for chance in clustering performance evaluation
==========================================================

The following plots demonstrate the impact of the number of clusters and
number of samples on various clustering performance evaluation metrics.

Non-adjusted measures such as the V-Measure show a dependency between
the number of clusters and the number of samples: the mean V-Measure
of random labeling increases significantly as the number of clusters is
closer to the total number of samples used to compute the measure.

Adjusted for chance measure such as ARI display some random variations
centered around a mean score of 0.0 for any number of samples and
clusters.

Only adjusted measures can hence safely be used as a consensus index
to evaluate the average stability of clustering algorithms for a given
value of k on various overlapping sub-samples of the dataset.

"""
print(__doc__)

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics


def uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                             fixed_n_classes=None, n_runs=5, seed=42):
    """Compute score for 2 random uniform cluster labelings.

    Both random labelings have the same number of clusters for each value
    possible value in ``n_clusters_range``.

    When fixed_n_classes is not None the first labeling is considered a ground
    truth class assignment with fixed number of classes.
    """
    random_labels = np.random.RandomState(seed).randint
    scores = np.zeros((len(n_clusters_range), n_runs))

    if fixed_n_classes is not None:
        labels_a = random_labels(low=0, high=fixed_n_classes, size=n_samples)

    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            if fixed_n_classes is None:
                labels_a = random_labels(low=0, high=k, size=n_samples)
            labels_b = random_labels(low=0, high=k, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores

score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score,
]

# 2 independent random clusterings with equal cluster number

n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(np.int)

plt.figure(1)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, np.median(scores, axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for 2 random uniform labelings\n"
          "with equal number of clusters")
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.legend(plots, names)
plt.ylim(ymin=-0.05, ymax=1.05)


# Random labeling with varying n_clusters against ground class labels
# with fixed number of clusters

n_samples = 1000
n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
n_classes = 10

plt.figure(2)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                                      fixed_n_classes=n_classes)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for random uniform labeling\n"
          "against reference assignment with %d classes" % n_classes)
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.ylim(ymin=-0.05, ymax=1.05)
plt.legend(plots, names)
plt.show()

```
146 - /tmp/repos/scikit-learn/sklearn/utils/_scipy_sparse_lsqr_backport.py
```python
"""
    A = aslinearoperator(A)
    if len(b.shape) > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
         'Ax - b is small enough, given atol, btol                  ',
         'The least-squares solution is good enough, given atol     ',
         'The estimate of cond(Abar) has exceeded conlim            ',
         'Ax - b is small enough for this machine                   ',
         'The least-squares solution is good enough for this machine',
         'Cond(Abar) seems to be too large for this machine         ',
         'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
        str2 = 'damp = %20.14e   calc_var = %8g' % (damp, calc_var)
        str3 = 'atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
        str4 = 'btol = %8.2e               iter_lim = %8g' % (btol, iter_lim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    nstop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    """
    Set up the first vectors u and v for the bidiagonalization.
    These satisfy  beta*u = b,  alfa*v = A'u.
    """
    __xm = np.zeros(m)  # a matrix for temporary holding
    __xn = np.zeros(n)  # a matrix for temporary holding
    v = np.zeros(n)
    u = b
    x = np.zeros(n)
    alfa = 0
    beta = np.linalg.norm(u)
    w = np.zeros(n)

    if beta > 0:
        u = (1/beta) * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)

    if alfa > 0:
        v = (1/alfa) * v
        w = v.copy()

    rhobar = alfa
    phibar = beta
    bnorm = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(str1, str2, str3)

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        """
        %     Perform the next step of the bidiagonalization to obtain the
        %     next  beta, u, alfa, v.  These satisfy the relations
        %                beta*u  =  a*v   -  alfa*u,
        %                alfa*v  =  A'*u  -  beta*v.
        """
        u = A.matvec(v) - alfa * u
        beta = np.linalg.norm(u)

        if beta > 0:
            u = (1/beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            v = A.rmatvec(u) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = (1 / alfa) * v

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk)**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        
```
147 - /tmp/repos/scikit-learn/examples/preprocessing/plot_discretization_classification.py
```python
for ds_cnt, (X, y) in enumerate(datasets):
    print('\ndataset %d\n---------' % ds_cnt)

    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.5, random_state=42)

    # create the grid for background colors
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot the dataset first
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    # iterate over classifiers
    for est_idx, (name, (estimator, param_grid)) in \
            enumerate(zip(names, classifiers)):
        ax = axes[ds_cnt, est_idx + 1]

        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('%s: %.2f' % (name, score))

        # plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]*[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(name.replace(' + ', '\n'))
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0'), size=15,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                transform=ax.transAxes, horizontalalignment='right')


plt.tight_layout()

# Add suptitles above the figure
plt.subplots_adjust(top=0.90)
suptitles = [
    'Linear classifiers',
    'Feature discretization and linear classifiers',
    'Non-linear classifiers',
]
for i, suptitle in zip([1, 3, 5], suptitles):
    ax = axes[0, i]
    ax.text(1.05, 1.25, suptitle, transform=ax.transAxes,
            horizontalalignment='center', size='x-large')
plt.show()

```
148 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
```python
if self.optimizer == 'fmin_cobyla':

            def minus_reduced_likelihood_function(log10t):
                return - self.reduced_likelihood_function(
                    theta=10. ** log10t)[0]

            constraints = []
            for i in range(self.theta0.size):
                constraints.append(lambda log10t, i=i:
                                   log10t[i] - np.log10(self.thetaL[0, i]))
                constraints.append(lambda log10t, i=i:
                                   np.log10(self.thetaU[0, i]) - log10t[i])

            for k in range(self.random_start):

                if k == 0:
                    # Use specified starting point as first guess
                    theta0 = self.theta0
                else:
                    # Generate a random starting point log10-uniformly
                    # distributed between bounds
                    log10theta0 = (np.log10(self.thetaL)
                                   + self.random_state.rand(*self.theta0.shape)
                                   * np.log10(self.thetaU / self.thetaL))
                    theta0 = 10. ** log10theta0

                # Run Cobyla
                try:
                    log10_optimal_theta = \
                        optimize.fmin_cobyla(minus_reduced_likelihood_function,
                                             np.log10(theta0).ravel(),
                                             constraints, disp=0)
                except ValueError as ve:
                    print("Optimization failed. Try increasing the ``nugget``")
                    raise ve

                optimal_theta = 10. ** log10_optimal_theta
                optimal_rlf_value, optimal_par = \
                    self.reduced_likelihood_function(theta=optimal_theta)

                # Compare the new optimizer to the best previous one
                if k > 0:
                    if optimal_rlf_value > best_optimal_rlf_value:
                        best_optimal_rlf_value = optimal_rlf_value
                        best_optimal_par = optimal_par
                        best_optimal_theta = optimal_theta
                else:
                    best_optimal_rlf_value = optimal_rlf_value
                    best_optimal_par = optimal_par
                    best_optimal_theta = optimal_theta
                if self.verbose and self.random_start > 1:
                    if (20 * k) / self.random_start > percent_completed:
                        percent_completed = (20 * k) / self.random_start
                        print("%s completed" % (5 * percent_completed))

            optimal_rlf_value = best_optimal_rlf_value
            optimal_par = best_optimal_par
            optimal_theta = best_optimal_theta

        elif self.optimizer == 'Welch':

            # Backup of the given attributes
            theta0, thetaL, thetaU = self.theta0, self.thetaL, self.thetaU
            corr = self.corr
            verbose = self.verbose

            # This will iterate over fmin_cobyla optimizer
            self.optimizer = 'fmin_cobyla'
            self.verbose = False

            # Initialize under isotropy assumption
            if verbose:
                print("Initialize under isotropy assumption...")
            self.theta0 = check_array(self.theta0.min())
            self.thetaL = check_array(self.thetaL.min())
            self.thetaU = check_array(self.thetaU.max())
            theta_iso, optimal_rlf_value_iso, par_iso = \
                self._arg_max_reduced_likelihood_function()
            optimal_theta = theta_iso + np.zeros(theta0.shape)

            # Iterate over all dimensions of theta allowing for anisotropy
            if verbose:
                print("Now improving allowing for anisotropy...")
            for i in self.random_state.permutation(theta0.size):
                if verbose:
                    print("Proceeding along dimension %d..." % (i + 1))
                self.theta0 = check_array(theta_iso)
                self.thetaL = check_array(thetaL[0, i])
                self.thetaU = check_array(thetaU[0, i])

                def corr_cut(t, d):
                    return corr(check_array(np.hstack([optimal_theta[0][0:i],
                                                       t[0],
                                                       optimal_theta[0][(i +
                                                                         1)::]])),
                                d)

                self.corr = corr_cut
                optimal_theta[0, i], optimal_rlf_value, optimal_par = \
                    self._arg_max_reduced_likelihood_function()

            # Restore the given attributes
            self.theta0, self.thetaL, self.thetaU = theta0, thetaL, thetaU
            self.corr = corr
            self.optimizer = 'Welch'
            self.verbose = verbose

        else:

            raise NotImplementedError("This optimizer ('%s') is not "
                                      "implemented yet. Please contribute!"
                                      % self.optimizer)

        return optimal_theta, optimal_rlf_value, optimal_par

    
```
149 - /tmp/repos/scikit-learn/examples/cluster/plot_kmeans_stability_low_dim_dense.py
```python
"""
============================================================
Empirical evaluation of the impact of k-means initialization
============================================================

Evaluate the ability of k-means initializations strategies to make
the algorithm convergence robust as measured by the relative standard
deviation of the inertia of the clustering (i.e. the sum of squared
distances to the nearest cluster center).

The first plot shows the best inertia reached for each combination
of the model (``KMeans`` or ``MiniBatchKMeans``) and the init method
(``init="random"`` or ``init="kmeans++"``) for increasing values of the
``n_init`` parameter that controls the number of initializations.

The second plot demonstrate one single run of the ``MiniBatchKMeans``
estimator using a ``init="random"`` and ``n_init=1``. This run leads to
a bad convergence (local optimum) with estimated centers stuck
between ground truth clusters.

The dataset used for evaluation is a 2D grid of isotropic Gaussian
clusters widely spaced.
"""
print(__doc__)

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

random_state = np.random.RandomState(0)

# Number of run (with randomly generated dataset) for each strategy so as
# to be able to compute an estimate of the standard deviation
n_runs = 5

# k-means models can do several random inits so as to be able to trade
# CPU time for convergence robustness
n_init_range = np.array([1, 5, 10, 15, 20])

# Datasets generation parameters
n_samples_per_center = 100
grid_size = 3
scale = 0.1
n_clusters = grid_size ** 2


def make_data(random_state, n_samples_per_center, grid_size, scale):
    random_state = check_random_state(random_state)
    centers = np.array([[i, j]
                        for i in range(grid_size)
                        for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1]))

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)
```
150 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
n_features = X.shape[1]
    n_samples = y.size
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    # We are initializing this to "zeros" and not empty, because
    # it is passed to scipy linalg functions and thus if it has NaNs,
    # even if they are in the upper part that it not used, we
    # get errors raised.
    # Once we support only scipy > 0.12 we can use check_finite=False and
    # go back to "empty"
    L = np.zeros((max_features, max_features), dtype=X.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (X,))

    if Gram is None or Gram is False:
        Gram = None
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')

    elif isinstance(Gram, string_types) and Gram == 'auto' or Gram is True:
        if Gram is True or X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
        else:
            Gram = None
    elif copy_Gram:
        Gram = Gram.copy()

    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning
    equality_tolerance = np.finfo(np.float32).eps

    while True:
        if Cov.size:
            if positive:
                C_idx = np.argmax(Cov)
            else:
                C_idx = np.argmax(np.abs(Cov))

            C_ = Cov[C_idx]

            if positive:
                C = C_
            else:
                C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
        if alpha[0] <= alpha_min + equality_tolerance:  # early stopping
            if abs(alpha[0] - alpha_min) > equality_tolerance:
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        
```
151 - /tmp/repos/scikit-learn/sklearn/linear_model/coordinate_descent.py
```python
for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if not multi_output and sparse.isspmatrix(X):
            model = cd_fast.sparse_enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol, rng, random, positive)
        elif multi_output:
            model = cd_fast.enet_coordinate_descent_multi_task(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=X.dtype.type,
                                         order='C')
            model = cd_fast.enet_coordinate_descent_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
                tol, rng, random, positive)
        elif precompute is False:
            model = cd_fast.enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
                positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


###############################################################################
# ElasticNet model


class ElasticNet(LinearModel, RegressorMixin):
    
```
152 - /tmp/repos/scikit-learn/examples/ensemble/plot_adaboost_hastie_10_2.py
```python
"""
=============================
Discrete versus Real AdaBoost
=============================

This example is based on Figure 10.2 from Hastie et al 2009 [1]_ and
illustrates the difference in performance between the discrete SAMME [2]_
boosting algorithm and real SAMME.R boosting algorithm. Both algorithms are
evaluated on a binary classification task where the target Y is a non-linear
function of 10 input features.

Discrete SAMME AdaBoost adapts based on errors in predicted class labels
whereas real SAMME.R uses the predicted class probabilities.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
    Learning Ed. 2", Springer, 2009.

.. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

"""
print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>,
#         Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier


n_estimators = 400
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label='Discrete AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label='Discrete AdaBoost Train Error',
        color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.show()
```
153 - /tmp/repos/scikit-learn/examples/cluster/plot_ward_structured_vs_unstructured.py
```python
"""
===========================================================
Hierarchical clustering: structured vs unstructured ward
===========================================================

Example builds a swiss roll dataset and runs
hierarchical clustering on their position.

For more information, see :ref:`hierarchical_clustering`.

In a first step, the hierarchical clustering is performed without connectivity
constraints on the structure and is solely based on distance, whereas in
a second step the clustering is restricted to the k-Nearest Neighbors
graph: it's a hierarchical clustering with structure prior.

Some of the clusters learned without connectivity constraints do not
respect the structure of the swiss roll and extend across different folds of
the manifolds. On the opposite, when opposing connectivity constraints,
the clusters form a nice parcellation of the swiss roll.
"""

# Authors : Vincent Michel, 2010
#           Alexandre Gramfort, 2010
#           Gael Varoquaux, 2010
# License: BSD 3 clause

print(__doc__)

import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll

# #############################################################################
# Generate data (swiss roll dataset)
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise)
# Make it thinner
X[:, 1] *= .5

# #############################################################################
# Compute clustering
print("Compute unstructured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# #############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)


# #############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# #############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# #############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)

plt.show()
```
154 - /tmp/repos/scikit-learn/examples/calibration/plot_calibration_curve.py
```python
"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to display
how well calibrated the predicted probabilities are and how to calibrate an
uncalibrated classifier.

The experiment is performed on an artificial dataset for binary classification
with 100.000 samples (1.000 of them are used for model fitting) with 20
features. Of the 20 features, only 2 are informative and 10 are redundant. The
first figure shows the estimated probabilities obtained with logistic
regression, Gaussian naive Bayes, and Gaussian naive Bayes with both isotonic
calibration and sigmoid calibration. The calibration performance is evaluated
with Brier score, reported in the legend (the smaller the better). One can
observe here that logistic regression is well calibrated while raw Gaussian
naive Bayes performs very badly. This is because of the redundant features
which violate the assumption of feature-independence and result in an overly
confident classifier, which is indicated by the typical transposed-sigmoid
curve.

Calibration of the probabilities of Gaussian naive Bayes with isotonic
regression can fix this issue as can be seen from the nearly diagonal
calibration curve. Sigmoid calibration also improves the brier score slightly,
albeit not as strongly as the non-parametric isotonic regression. This can be
attributed to the fact that we have plenty of calibration data such that the
greater flexibility of the non-parametric model can be exploited.

The second figure shows the calibration curve of a linear support-vector
classifier (LinearSVC). LinearSVC shows the opposite behavior as Gaussian
naive Bayes: the calibration curve has a sigmoid curve, which is typical for
an under-confident classifier. In the case of LinearSVC, this is caused by the
margin property of the hinge loss, which lets the model focus on hard samples
that are close to the decision boundary (the support vectors).

Both kinds of calibration can fix this issue and yield nearly identical
results. This shows that sigmoid calibration can deal with situations where
the calibration curve of the base classifier is sigmoid (e.g., for LinearSVC)
but not where it is transposed-sigmoid (e.g., Gaussian naive Bayes).
"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split


# Create dataset of classification task with many redundant and few
# informative features
X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                    random_state=42)



```
155 - /tmp/repos/scikit-learn/examples/text/plot_document_clustering.py
```python



# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

```
156 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
"""
=========================================================
Imputing missing values with variants of IterativeImputer
=========================================================

The :class:`sklearn.impute.IterativeImputer` class is very flexible - it can be
used with a variety of estimators to do round-robin regression, treating every
variable as an output in turn.

In this example we compare some estimators for the purpose of missing feature
imputation with :class:`sklearn.impute.IterativeImputer`:

* :class:`~sklearn.linear_model.BayesianRidge`: regularized linear regression
* :class:`~sklearn.tree.DecisionTreeRegressor`: non-linear regression
* :class:`~sklearn.ensemble.ExtraTreesRegressor`: similar to missForest in R
* :class:`~sklearn.neighbors.KNeighborsRegressor`: comparable to other KNN
  imputation approaches

Of particular interest is the ability of
:class:`sklearn.impute.IterativeImputer` to mimic the behavior of missForest, a
popular imputation package for R. In this example, we have chosen to use
:class:`sklearn.ensemble.ExtraTreesRegressor` instead of
:class:`sklearn.ensemble.RandomForestRegressor` (as in missForest) due to its
increased speed.

Note that :class:`sklearn.neighbors.KNeighborsRegressor` is different from KNN
imputation, which learns from samples with missing values by using a distance
metric that accounts for missing values, rather than imputing them.

The goal is to compare different estimators to see which one is best for the
:class:`sklearn.impute.IterativeImputer` when using a
:class:`sklearn.linear_model.BayesianRidge` estimator on the California housing
dataset with a single value randomly removed from each row.

For this particular pattern of missing values we see that
:class:`sklearn.ensemble.ExtraTreesRegressor` and
:class:`sklearn.linear_model.BayesianRidge` give the best results.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

N_SPLITS = 5

rng = np.random.RandomState(0)

X_full, y_full = fetch_california_housing(return_X_y=True)
# ~2k samples is enough for the purpose of the example.
# Remove the following two lines for a slower run with different error bars.
X_full = X_full[::10]
y_full = y_full[::10]
n_samples, n_features = X_full.shape

# Estimate the score on the entire dataset, with no missing values
br_estimator = BayesianRidge()
score_full_data = pd.DataFrame(
    cross_val_score(
        br_estimator, X_full, y_full, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    ),
    columns=['Full Data']
)

# Add a single missing value to each row
X_missing = X_full.copy()
y_missing = y_full
missing_samples = np.arange(n_samples)
missing_features = rng.choice(n_features, n_samples, replace=True)
X_missing[missing_samples, missing_features] = np.nan

# Estimate the score after imputation (mean and median strategies)
score_simple_imputer = pd.DataFrame()
for strategy in ('mean', 'median'):
    estimator = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy=strategy),
        br_estimator
    )
    score_simple_imputer[strategy] = cross_val_score(
        estimator, X_missing, y_missing, scoring='neg_mean_squared_error',
        cv=N_SPLITS
    )

# Estimate the score after iterative imputation of the missing values
# with different estimators
estimators = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features='sqrt', random_state=0),
    ExtraTreesRegressor(n_estimators=10, random_state=0),
    KNeighborsRegressor(n_neighbors=15)
]
score_iterative_imputer = pd.DataFrame()

```
157 - /tmp/repos/scikit-learn/benchmarks/bench_plot_randomized_svd.py
```python
def bench_c(datasets, n_comps):
    all_time = defaultdict(list)
    if enable_spectral_norm:
        all_spectral = defaultdict(list)
    all_frobenius = defaultdict(list)

    for dataset_name in datasets:
        X = get_data(dataset_name)
        if X is None:
            continue

        if enable_spectral_norm:
            X_spectral_norm = norm_diff(X, norm=2, msg=False)
        X_fro_norm = norm_diff(X, norm='fro', msg=False)
        n_comps = np.minimum(n_comps, np.min(X.shape))

        label = "sklearn"
        print("%s %d x %d - %s" %
              (dataset_name, X.shape[0], X.shape[1], label))
        U, s, V, time = svd_timing(X, n_comps, n_iter=2, n_oversamples=10,
                                   method=label)

        all_time[label].append(time)
        if enable_spectral_norm:
            A = U.dot(np.diag(s).dot(V))
            all_spectral[label].append(norm_diff(X - A, norm=2) /
                                       X_spectral_norm)
        f = scalable_frobenius_norm_discrepancy(X, U, s, V)
        all_frobenius[label].append(f / X_fro_norm)

        if fbpca_available:
            label = "fbpca"
            print("%s %d x %d - %s" %
                  (dataset_name, X.shape[0], X.shape[1], label))
            U, s, V, time = svd_timing(X, n_comps, n_iter=2, n_oversamples=2,
                                       method=label)
            all_time[label].append(time)
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(norm_diff(X - A, norm=2) /
                                           X_spectral_norm)
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

    if len(all_time) == 0:
        raise ValueError("No tests ran. Aborting.")

    if enable_spectral_norm:
        title = "normalized spectral norm diff vs running time"
        scatter_time_vs_s(all_time, all_spectral, datasets, title)
    title = "normalized Frobenius norm diff vs running time"
    scatter_time_vs_s(all_time, all_frobenius, datasets, title)


if __name__ == '__main__':
    random_state = check_random_state(1234)

    power_iter = np.linspace(0, 6, 7, dtype=int)
    n_comps = 50

    for dataset_name in datasets:
        X = get_data(dataset_name)
        if X is None:
            continue
        print(" >>>>>> Benching sklearn and fbpca on %s %d x %d" %
              (dataset_name, X.shape[0], X.shape[1]))
        bench_a(X, dataset_name, power_iter, n_oversamples=2,
                n_comps=np.minimum(n_comps, np.min(X.shape)))

    print(" >>>>>> Benching on simulated low rank matrix with variable rank")
    bench_b(power_iter)

    print(" >>>>>> Benching sklearn and fbpca default configurations")
    bench_c(datasets + big_sparse_datasets, n_comps)

    plt.show()

```
158 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python


        for iteration in range(begin_at_stage, self.max_iter):

            if self.verbose:
                iteration_start_time = time()
                print("[{}/{}] ".format(iteration + 1, self.max_iter),
                      end='', flush=True)

            # Update gradients and hessians, inplace
            self.loss_.update_gradients_and_hessians(gradients, hessians,
                                                     y_train, raw_predictions)

            # Append a list since there may be more than 1 predictor per iter
            predictors.append([])

            # Build `n_trees_per_iteration` trees.
            for k in range(self.n_trees_per_iteration_):

                grower = TreeGrower(
                    X_binned_train, gradients[k, :], hessians[k, :],
                    max_bins=self.max_bins,
                    actual_n_bins=self.bin_mapper_.actual_n_bins_,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    shrinkage=self.learning_rate)
                grower.grow()

                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time
                acc_compute_hist_time += grower.total_compute_hist_time

                predictor = grower.make_predictor(
                    bin_thresholds=self.bin_mapper_.bin_thresholds_
                )
                predictors[-1].append(predictor)

                # Update raw_predictions with the predictions of the newly
                # created tree.
                tic_pred = time()
                _update_raw_predictions(raw_predictions[k, :], grower)
                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred

            should_early_stop = False
            if self.do_early_stopping_:
                if self.scoring == 'loss':
                    # Update raw_predictions_val with the newest tree(s)
                    if self._use_validation_data:
                        for k, pred in enumerate(self._predictors[-1]):
                            raw_predictions_val[k, :] += (
                                pred.predict_binned(X_binned_val))

                    should_early_stop = self._check_early_stopping_loss(
                        raw_predictions, y_train,
                        raw_predictions_val, y_val
                    )

                else:
                    should_early_stop = self._check_early_stopping_scorer(
                        X_binned_small_train, y_small_train,
                        X_binned_val, y_val,
                    )

            if self.verbose:
                self._print_iteration_stats(iteration_start_time)

            # maybe we could also early stop if all the trees are stumps?
            if should_early_stop:
                break

        if self.verbose:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self._predictors
                for predictor in predictors_at_ith_iteration
            )
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self._predictors)
            print("Fit {} trees in {:.3f} s, ({} total leaves)".format(
                n_predictors, duration, n_total_leaves))
            print("{:<32} {:.3f}s".format('Time spent computing histograms:',
                                          acc_compute_hist_time))
            print("{:<32} {:.3f}s".format('Time spent finding best splits:',
                                          acc_find_split_time))
            print("{:<32} {:.3f}s".format('Time spent applying splits:',
                                          acc_apply_split_time))
            print("{:<32} {:.3f}s".format('Time spent predicting:',
                                          acc_prediction_time))

        self.train_score_ = np.asarray(self.train_score_)
        self.validation_score_ = np.asarray(self.validation_score_)
        del self._in_fit  # hard delete so we're sure it can't be used anymore
        return self

    def _is_fitted(self):
        return len(getattr(self, '_predictors', [])) > 0

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        for var in ('train_score_', 'validation_score_', '_rng'):
            if hasattr(self, var):
                delattr(self, var)

    
```
159 - /tmp/repos/scikit-learn/examples/linear_model/plot_sgd_early_stopping.py
```python



@ignore_warnings(category=ConvergenceWarning)
def fit_and_score(estimator, max_iter, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.set_params(max_iter=max_iter)
    estimator.set_params(random_state=0)

    start = time.time()
    estimator.fit(X_train, y_train)

    fit_time = time.time() - start
    n_iter = estimator.n_iter_
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)

    return fit_time, n_iter, train_score, test_score


# Define the estimators to compare
estimator_dict = {
    'No stopping criterion':
    linear_model.SGDClassifier(tol=1e-3, n_iter_no_change=3),
    'Training loss':
    linear_model.SGDClassifier(early_stopping=False, n_iter_no_change=3,
                               tol=0.1),
    'Validation score':
    linear_model.SGDClassifier(early_stopping=True, n_iter_no_change=3,
                               tol=0.0001, validation_fraction=0.2)
}

# Load the dataset
X, y = load_mnist(n_samples=10000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=0)

results = []
for estimator_name, estimator in estimator_dict.items():
    print(estimator_name + ': ', end='')
    for max_iter in range(1, 50):
        print('.', end='')
        sys.stdout.flush()

        fit_time, n_iter, train_score, test_score = fit_and_score(
            estimator, max_iter, X_train, X_test, y_train, y_test)

        results.append((estimator_name, max_iter, fit_time, n_iter,
                        train_score, test_score))
    print('')

# Transform the results in a pandas dataframe for easy plotting
columns = [
    'Stopping criterion', 'max_iter', 'Fit time (sec)', 'n_iter_',
    'Train score', 'Test score'
]
results_df = pd.DataFrame(results, columns=columns)

# Define what to plot (x_axis, y_axis)
lines = 'Stopping criterion'
plot_list = [
    ('max_iter', 'Train score'),
    ('max_iter', 'Test score'),
    ('max_iter', 'n_iter_'),
    ('max_iter', 'Fit time (sec)'),
]

nrows = 2
ncols = int(np.ceil(len(plot_list) / 2.))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols,
                                                            4 * nrows))
axes[0, 0].get_shared_y_axes().join(axes[0, 0], axes[0, 1])

for ax, (x_axis, y_axis) in zip(axes.ravel(), plot_list):
    for criterion, group_df in results_df.groupby(lines):
        group_df.plot(x=x_axis, y=y_axis, label=criterion, ax=ax)
    ax.set_title(y_axis)
    ax.legend(title=lines)

fig.tight_layout()
plt.show()

```
160 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
"""
The :mod:`sklearn.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.
"""


from .ranking import auc
from .ranking import average_precision_score
from .ranking import coverage_error
from .ranking import label_ranking_average_precision_score
from .ranking import label_ranking_loss
from .ranking import precision_recall_curve
from .ranking import roc_auc_score
from .ranking import roc_curve

from .classification import accuracy_score
from .classification import balanced_accuracy_score
from .classification import classification_report
from .classification import cohen_kappa_score
from .classification import confusion_matrix
from .classification import f1_score
from .classification import fbeta_score
from .classification import hamming_loss
from .classification import hinge_loss
from .classification import jaccard_similarity_score
from .classification import log_loss
from .classification import matthews_corrcoef
from .classification import precision_recall_fscore_support
from .classification import precision_score
from .classification import recall_score
from .classification import zero_one_loss
from .classification import brier_score_loss

from . import cluster
from .cluster import adjusted_mutual_info_score
from .cluster import adjusted_rand_score
from .cluster import completeness_score
from .cluster import consensus_score
from .cluster import homogeneity_completeness_v_measure
from .cluster import homogeneity_score
from .cluster import mutual_info_score
from .cluster import normalized_mutual_info_score
from .cluster import fowlkes_mallows_score
from .cluster import silhouette_samples
from .cluster import silhouette_score
from .cluster import calinski_harabaz_score
from .cluster import v_measure_score

from .pairwise import euclidean_distances
from .pairwise import pairwise_distances
from .pairwise import pairwise_distances_argmin
from .pairwise import pairwise_distances_argmin_min
from .pairwise import pairwise_kernels

from .regression import explained_variance_score
from .regression import mean_absolute_error
from .regression import mean_squared_error
from .regression import mean_squared_log_error
from .regression import median_absolute_error
from .regression import r2_score

from .scorer import check_scoring
from .scorer import make_scorer
from .scorer import SCORERS
from .scorer import get_scorer

__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]
```
161 - /tmp/repos/scikit-learn/benchmarks/bench_20newsgroups.py
```python
from __future__ import print_function, division
from time import time
import argparse
import numpy as np

from sklearn.dummy import DummyClassifier

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ESTIMATORS = {
    "dummy": DummyClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100,
                                            max_features="sqrt",
                                            min_samples_split=10),
    "extra_trees": ExtraTreesClassifier(n_estimators=100,
                                        max_features="sqrt",
                                        min_samples_split=10),
    "logistic_regression": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "adaboost": AdaBoostClassifier(n_estimators=10),
}


###############################################################################
# Data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=ESTIMATORS)
    args = vars(parser.parse_args())

    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = check_array(data_train.data, dtype=np.float32,
                          accept_sparse="csc")
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")
    y_train = data_train.target
    y_test = data_test.target

    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train.shape))
    print("X_train.format = {0}".format(X_train.format))
    print("X_train.dtype = {0}".format(X_train.dtype))
    print("X_train density = {0}"
          "".format(X_train.nnz / np.product(X_train.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test.shape))
    print("X_test.format = {0}".format(X_test.format))
    print("X_test.dtype = {0}".format(X_test.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}
    for name in sorted(args["estimators"]):
        clf = ESTIMATORS[name]
        try:
            clf.set_params(random_state=0)
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time",
                           "Accuracy"))
    print("-" * 44)
    for name in sorted(accuracy, key=accuracy.get):
        print("%s %s %s %s" % (name.ljust(16),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % accuracy[name]).center(10)))

    print()
```
162 - /tmp/repos/scikit-learn/sklearn/linear_model/coordinate_descent.py
```python
if check_input:
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=X.dtype.type, order='C', copy=False,
                             ensure_2d=False)

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    if multi_output and positive:
        raise ValueError('positive=True is not allowed for multi-output'
                         ' (y.ndim != 1)')

    # MultiTaskElasticNet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=X.dtype)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1], dtype=X.dtype))
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    
```
163 - /tmp/repos/scikit-learn/benchmarks/bench_plot_fastkmeans.py
```python



if __name__ == '__main__':
    from mpl_toolkits.mplot3d import axes3d  # register the 3d projection
    import matplotlib.pyplot as plt

    samples_range = np.linspace(50, 150, 5).astype(np.int)
    features_range = np.linspace(150, 50000, 5).astype(np.int)
    chunks = np.linspace(500, 10000, 15).astype(np.int)

    results = compute_bench(samples_range, features_range)
    results_2 = compute_bench_2(chunks)

    max_time = max([max(i) for i in [t for (label, t) in six.iteritems(results)
                                     if "speed" in label]])
    max_inertia = max([max(i) for i in [
        t for (label, t) in six.iteritems(results)
        if "speed" not in label]])

    fig = plt.figure('scikit-learn K-Means benchmark results')
    for c, (label, timings) in zip('brcy',
                                   sorted(six.iteritems(results))):
        if 'speed' in label:
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            ax.set_zlim3d(0.0, max_time * 1.1)
        else:
            ax = fig.add_subplot(2, 2, 2, projection='3d')
            ax.set_zlim3d(0.0, max_inertia * 1.1)

        X, Y = np.meshgrid(samples_range, features_range)
        Z = np.asarray(timings).reshape(samples_range.shape[0],
                                        features_range.shape[0])
        ax.plot_surface(X, Y, Z.T, cstride=1, rstride=1, color=c, alpha=0.5)
        ax.set_xlabel('n_samples')
        ax.set_ylabel('n_features')

    i = 0
    for c, (label, timings) in zip('br',
                                   sorted(six.iteritems(results_2))):
        i += 1
        ax = fig.add_subplot(2, 2, i + 2)
        y = np.asarray(timings)
        ax.plot(chunks, y, color=c, alpha=0.8)
        ax.set_xlabel('Chunks')
        ax.set_ylabel(label)

    plt.show()

```
164 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
"""K-means clustering"""

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
# License: BSD 3 clause

import warnings

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import pairwise_distances_argmin_min
from ..utils.extmath import row_norms, squared_norm, stable_cumsum
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _num_samples
from ..utils import check_array
from ..utils import check_random_state
from ..utils import as_float_array
from ..utils import gen_batches
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES
from ..externals.joblib import Parallel
from ..externals.joblib import delayed
from ..externals.six import string_types
from ..exceptions import ConvergenceWarning
from . import _k_means
from ._k_means_elkan import k_means_elkan


###############################################################################
# Initialization heuristic



```
**165 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
"""
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    _check_solver_option(solver, multi_class, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)

    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == 'multinomial':
        class_weight_ = compute_class_weight(class_weight, classes, y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver not in ['sag', 'saga']:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      order='F', dtype=X.dtype)

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))
            w0[:, :coef.shape[1]] = coef

    
```
166 - /tmp/repos/scikit-learn/sklearn/metrics/classification.py
```python
if y_true_unique.size > 2:
        if (labels is None and pred_decision.ndim > 1 and
                (np.size(y_true_unique) != pred_decision.shape[1])):
            raise ValueError("Please include all labels in y_true "
                             "or pass labels as third argument")
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1),
                         axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = column_or_1d(pred_decision)
        pred_decision = np.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    np.clip(losses, 0, None, out=losses)
    return np.average(losses, weights=sample_weight)



```
167 - /tmp/repos/scikit-learn/sklearn/cluster/__init__.py
```python
"""
The :mod:`sklearn.cluster` module gathers popular unsupervised clustering
algorithms.
"""

from .spectral import spectral_clustering, SpectralClustering
from .mean_shift_ import (mean_shift, MeanShift,
                          estimate_bandwidth, get_bin_seeds)
from .affinity_propagation_ import affinity_propagation, AffinityPropagation
from .hierarchical import (ward_tree, AgglomerativeClustering, linkage_tree,
                           FeatureAgglomeration)
from .k_means_ import k_means, KMeans, MiniBatchKMeans
from .dbscan_ import dbscan, DBSCAN
from .bicluster import SpectralBiclustering, SpectralCoclustering
from .birch import Birch

__all__ = ['AffinityPropagation',
           'AgglomerativeClustering',
           'Birch',
           'DBSCAN',
           'KMeans',
           'FeatureAgglomeration',
           'MeanShift',
           'MiniBatchKMeans',
           'SpectralClustering',
           'affinity_propagation',
           'dbscan',
           'estimate_bandwidth',
           'get_bin_seeds',
           'k_means',
           'linkage_tree',
           'mean_shift',
           'spectral_clustering',
           'ward_tree',
           'SpectralBiclustering',
           'SpectralCoclustering']
```
168 - /tmp/repos/scikit-learn/sklearn/base.py
```python
class BiclusterMixin(object):
    """Mixin class for all bicluster estimators in scikit-learn"""

    @property
    def biclusters_(self):
        """Convenient way to get row and column indicators together.

        Returns the ``rows_`` and ``columns_`` members.
        """
        return self.rows_, self.columns_

    def get_indices(self, i):
        """Row and column indices of the i'th bicluster.

        Only works if ``rows_`` and ``columns_`` attributes exist.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : np.array, dtype=np.intp
            Indices of rows in the dataset that belong to the bicluster.
        col_ind : np.array, dtype=np.intp
            Indices of columns in the dataset that belong to the bicluster.

        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return np.nonzero(rows)[0], np.nonzero(columns)[0]

    def get_shape(self, i):
        """Shape of the i'th bicluster.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        shape : (int, int)
            Number of rows and columns (resp.) in the bicluster.
        """
        indices = self.get_indices(i)
        return tuple(len(i) for i in indices)

    def get_submatrix(self, i, data):
        """Returns the submatrix corresponding to bicluster `i`.

        Parameters
        ----------
        i : int
            The index of the cluster.
        data : array
            The data.

        Returns
        -------
        submatrix : array
            The submatrix corresponding to bicluster i.

        Notes
        -----
        Works with sparse matrices. Only works if ``rows_`` and
        ``columns_`` attributes exist.
        """
        from .utils.validation import check_array
        data = check_array(data, accept_sparse='csr')
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]


###############################################################################
class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class DensityMixin(object):
    """Mixin class for all density estimators in scikit-learn."""
    _estimator_type = "DensityEstimator"

    def score(self, X, y=None):
        """Returns the score of the model on the data X

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        score : float
        """
        pass


class OutlierMixin(object):
    """Mixin class for all outlier detection estimators in scikit-learn."""
    _estimator_type = "outlier_detector"

    def fit_predict(self, X, y=None):
        """Performs outlier detection on X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X).predict(X)


###############################################################################
class MetaEstimatorMixin(object):
    """Mixin class for all meta estimators in scikit-learn."""
    # this is just a tag for the moment


###############################################################################


```
169 - /tmp/repos/scikit-learn/examples/manifold/plot_compare_methods.py
```python
"""
=========================================
 Comparison of Manifold Learning methods
=========================================

An illustration of dimensionality reduction on the S-curve dataset
with various manifold learning methods.

For a discussion and comparison of these algorithms, see the
:ref:`manifold module page <manifold>`

For a similar example, where the methods are applied to a
sphere dataset, see :ref:`sphx_glr_auto_examples_manifold_plot_manifold_sphere.py`

Note that the purpose of the MDS is to find a low-dimensional
representation of the data (here 2D) in which the distances respect well
the distances in the original high-dimensional space, unlike other
manifold-learning algorithms, it does not seeks an isotropic
representation of the data in the low-dimensional space.
"""

# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))

```
**170 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
```python
def _logcosh(x, fun_args=None):
    alpha = fun_args.get('alpha', 1.0)  # comment it out?

    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0])
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i ** 2)).mean()
    return gx, g_x


def _exp(x, fun_args):
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)


def _cube(x, fun_args):
    return x ** 3, (3 * x ** 2).mean(axis=-1)


def fastica(X, n_components=None, algorithm="parallel", whiten=True,
            fun="logcosh", fun_args=None, max_iter=200, tol=1e-04, w_init=None,
            random_state=None, return_X_mean=False, compute_sources=True,
            return_n_iter=False):
    
```
171 - /tmp/repos/scikit-learn/benchmarks/bench_isolation_forest.py
```python
for dat in datasets:

    # Loading and vectorizing the data:
    print('====== %s ======' % dat)
    print('--- Fetching data...')
    if dat in ['http', 'smtp', 'SF', 'SA']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True,
                                 percent10=True, random_state=random_state)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y, random_state=random_state)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)
        print('----- ')

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True, random_state=random_state)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)
        print_outlier_ratio(y)

    print('--- Vectorizing data...')

    if dat == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    if dat == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    if dat in ('http', 'smtp'):
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    n_samples, n_features = X.shape
    n_samples_train = n_samples // 2

    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    print('--- Fitting the IsolationForest estimator...')
    model = IsolationForest(n_jobs=-1, random_state=random_state)
    tstart = time()
    model.fit(X_train)
    fit_time = time() - tstart
    tstart = time()

    scoring = - model.decision_function(X_test)  # the lower, the more abnormal

    print("--- Preparing the plot elements...")
    if with_decision_function_histograms:
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        bins = np.linspace(-0.5, 0.5, 200)
        ax[0].hist(scoring, bins, color='black')
        ax[0].set_title('Decision function for %s dataset' % dat)
        ax[1].hist(scoring[y_test == 0], bins, color='b', label='normal data')
        ax[1].legend(loc="lower right")
        ax[2].hist(scoring[y_test == 1], bins, color='r', label='outliers')
        ax[2].legend(loc="lower right")

    # Show ROC Curves
    predict_time = time() - tstart
    fpr, tpr, thresholds = roc_curve(y_test, scoring)
    auc_score = auc(fpr, tpr)
    label = ('%s (AUC: %0.3f, train_time= %0.2fs, '
             'test_time= %0.2fs)' % (dat, auc_score, fit_time, predict_time))
    # Print AUC score and train/test time:
    print(label)
    ax_roc.plot(fpr, tpr, lw=1, label=label)


ax_roc.set_xlim([-0.05, 1.05])
ax_roc.set_ylim([-0.05, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')

```
172 - /tmp/repos/scikit-learn/examples/neural_networks/plot_mlp_training_curves.py
```python
"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.
"""

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# load / generate some toy datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
data_sets = [(iris.data, iris.target),
             (digits.data, digits.target),
             datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
             datasets.make_moons(noise=0.3, random_state=0)]

for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
                                                    'circles', 'moons']):
    plot_on_dataset(*data, ax=ax, name=name)

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()

```
173 - /tmp/repos/scikit-learn/examples/covariance/plot_mahalanobis_distances.py
```python
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
zz = np.c_[xx.ravel(), yy.ravel()]

mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = subfig1.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                  cmap=plt.cm.PuBu_r,
                                  linestyles='dashed')

mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = subfig1.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                 cmap=plt.cm.YlOrBr_r, linestyles='dotted')

subfig1.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
                inlier_plot, outlier_plot],
               ['MLE dist', 'robust dist', 'inliers', 'outliers'],
               loc="upper right", borderaxespad=0)
plt.xticks(())
plt.yticks(())

# Plot the scores for each point
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
subfig2 = plt.subplot(2, 2, 3)
subfig2.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=.25)
subfig2.plot(1.26 * np.ones(n_samples - n_outliers),
             emp_mahal[:-n_outliers], '+k', markeredgewidth=1)
subfig2.plot(2.26 * np.ones(n_outliers),
             emp_mahal[-n_outliers:], '+k', markeredgewidth=1)
subfig2.axes.set_xticklabels(('inliers', 'outliers'), size=15)
subfig2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
subfig2.set_title("1. from non-robust estimates\n(Maximum Likelihood)")
plt.yticks(())

robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
subfig3 = plt.subplot(2, 2, 4)
subfig3.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]],
                widths=.25)
subfig3.plot(1.26 * np.ones(n_samples - n_outliers),
             robust_mahal[:-n_outliers], '+k', markeredgewidth=1)
subfig3.plot(2.26 * np.ones(n_outliers),
             robust_mahal[-n_outliers:], '+k', markeredgewidth=1)
subfig3.axes.set_xticklabels(('inliers', 'outliers'), size=15)
subfig3.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
subfig3.set_title("2. from robust estimates\n(Minimum Covariance Determinant)")
plt.yticks(())

plt.show()

```
174 - /tmp/repos/scikit-learn/examples/model_selection/plot_grid_search_refit_callable.py
```python
"""
==================================================
Balance model complexity and cross-validated score
==================================================

This example balances model complexity and cross-validated score by
finding a decent accuracy within 1 standard deviation of the best accuracy
score while minimising the number of PCA components [1].

The figure shows the trade-off between cross-validated score and the number
of PCA components. The balanced case is when n_components=6 and accuracy=0.80,
which falls into the range within 1 standard deviation of the best accuracy
score.

[1] Hastie, T., Tibshirani, R.,, Friedman, J. (2001). Model Assessment and
Selection. The Elements of Statistical Learning (pp. 219-260). New York,
NY, USA: Springer New York Inc..
"""
# Author: Wenhao Zhang <wenhaoz@ucla.edu>

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def lower_bound(cv_results):
    """
    Calculate the lower bound within 1 standard deviation
    of the best `mean_test_scores`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`

    Returns
    -------
    float
        Lower bound within 1 standard deviation of the
        best `mean_test_score`.
    """
    best_score_idx = np.argmax(cv_results['mean_test_score'])

    return (cv_results['mean_test_score'][best_score_idx]
            - cv_results['std_test_score'][best_score_idx])


def best_low_complexity(cv_results):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.

    Return
    ------
    int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.
    """
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results['mean_test_score'] >= threshold)
    best_idx = candidate_idx[cv_results['param_reduce_dim__n_components']
                             [candidate_idx].argmin()]
    return best_idx


pipe = Pipeline([
        ('reduce_dim', PCA(random_state=42)),
        ('classify', LinearSVC(random_state=42)),
])

param_grid = {
    'reduce_dim__n_components': [2, 4, 6, 8]
}

grid = GridSearchCV(pipe, cv=10, n_jobs=1, param_grid=param_grid,
                    scoring='accuracy', refit=best_low_complexity)
digits = load_digits()
grid.fit(digits.data, digits.target)

n_components = grid.cv_results_['param_reduce_dim__n_components']
test_scores = grid.cv_results_['mean_test_score']

plt.figure()
plt.bar(n_components, test_scores, width=1.3, color='b')

lower = lower_bound(grid.cv_results_)
plt.axhline(np.max(test_scores), linestyle='--', color='y',
            label='Best score')
plt.axhline(lower, linestyle='--', color='.5', label='Best score - 1 std')

plt.title("Balance model complexity and cross-validated score")
plt.xlabel('Number of PCA components used')
plt.ylabel('Digit classification accuracy')
plt.xticks(n_components.tolist())
plt.ylim((0, 1.0))
plt.legend(loc='upper left')

best_index_ = grid.best_index_

print("The best_index_ is %d" % best_index_)
print("The n_components selected is %d" % n_components[best_index_])
print("The corresponding accuracy score is %.2f"
      % grid.cv_results_['mean_test_score'][best_index_])
plt.show()

```
175 - /tmp/repos/scikit-learn/examples/svm/plot_rbf_parameters.py
```python


# #############################################################################
# Load and prepare data set
#
# dataset for grid search

iris = load_iris()
X = iris.data
y = iris.target

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

```
176 - /tmp/repos/scikit-learn/examples/neighbors/plot_lof_novelty_detection.py
```python
"""
=================================================
Novelty detection with Local Outlier Factor (LOF)
=================================================

The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection
method which computes the local density deviation of a given data point with
respect to its neighbors. It considers as outliers the samples that have a
substantially lower density than their neighbors. This example shows how to
use LOF for novelty detection. Note that when LOF is used for novelty
detection you MUST not use predict, decision_function and score_samples on the
training set as this would lead to wrong results. You must only use these
methods on new unseen data (which are not in the training set). See
:ref:`User Guide <outlier_detection>`: for details on the difference between
outlier detection and novelty detection and how to use LOF for outlier
detection.

The number of neighbors considered, (parameter n_neighbors) is typically
set 1) greater than the minimum number of samples a cluster has to contain,
so that other samples can be local outliers relative to this cluster, and 2)
smaller than the maximum number of close by samples that can potentially be
local outliers.
In practice, such informations are generally not available, and taking
n_neighbors=20 appears to work well in general.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

np.random.seed(42)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate normal (not abnormal) training observations
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate new normal (not abnormal) observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model for novelty detection (novelty=True)
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
# DO NOT use predict, decision_function and score_samples on X_train as this
# would give wrong results but only on new unseen data (not used in X_train),
# e.g. X_test, X_outliers or the meshgrid
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the learned frontier, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection with LOF")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_test, n_error_outliers))
plt.show()
```
177 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        for score in self._staged_decision_function(X):
            decisions = self.loss_._score_to_decision(score)
            yield self.classes_.take(decisions, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        score = self.decision_function(X)
        try:
            return self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape = [n_samples]
            The predicted value of the input samples.
        """
        try:
            for score in self._staged_decision_function(X):
                yield self.loss_._score_to_proba(score)
        except NotFittedError:
            raise
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)


class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

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

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

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

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    init : estimator, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           optional parameter *presort*.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20


    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
**178 - /tmp/repos/scikit-learn/sklearn/cross_decomposition/pls_.py**:
```python
for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.algorithm == "nipals":
                x_weights, y_weights, n_iter_ = \
                    _nipals_twoblocks_inner_loop(
                        X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                        tol=self.tol, norm_y_weights=self.norm_y_weights)
                self.n_iter_.append(n_iter_)
            elif self.algorithm == "svd":
                x_weights, y_weights = _svd_cross_product(X=Xk, Y=Yk)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            x_scores = np.dot(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
            # - regress Xk's on x_score
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - subtract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, y_scores)
                              / np.dot(y_scores.T, y_scores))
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, x_scores)
                              / np.dot(x_scores.T, x_scores))
                Yk -= np.dot(x_scores, y_loadings.T)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                      check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        
```
179 - /tmp/repos/scikit-learn/examples/neighbors/plot_lof_outlier_detection.py
```python
"""
=================================================
Outlier detection with Local Outlier Factor (LOF)
=================================================

The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection
method which computes the local density deviation of a given data point with
respect to its neighbors. It considers as outliers the samples that have a
substantially lower density than their neighbors. This example shows how to
use LOF for outlier detection which is the default use case of this estimator
in scikit-learn. Note that when LOF is used for outlier detection it has no
predict, decision_function and score_samples methods. See
:ref:`User Guide <outlier_detection>`: for details on the difference between
outlier detection and novelty detection and how to use LOF for novelty
detection.

The number of neighbors considered (parameter n_neighbors) is typically
set 1) greater than the minimum number of samples a cluster has to contain,
so that other samples can be local outliers relative to this cluster, and 2)
smaller than the maximum number of close by samples that can potentially be
local outliers.
In practice, such informations are generally not available, and taking
n_neighbors=20 appears to work well in general.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```
180 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array


def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    
```
181 - /tmp/repos/scikit-learn/sklearn/externals/_arff.py
```python
class BadNominalValue(ArffException):
    '''Error raised when a value in used in some data instance but is not
    declared into it respective attribute declaration.'''

    def __init__(self, value):
        super(BadNominalValue, self).__init__()
        self.message = (
            ('Data value %s not found in nominal declaration, ' % value)
            + 'at line %d.'
        )

class BadNominalFormatting(ArffException):
    '''Error raised when a nominal value with space is not properly quoted.'''
    def __init__(self, value):
        super(BadNominalFormatting, self).__init__()
        self.message = (
            ('Nominal data value "%s" not properly quoted in line ' % value) +
            '%d.'
        )

class BadNumericalValue(ArffException):
    '''Error raised when and invalid numerical value is used in some data
    instance.'''
    message = 'Invalid numerical value, at line %d.'

class BadStringValue(ArffException):
    '''Error raise when a string contains space but is not quoted.'''
    message = 'Invalid string value at line %d.'

class BadLayout(ArffException):
    '''Error raised when the layout of the ARFF file has something wrong.'''
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        super(BadLayout, self).__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')


class BadObject(ArffException):
    '''Error raised when the object representing the ARFF file has something
    wrong.'''
    def __init__(self, msg='Invalid object.'):
        self.msg = msg

    def __str__(self):
        return '%s' % self.msg

# =============================================================================

# INTERNAL ====================================================================
def _unescape_sub_callback(match):
    return _UNESCAPE_SUB_MAP[match.group()]


def encode_string(s):
    if _RE_QUOTE_CHARS.search(s):
        return u"'%s'" % _RE_ESCAPE_CHARS.sub(_unescape_sub_callback, s)
    return s


class EncodedNominalConversor(object):
    def __init__(self, values):
        self.values = {v: i for i, v in enumerate(values)}
        self.values[0] = 0

    def __call__(self, value):
        try:
            return self.values[value]
        except KeyError:
            raise BadNominalValue(value)


class NominalConversor(object):
    def __init__(self, values):
        self.values = set(values)
        self.zero_value = values[0]

    def __call__(self, value):
        if value not in self.values:
            if value == 0:
                # Sparse decode
                # See issue #52: nominals should take their first value when
                # unspecified in a sparse matrix. Naturally, this is consistent
                # with EncodedNominalConversor.
                return self.zero_value
            raise BadNominalValue(value)
        return unicode(value)



```
182 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python


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
    
```
183 - /tmp/repos/scikit-learn/examples/linear_model/plot_huber_vs_ridge.py
```python
"""
=======================================================
HuberRegressor vs Ridge on dataset with strong outliers
=======================================================

Fit Ridge and HuberRegressor on a dataset with outliers.

The example shows that the predictions in ridge are strongly influenced
by the outliers present in the dataset. The Huber regressor is less
influenced by the outliers since the model uses the linear loss for these.
As the parameter epsilon is increased for the Huber regressor, the decision
function approaches that of the ridge.
"""

# Authors: Manoj Kumar mks542@nyu.edu
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge

# Generate toy data.
rng = np.random.RandomState(0)
X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0,
                       bias=100.0)

# Add four strong outliers to the dataset.
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.
X_outliers[2:, :] += X.min() - X.mean() / 4.
y_outliers[:2] += y.min() - y.mean() / 4.
y_outliers[2:] += y.max() + y.mean() / 4.
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, 'b.')

# Fit the huber regressor over a series of epsilon values.
colors = ['r-', 'b-', 'y-', 'm-']

x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1.35, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                           epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

# Fit a ridge regressor to compare it to huber regressor.
ridge = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, 'g-', label="ridge regression")

plt.title("Comparison of HuberRegressor vs Ridge")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()
```
184 - /tmp/repos/scikit-learn/examples/ensemble/plot_forest_iris.py
```python
for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()

```
185 - /tmp/repos/scikit-learn/benchmarks/bench_tree.py
```python
"""
To run this, you'll need to have installed.

  * scikit-learn

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10 ** 6  # number of microseconds in a second


def bench_scikit_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_classifier_results.append(
        delta.seconds + delta.microseconds / mu_second)


def bench_scikit_tree_regressor(X, Y):
    """Benchmark with scikit-learn decision tree regressor"""

    from sklearn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_regressor_results.append(
        delta.seconds + delta.microseconds / mu_second)


if __name__ == '__main__':

    print('============================================')
    print('Warning: this is going to take a looong time')
    print('============================================')

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print('============================================')
        print('Entering iteration %s of %s' % (i, n))
        print('============================================')
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure('scikit-learn tree benchmark results')
    plt.subplot(211)
    plt.title('Learning with varying number of samples')
    plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
    plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
    plt.legend(loc='upper left')
    plt.xlabel('number of samples')
    plt.ylabel('Time (s)')

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print('============================================')
        print('Entering iteration %s of %s' % (i, n))
        print('============================================')
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title('Learning in high dimensional spaces')
    plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
    plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
    plt.legend(loc='upper left')
    plt.xlabel('number of dimensions')
    plt.ylabel('Time (s)')
    plt.axis('tight')
    plt.show()

```
**186 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
def insert_cf_subcluster(self, subcluster):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
        dist_matrix *= -2.
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(
                subcluster)

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = \
                    self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = \
                    self.subclusters_[closest_index].sq_norm_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_, threshold, branching_factor)
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2)

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(
                subcluster, self.threshold)
            if merged:
                self.init_centroids_[closest_index] = \
                    closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = \
                    closest_subcluster.sq_norm_
                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True



```
187 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
def _validate_shuffle_split_init(test_size, train_size):
    """Validation helper to check the test_size and train_size at init

    NOTE This does not take into account the number of samples which is known
    only at split
    """
    if test_size == "default":
        if train_size is not None:
            warnings.warn("From version 0.21, test_size will always "
                          "complement train_size unless both "
                          "are specified.",
                          FutureWarning)
        test_size = 0.1

    if test_size is None and train_size is None:
        raise ValueError('test_size and train_size can not both be None')

    if test_size is not None:
        if np.asarray(test_size).dtype.kind == 'f':
            if test_size >= 1.:
                raise ValueError(
                    'test_size=%f should be smaller '
                    'than 1.0 or be an integer' % test_size)
        elif np.asarray(test_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for test_size: %r" % test_size)

    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size >= 1.:
                raise ValueError("train_size=%f should be smaller "
                                 "than 1.0 or be an integer" % train_size)
            elif (np.asarray(test_size).dtype.kind == 'f' and
                    (train_size + test_size) > 1.):
                raise ValueError('The sum of test_size and train_size = %f, '
                                 'should be smaller than 1.0. Reduce '
                                 'test_size and/or train_size.' %
                                 (train_size + test_size))
        elif np.asarray(train_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for train_size: %r" % train_size)


def _validate_shuffle_split(n_samples, test_size, train_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if (test_size is not None and
            np.asarray(test_size).dtype.kind == 'i' and
            test_size >= n_samples):
        raise ValueError('test_size=%d should be smaller than the number of '
                         'samples %d' % (test_size, n_samples))

    if (train_size is not None and
            np.asarray(train_size).dtype.kind == 'i' and
            train_size >= n_samples):
        raise ValueError("train_size=%d should be smaller than the number of"
                         " samples %d" % (train_size, n_samples))

    if test_size == "default":
        test_size = 0.1

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = ceil(test_size * n_samples)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif np.asarray(train_size).dtype.kind == 'f':
        n_train = floor(train_size * n_samples)
    else:
        n_train = float(train_size)

    if test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    return int(n_train), int(n_test)



```
188 - /tmp/repos/scikit-learn/sklearn/utils/optimize.py
```python
"""
Our own implementation of the Newton algorithm

Unlike the scipy.optimize version, this version of the Newton conjugate
gradient solver uses only one function call to retrieve the
func value, the gradient value and a callable for the Hessian matvec
product. If the function call is very expensive (e.g. for logistic
regression with large design matrix), this approach gives very
significant speedups.
"""
# This is a modified file from scipy.optimize
# Original authors: Travis Oliphant, Eric Jones
# Modifications by Gael Varoquaux, Mathieu Blondel and Tom Dupre la Tour
# License: BSD

import numpy as np
import warnings
from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1

from ..exceptions import ConvergenceWarning


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is None:
        # line search failed: try different one.
        ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                 old_fval, old_old_fval, **kwargs)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def _cg(fhess_p, fgrad, maxiter, tol):
    """
    Solve iteratively the linear system 'fhess_p . xsupi = fgrad'
    with a conjugate gradient descent.

    Parameters
    ----------
    fhess_p : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient

    fgrad : ndarray, shape (n_features,) or (n_features + 1,)
        Gradient vector

    maxiter : int
        Number of CG iterations.

    tol : float
        Stopping criterion.

    Returns
    -------
    xsupi : ndarray, shape (n_features,) or (n_features + 1,)
        Estimated solution
    """
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)
    ri = fgrad
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)

    while i <= maxiter:
        if np.sum(np.abs(ri)) <= tol:
            break

        Ap = fhess_p(psupi)
        # check curvature
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 3 * np.finfo(np.float64).eps:
            break
        elif curv < 0:
            if i > 0:
                break
            else:
                # fall back to steepest descent direction
                xsupi += dri0 / curv * psupi
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri = ri + alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        i = i + 1
        dri0 = dri1          # update np.dot(ri,ri) for next time.

    return xsupi



```
189 - /tmp/repos/scikit-learn/examples/text/plot_document_classification_20newsgroups.py
```python
"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# #############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
```
190 - /tmp/repos/scikit-learn/examples/covariance/plot_covariance_estimation.py
```python
# LW likelihood
plt.vlines(lw.shrinkage_, ymin, -loglik_lw, color='magenta',
           linewidth=3, label='Ledoit-Wolf estimate')
# OAS likelihood
plt.vlines(oa.shrinkage_, ymin, -loglik_oa, color='purple',
           linewidth=3, label='OAS estimate')
# best CV estimator likelihood
plt.vlines(cv.best_estimator_.shrinkage, ymin,
           -cv.best_estimator_.score(X_test), color='cyan',
           linewidth=3, label='Cross-validation best estimate')

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.legend()

plt.show()

```
191 - /tmp/repos/scikit-learn/examples/svm/plot_svm_tie_breaking.py
```python
"""
=========================================================
SVM Tie Breaking Example
=========================================================
Tie breaking is costly if ``decision_function_shape='ovr'``, and therefore it
is not enabled by default. This example illustrates the effect of the
``break_ties`` parameter for a multiclass classification problem and
``decision_function_shape='ovr'``.

The two plots differ only in the area in the middle where the classes are
tied. If ``break_ties=False``, all input in that area would be classified as
one class, whereas if ``break_ties=True``, the tie-breaking mechanism will
create a non-convex decision boundary in that area.
"""
print(__doc__)


# Code source: Andreas Mueller, Adrin Jalali
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=27)

fig, sub = plt.subplots(2, 1, figsize=(5, 8))
titles = ("break_ties = False",
          "break_ties = True")

for break_ties, title, ax in zip((False, True), titles, sub.flatten()):

    svm = SVC(kernel="linear", C=1, break_ties=break_ties,
              decision_function_shape='ovr').fit(X, y)

    xlim = [X[:, 0].min(), X[:, 0].max()]
    ylim = [X[:, 1].min(), X[:, 1].max()]

    xs = np.linspace(xlim[0], xlim[1], 1000)
    ys = np.linspace(ylim[0], ylim[1], 1000)
    xx, yy = np.meshgrid(xs, ys)

    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    colors = [plt.cm.Accent(i) for i in [0, 4, 7]]

    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent")
    classes = [(0, 1), (0, 2), (1, 2)]
    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
    ax.imshow(-pred.reshape(xx.shape), cmap="Accent", alpha=.2,
              extent=(xlim[0], xlim[1], ylim[1], ylim[0]))

    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
        line2 = -(line * coef[1] + intercept) / coef[0]
        ax.plot(line2, line, "-", c=colors[col[0]])
        ax.plot(line2, line, "--", c=colors[col[1]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.show()
```
192 - /tmp/repos/scikit-learn/examples/covariance/plot_mahalanobis_distances.py
```python
r"""
================================================================
Robust covariance estimation and Mahalanobis distances relevance
================================================================

An example to show covariance estimation with the Mahalanobis
distances on Gaussian distributed data.

For Gaussian distributed data, the distance of an observation
:math:`x_i` to the mode of the distribution can be computed using its
Mahalanobis distance: :math:`d_{(\mu,\Sigma)}(x_i)^2 = (x_i -
\mu)'\Sigma^{-1}(x_i - \mu)` where :math:`\mu` and :math:`\Sigma` are
the location and the covariance of the underlying Gaussian
distribution.

In practice, :math:`\mu` and :math:`\Sigma` are replaced by some
estimates.  The usual covariance maximum likelihood estimate is very
sensitive to the presence of outliers in the data set and therefor,
the corresponding Mahalanobis distances are. One would better have to
use a robust estimator of covariance to guarantee that the estimation is
resistant to "erroneous" observations in the data set and that the
associated Mahalanobis distances accurately reflect the true
organisation of the observations.

The Minimum Covariance Determinant estimator is a robust,
high-breakdown point (i.e. it can be used to estimate the covariance
matrix of highly contaminated datasets, up to
:math:`\frac{n_\text{samples}-n_\text{features}-1}{2}` outliers)
estimator of covariance. The idea is to find
:math:`\frac{n_\text{samples}+n_\text{features}+1}{2}`
observations whose empirical covariance has the smallest determinant,
yielding a "pure" subset of observations from which to compute
standards estimates of location and covariance.

The Minimum Covariance Determinant estimator (MCD) has been introduced
by P.J.Rousseuw in [1].

This example illustrates how the Mahalanobis distances are affected by
outlying data: observations drawn from a contaminating distribution
are not distinguishable from the observations coming from the real,
Gaussian distribution that one may want to work with. Using MCD-based
Mahalanobis distances, the two populations become
distinguishable. Associated applications are outliers detection,
observations ranking, clustering, ...
For visualization purpose, the cubic root of the Mahalanobis distances
are represented in the boxplot, as Wilson and Hilferty suggest [2]

[1] P. J. Rousseeuw. Least median of squares regression. J. Am
    Stat Ass, 79:871, 1984.
[2] Wilson, E. B., & Hilferty, M. M. (1931). The distribution of chi-square.
    Proceedings of the National Academy of Sciences of the United States
    of America, 17, 684-688.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.covariance import EmpiricalCovariance, MinCovDet

n_samples = 125
n_outliers = 25
n_features = 2

# generate data
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
# add some outliers
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

# fit a Minimum Covariance Determinant (MCD) robust estimator to data
robust_cov = MinCovDet().fit(X)

# compare estimators learnt from the full data set with true parameters
emp_cov = EmpiricalCovariance().fit(X)

# #############################################################################
# Display results
fig = plt.figure()
plt.subplots_adjust(hspace=-.1, wspace=.4, top=.95, bottom=.05)

# Show data set
subfig1 = plt.subplot(3, 1, 1)
inlier_plot = subfig1.scatter(X[:, 0], X[:, 1],
                              color='black', label='inliers')
outlier_plot = subfig1.scatter(X[:, 0][-n_outliers:], X[:, 1][-n_outliers:],
                               color='red', label='outliers')
subfig1.set_xlim(subfig1.get_xlim()[0], 11.)
subfig1.set_title("Mahalanobis distances of a contaminated data set:")

# Show contours of the distance functions

```
193 - /tmp/repos/scikit-learn/examples/covariance/plot_sparse_cov.py
```python
precs = [('Empirical', linalg.inv(emp_cov)), ('Ledoit-Wolf', lw_prec_),
         ('GraphicalLasso', prec_), ('True', prec)]
vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.imshow(np.ma.masked_equal(this_prec, 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    ax.set_axis_bgcolor('.7')

# plot the model selection metric
plt.figure(figsize=(4, 3))
plt.axes([.2, .15, .75, .7])
plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
plt.axvline(model.alpha_, color='.5')
plt.title('Model selection')
plt.ylabel('Cross-validation score')
plt.xlabel('alpha')

plt.show()

```
194 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def _nls_subproblem(X, W, H, tol, max_iter, alpha=0., l1_ratio=0.,
                    sigma=0.01, beta=0.1):
    """Non-negative least square solver
    Solves a non-negative least squares subproblem using the projected
    gradient descent algorithm.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.
    W : array-like, shape (n_samples, n_components)
        Constant matrix.
    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.
    tol : float
        Tolerance of the stopping condition.
    max_iter : int
        Maximum number of iterations before timing out.
    alpha : double, default: 0.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.
    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty.
        For l1_ratio = 1 it is an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    sigma : float
        Constant used in the sufficient decrease condition checked by the line
        search.  Smaller values lead to a looser sufficient decrease condition,
        thus reducing the time taken by the line search, but potentially
        increasing the number of iterations of the projected gradient
        procedure. 0.01 is a commonly used value in the optimization
        literature.
    beta : float
        Factor by which the step size is decreased (resp. increased) until
        (resp. as long as) the sufficient decrease condition is satisfied.
        Larger values allow to find a better step size but lead to longer line
        search. 0.1 is a commonly used value in the optimization literature.
    Returns
    -------
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    grad : array-like, shape (n_components, n_features)
        The gradient.
    n_iter : int
        The number of iterations done by the algorithm.
    References
    ----------
    C.-J. Lin. Projected gradient methods for non-negative matrix
    factorization. Neural Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/
    """
    WtX = safe_sparse_dot(W.T, X)
    WtW = np.dot(W.T, W)

    # values justified in the paper (alpha is renamed gamma)
    gamma = 1
    for n_iter in range(1, max_iter + 1):
        grad = np.dot(WtW, H) - WtX
        if alpha > 0 and l1_ratio == 1.:
            grad += alpha
        elif alpha > 0:
            grad += alpha * (l1_ratio + (1 - l1_ratio) * H)

        # The following multiplication with a boolean array is more than twice
        # as fast as indexing into grad.
        if _norm(grad * np.logical_or(grad < 0, H > 0)) < tol:
            break

        Hp = H

        for inner_iter in range(20):
            # Gradient step.
            Hn = H - gamma * grad
            # Projection step.
            Hn *= Hn > 0
            d = Hn - H
            gradd = np.dot(grad.ravel(), d.ravel())
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            suff_decr = (1 - sigma) * gradd + 0.5 * dQd < 0
            if inner_iter == 0:
                decr_gamma = not suff_decr

            if decr_gamma:
                if suff_decr:
                    H = Hn
                    break
                else:
                    gamma *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                gamma /= beta
                Hp = Hn

    if n_iter == max_iter:
        warnings.warn("Iteration limit reached in nls subproblem.",
                      ConvergenceWarning)

    return H, grad, n_iter



```
195 - /tmp/repos/scikit-learn/examples/applications/plot_topics_extraction_with_nmf_lda.py
```python



# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

```
196 - /tmp/repos/scikit-learn/examples/neighbors/plot_lof.py
```python
"""
=================================================
Anomaly detection with Local Outlier Factor (LOF)
=================================================

This example presents the Local Outlier Factor (LOF) estimator. The LOF
algorithm is an unsupervised outlier detection method which computes the local
density deviation of a given data point with respect to its neighbors.
It considers as outlier samples that have a substantially lower density than
their neighbors.

The number of neighbors considered, (parameter n_neighbors) is typically
chosen 1) greater than the minimum number of objects a cluster has to contain,
so that other objects can be local outliers relative to this cluster, and 2)
smaller than the maximum number of close by objects that can potentially be
local outliers.
In practice, such informations are generally not available, and taking
n_neighbors=20 appears to work well in general.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Local Outlier Factor (LOF)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a = plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                edgecolor='k', s=20)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()
```
197 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
"""

    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if safe_min(X) == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = avg * np.ones((n_samples, n_components))
        else:
            W = np.zeros((n_samples, n_components))
    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    if solver == 'cd':
        W, H, n_iter = _fit_coordinate_descent(X, W, H, tol, max_iter,
                                               l1_reg_W, l1_reg_H,
                                               l2_reg_W, l2_reg_H,
                                               update_H=update_H,
                                               verbose=verbose,
                                               shuffle=shuffle,
                                               random_state=random_state)
    elif solver == 'mu':
        W, H, n_iter = _fit_multiplicative_update(X, W, H, beta_loss, max_iter,
                                                  tol, l1_reg_W, l1_reg_H,
                                                  l2_reg_W, l2_reg_H, update_H,
                                                  verbose)

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter


class NMF(BaseEstimator, TransformerMixin):
    r"""Non-Negative Matrix Factorization (NMF)

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is::

        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

    Where::

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.

    The objective function is minimized with an alternating minimization of W
    and H.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.

    init :  'random' | 'nndsvd' |  'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise random.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    solver : 'cd' | 'mu'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver.
        'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    alpha : double, default: 0.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.

        .. versionadded:: 0.17
           *alpha* used in the Coordinate Descent solver.

    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

        .. versionadded:: 0.17
           Regularization parameter *l1_ratio* used in the Coordinate Descent
           solver.

    verbose : bool, default=False
        Whether to be verbose.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Factorization matrix, sometimes called 'dictionary'.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    
```
198 - /tmp/repos/scikit-learn/sklearn/cluster/hierarchical.py
```python
"""Hierarchical Agglomerative Clustering

These routines perform some hierarchical agglomerative clustering of some
input data.

Authors : Vincent Michel, Bertrand Thirion, Alexandre Gramfort,
          Gael Varoquaux
License: BSD 3 clause
"""
from heapq import heapify, heappop, heappush, heappushpop
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..base import BaseEstimator, ClusterMixin
from ..externals import six
from ..metrics.pairwise import paired_distances, pairwise_distances
from ..utils import check_array
from ..utils.validation import check_memory

from . import _hierarchical
from ._feature_agglomeration import AgglomerationTransform
from ..utils.fast_dict import IntFloatDict

from ..externals.six.moves import xrange

###############################################################################
# For non fully-connected graphs


def _fix_connectivity(X, connectivity, affinity):
    """
    Fixes the connectivity matrix

        - copies it
        - makes it symmetric
        - converts it to LIL if necessary
        - completes it if necessary
    """
    n_samples = X.shape[0]
    if (connectivity.shape[0] != n_samples or
            connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s '
                         'when X is %s' % (connectivity.shape, X.shape))

    # Make the connectivity matrix symmetric:
    connectivity = connectivity + connectivity.T

    # Convert connectivity matrix to LIL
    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()

    # Compute the number of nodes
    n_components, labels = connected_components(connectivity)

    if n_components > 1:
        warnings.warn("the number of connected components of the "
                      "connectivity matrix is %d > 1. Completing it to avoid "
                      "stopping the tree early." % n_components,
                      stacklevel=2)
        # XXX: Can we do without completing the matrix?
        for i in xrange(n_components):
            idx_i = np.where(labels == i)[0]
            Xi = X[idx_i]
            for j in xrange(i):
                idx_j = np.where(labels == j)[0]
                Xj = X[idx_j]
                D = pairwise_distances(Xi, Xj, metric=affinity)
                ii, jj = np.where(D == np.min(D))
                ii = ii[0]
                jj = jj[0]
                connectivity[idx_i[ii], idx_j[jj]] = True
                connectivity[idx_j[jj], idx_i[ii]] = True

    return connectivity, n_components



```
199 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python
def one_run(n_samples):
    X_train = X_train_[:n_samples]
    X_test = X_test_[:n_samples]
    y_train = y_train_[:n_samples]
    y_test = y_test_[:n_samples]
    assert X_train.shape[0] == n_samples
    assert X_test.shape[0] == n_samples
    print("Data size: %d samples train, %d samples test."
          % (n_samples, n_samples))
    print("Fitting a sklearn model...")
    tic = time()
    est = Estimator(learning_rate=lr,
                    max_iter=n_trees,
                    max_bins=max_bins,
                    max_leaf_nodes=n_leaf_nodes,
                    n_iter_no_change=None,
                    random_state=0,
                    verbose=0)
    est.fit(X_train, y_train)
    sklearn_fit_duration = time() - tic
    tic = time()
    sklearn_score = est.score(X_test, y_test)
    sklearn_score_duration = time() - tic
    print("score: {:.4f}".format(sklearn_score))
    print("fit duration: {:.3f}s,".format(sklearn_fit_duration))
    print("score duration: {:.3f}s,".format(sklearn_score_duration))

    lightgbm_score = None
    lightgbm_fit_duration = None
    lightgbm_score_duration = None
    if args.lightgbm:
        print("Fitting a LightGBM model...")
        # get_lightgbm does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')

        tic = time()
        lightgbm_est.fit(X_train, y_train)
        lightgbm_fit_duration = time() - tic
        tic = time()
        lightgbm_score = lightgbm_est.score(X_test, y_test)
        lightgbm_score_duration = time() - tic
        print("score: {:.4f}".format(lightgbm_score))
        print("fit duration: {:.3f}s,".format(lightgbm_fit_duration))
        print("score duration: {:.3f}s,".format(lightgbm_score_duration))

    xgb_score = None
    xgb_fit_duration = None
    xgb_score_duration = None
    if args.xgboost:
        print("Fitting an XGBoost model...")
        # get_xgb does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        xgb_est = get_equivalent_estimator(est, lib='xgboost')

        tic = time()
        xgb_est.fit(X_train, y_train)
        xgb_fit_duration = time() - tic
        tic = time()
        xgb_score = xgb_est.score(X_test, y_test)
        xgb_score_duration = time() - tic
        print("score: {:.4f}".format(xgb_score))
        print("fit duration: {:.3f}s,".format(xgb_fit_duration))
        print("score duration: {:.3f}s,".format(xgb_score_duration))

    cat_score = None
    cat_fit_duration = None
    cat_score_duration = None
    if args.catboost:
        print("Fitting a CatBoost model...")
        # get_cat does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        cat_est = get_equivalent_estimator(est, lib='catboost')

        tic = time()
        cat_est.fit(X_train, y_train)
        cat_fit_duration = time() - tic
        tic = time()
        cat_score = cat_est.score(X_test, y_test)
        cat_score_duration = time() - tic
        print("score: {:.4f}".format(cat_score))
        print("fit duration: {:.3f}s,".format(cat_fit_duration))
        print("score duration: {:.3f}s,".format(cat_score_duration))

    return (sklearn_score, sklearn_fit_duration, sklearn_score_duration,
            lightgbm_score, lightgbm_fit_duration, lightgbm_score_duration,
            xgb_score, xgb_fit_duration, xgb_score_duration,
            cat_score, cat_fit_duration, cat_score_duration)
```
200 - /tmp/repos/scikit-learn/sklearn/covariance/robust_covariance.py
```python
def _c_step(X, n_support, random_state, remaining_iterations=30,
            initial_estimates=None, verbose=False,
            cov_computation_method=empirical_covariance):
    n_samples, n_features = X.shape
    dist = np.inf

    # Initialisation
    support = np.zeros(n_samples, dtype=bool)
    if initial_estimates is None:
        # compute initial robust estimates from a random subset
        support[random_state.permutation(n_samples)[:n_support]] = True
    else:
        # get initial robust estimates from the function parameters
        location = initial_estimates[0]
        covariance = initial_estimates[1]
        # run a special iteration for that case (to get an initial support)
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
        # compute new estimates
        support[np.argsort(dist)[:n_support]] = True

    X_support = X[support]
    location = X_support.mean(0)
    covariance = cov_computation_method(X_support)

    # Iterative procedure for Minimum Covariance Determinant computation
    det = fast_logdet(covariance)
    # If the data already has singular covariance, calculate the precision,
    # as the loop below will not be entered.
    if np.isinf(det):
        precision = linalg.pinvh(covariance)

    previous_det = np.inf
    while (det < previous_det and remaining_iterations > 0
            and not np.isinf(det)):
        # save old estimates values
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support = support
        # compute a new support from the full data set mahalanobis distances
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(axis=1)
        # compute new estimates
        support = np.zeros(n_samples, dtype=bool)
        support[np.argsort(dist)[:n_support]] = True
        X_support = X[support]
        location = X_support.mean(axis=0)
        covariance = cov_computation_method(X_support)
        det = fast_logdet(covariance)
        # update remaining iterations for early stopping
        remaining_iterations -= 1

    previous_dist = dist
    dist = (np.dot(X - location, precision) * (X - location)).sum(axis=1)
    # Check if best fit already found (det => 0, logdet => -inf)
    if np.isinf(det):
        results = location, covariance, det, support, dist
    # Check convergence
    if np.allclose(det, previous_det):
        # c_step procedure converged
        if verbose:
            print("Optimal couple (location, covariance) found before"
                  " ending iterations (%d left)" % (remaining_iterations))
        results = location, covariance, det, support, dist
    elif det > previous_det:
        # determinant has increased (should not happen)
        warnings.warn("Warning! det > previous_det (%.15f > %.15f)"
                      % (det, previous_det), RuntimeWarning)
        results = previous_location, previous_covariance, \
            previous_det, previous_support, previous_dist

    # Check early stopping
    if remaining_iterations == 0:
        if verbose:
            print('Maximum number of iterations reached')
        results = location, covariance, det, support, dist

    return results


def select_candidates(X, n_support, n_trials, select=1, n_iter=30,
                      verbose=False,
                      cov_computation_method=empirical_covariance,
                      random_state=None):
    
```
201 - /tmp/repos/scikit-learn/benchmarks/bench_plot_incremental_pca.py
```python
"""
========================
IncrementalPCA benchmark
========================

Benchmarks for IncrementalPCA

"""

import numpy as np
import gc
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import IncrementalPCA, RandomizedPCA, PCA


def plot_results(X, y, label):
    plt.plot(X, y, label=label, marker='o')


def benchmark(estimator, data):
    gc.collect()
    print("Benching %s" % estimator)
    t0 = time()
    estimator.fit(data)
    training_time = time() - t0
    data_t = estimator.transform(data)
    data_r = estimator.inverse_transform(data_t)
    reconstruction_error = np.mean(np.abs(data - data_r))
    return {'time': training_time, 'error': reconstruction_error}


def plot_feature_times(all_times, batch_size, all_components, data):
    plt.figure()
    plot_results(all_components, all_times['pca'], label="PCA")
    plot_results(all_components, all_times['ipca'],
                 label="IncrementalPCA, bsize=%i" % batch_size)
    plot_results(all_components, all_times['rpca'], label="RandomizedPCA")
    plt.legend(loc="upper left")
    plt.suptitle("Algorithm runtime vs. n_components\n \
                 LFW, size %i x %i" % data.shape)
    plt.xlabel("Number of components (out of max %i)" % data.shape[1])
    plt.ylabel("Time (seconds)")


def plot_feature_errors(all_errors, batch_size, all_components, data):
    plt.figure()
    plot_results(all_components, all_errors['pca'], label="PCA")
    plot_results(all_components, all_errors['ipca'],
                 label="IncrementalPCA, bsize=%i" % batch_size)
    plot_results(all_components, all_errors['rpca'], label="RandomizedPCA")
    plt.legend(loc="lower left")
    plt.suptitle("Algorithm error vs. n_components\n"
                 "LFW, size %i x %i" % data.shape)
    plt.xlabel("Number of components (out of max %i)" % data.shape[1])
    plt.ylabel("Mean absolute error")


def plot_batch_times(all_times, n_features, all_batch_sizes, data):
    plt.figure()
    plot_results(all_batch_sizes, all_times['pca'], label="PCA")
    plot_results(all_batch_sizes, all_times['rpca'], label="RandomizedPCA")
    plot_results(all_batch_sizes, all_times['ipca'], label="IncrementalPCA")
    plt.legend(loc="lower left")
    plt.suptitle("Algorithm runtime vs. batch_size for n_components %i\n \
                 LFW, size %i x %i" % (
                 n_features, data.shape[0], data.shape[1]))
    plt.xlabel("Batch size")
    plt.ylabel("Time (seconds)")


def plot_batch_errors(all_errors, n_features, all_batch_sizes, data):
    plt.figure()
    plot_results(all_batch_sizes, all_errors['pca'], label="PCA")
    plot_results(all_batch_sizes, all_errors['ipca'], label="IncrementalPCA")
    plt.legend(loc="lower left")
    plt.suptitle("Algorithm error vs. batch_size for n_components %i\n \
                 LFW, size %i x %i" % (
                 n_features, data.shape[0], data.shape[1]))
    plt.xlabel("Batch size")
    plt.ylabel("Mean absolute error")


def fixed_batch_size_comparison(data):
    all_features = [i.astype(int) for i in np.linspace(data.shape[1] // 10,
                                                       data.shape[1], num=5)]
    batch_size = 1000
    # Compare runtimes and error for fixed batch size
    all_times = defaultdict(list)
    all_errors = defaultdict(list)
    for n_components in all_features:
        pca = PCA(n_components=n_components)
        rpca = RandomizedPCA(n_components=n_components, random_state=1999)
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        results_dict = {k: benchmark(est, data) for k, est in [('pca', pca),
                                                               ('ipca', ipca),
                                                               ('rpca', rpca)]}

        for k in sorted(results_dict.keys()):
            all_times[k].append(results_dict[k]['time'])
            all_errors[k].append(results_dict[k]['error'])

    plot_feature_times(all_times, batch_size, all_features, data)
    plot_feature_errors(all_errors, batch_size, all_features, data)



```
202 - /tmp/repos/scikit-learn/examples/applications/plot_model_complexity_influence.py
```python
"""
==========================
Model Complexity Influence
==========================

Demonstrate how model complexity influences both prediction accuracy and
computational performance.

The dataset is the Boston Housing dataset (resp. 20 Newsgroups) for
regression (resp. classification).

For each class of models we make the model complexity vary through the choice
of relevant model parameters and measure the influence on both computational
performance (latency) and predictive power (MSE or Hamming Loss).
"""

print(__doc__)

# Author: Eustache Diemert <eustache@diemert.fr>
# License: BSD 3 clause

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.parasite_axes import host_subplot
from mpl_toolkits.axisartist.axislines import Axes
from scipy.sparse.csr import csr_matrix

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.svm.classes import NuSVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import hamming_loss

# #############################################################################
# Routines


# Initialize random generator
np.random.seed(0)


def generate_data(case, sparse=False):
    """Generate regression/classification data."""
    bunch = None
    if case == 'regression':
        bunch = datasets.load_boston()
    elif case == 'classification':
        bunch = datasets.fetch_20newsgroups_vectorized(subset='all')
    X, y = shuffle(bunch.data, bunch.target)
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    if sparse:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
            'y_test': y_test}
    return data


def benchmark_influence(conf):
    """
    Benchmark influence of :changing_param: on both MSE and latency.
    """
    prediction_times = []
    prediction_powers = []
    complexities = []
    for param_value in conf['changing_param_values']:
        conf['tuned_params'][conf['changing_param']] = param_value
        estimator = conf['estimator'](**conf['tuned_params'])
        print("Benchmarking %s" % estimator)
        estimator.fit(conf['data']['X_train'], conf['data']['y_train'])
        conf['postfit_hook'](estimator)
        complexity = conf['complexity_computer'](estimator)
        complexities.append(complexity)
        start_time = time.time()
        for _ in range(conf['n_samples']):
            y_pred = estimator.predict(conf['data']['X_test'])
        elapsed_time = (time.time() - start_time) / float(conf['n_samples'])
        prediction_times.append(elapsed_time)
        pred_score = conf['prediction_performance_computer'](
            conf['data']['y_test'], y_pred)
        prediction_powers.append(pred_score)
        print("Complexity: %d | %s: %.4f | Pred. Time: %fs\n" % (
            complexity, conf['prediction_performance_label'], pred_score,
            elapsed_time))
    return prediction_powers, prediction_times, complexities


def plot_influence(conf, mse_values, prediction_times, complexities):
    """
    Plot influence of model complexity on both accuracy and latency.
    """
    plt.figure(figsize=(12, 6))
    host = host_subplot(111, axes_class=Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    host.set_xlabel('Model Complexity (%s)' % conf['complexity_label'])
    y1_label = conf['prediction_performance_label']
    y2_label = "Time (s)"
    host.set_ylabel(y1_label)
    par1.set_ylabel(y2_label)
    p1, = host.plot(complexities, mse_values, 'b-', label="prediction error")
    p2, = par1.plot(complexities, prediction_times, 'r-',
                    label="latency")
    host.legend(loc='upper right')
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    plt.title('Influence of Model Complexity - %s' % conf['estimator'].__name__)
    plt.show()



```
203 - /tmp/repos/scikit-learn/benchmarks/bench_plot_incremental_pca.py
```python
def variable_batch_size_comparison(data):
    batch_sizes = [i.astype(int) for i in np.linspace(data.shape[0] // 10,
                                                      data.shape[0], num=10)]

    for n_components in [i.astype(int) for i in
                         np.linspace(data.shape[1] // 10,
                                     data.shape[1], num=4)]:
        all_times = defaultdict(list)
        all_errors = defaultdict(list)
        pca = PCA(n_components=n_components)
        rpca = RandomizedPCA(n_components=n_components, random_state=1999)
        results_dict = {k: benchmark(est, data) for k, est in [('pca', pca),
                                                               ('rpca', rpca)]}

        # Create flat baselines to compare the variation over batch size
        all_times['pca'].extend([results_dict['pca']['time']] *
                                len(batch_sizes))
        all_errors['pca'].extend([results_dict['pca']['error']] *
                                 len(batch_sizes))
        all_times['rpca'].extend([results_dict['rpca']['time']] *
                                 len(batch_sizes))
        all_errors['rpca'].extend([results_dict['rpca']['error']] *
                                  len(batch_sizes))
        for batch_size in batch_sizes:
            ipca = IncrementalPCA(n_components=n_components,
                                  batch_size=batch_size)
            results_dict = {k: benchmark(est, data) for k, est in [('ipca',
                                                                   ipca)]}
            all_times['ipca'].append(results_dict['ipca']['time'])
            all_errors['ipca'].append(results_dict['ipca']['error'])

        plot_batch_times(all_times, n_components, batch_sizes, data)
        # RandomizedPCA error is always worse (approx 100x) than other PCA
        # tests
        plot_batch_errors(all_errors, n_components, batch_sizes, data)

faces = fetch_lfw_people(resize=.2, min_faces_per_person=5)
# limit dataset to 5000 people (don't care who they are!)
X = faces.data[:5000]
n_samples, h, w = faces.images.shape
n_features = X.shape[1]

X -= X.mean(axis=0)
X /= X.std(axis=0)

fixed_batch_size_comparison(X)
variable_batch_size_comparison(X)
plt.show()

```
204 - /tmp/repos/scikit-learn/examples/linear_model/plot_bayesian_ridge.py
```python
"""
=========================
Bayesian Ridge Regression
=========================

Computes a Bayesian Ridge Regression on a synthetic dataset.

See :ref:`bayesian_ridge_regression` for more information on the regressor.

Compared to the OLS (ordinary least squares) estimator, the coefficient
weights are slightly shifted toward zeros, which stabilises them.

As the prior on the weights is a Gaussian prior, the histogram of the
estimated weights is Gaussian.

The estimation of the model is done by iteratively maximizing the
marginal log-likelihood of the observations.

We also plot predictions and uncertainties for Bayesian Ridge Regression
for one dimensional regression using polynomial feature expansion.
Note the uncertainty starts going up on the right side of the plot.
This is because these test samples are outside of the range of the training
samples.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import BayesianRidge, LinearRegression

# #############################################################################
# Generating simulated data with Gaussian weights
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)  # Create Gaussian data
# Create weights with a precision lambda_ of 4.
lambda_ = 4.
w = np.zeros(n_features)
# Only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# Create noise with a precision alpha of 50.
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# Create the target
y = np.dot(X, w) + noise

# #############################################################################
# Fit the Bayesian Ridge Regression and an OLS for comparison
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

# #############################################################################
# Plot true weights, estimated weights, histogram of the weights, and
# predictions with standard deviations
lw = 2
plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
         label="Bayesian Ridge estimate")
plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
         edgecolor='black')
plt.scatter(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)),
            color='navy', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="upper left")

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color='navy', linewidth=lw)
plt.ylabel("Score")
plt.xlabel("Iterations")


# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=0.1)
clf_poly = BayesianRidge()
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
plt.figure(figsize=(6, 5))
plt.errorbar(X_plot, y_mean, y_std, color='navy',
             label="Polynomial Bayesian Ridge Regression", linewidth=lw)
plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
         label="Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()

```
205 - /tmp/repos/scikit-learn/examples/cross_decomposition/plot_compare_cross_decomposition.py
```python
"""
===================================
Compare cross decomposition methods
===================================

Simple usage of various cross decomposition algorithms:
- PLSCanonical
- PLSRegression, with multivariate response, a.k.a. PLS2
- PLSRegression, with univariate response, a.k.a. PLS1
- CCA

Given 2 multivariate covarying two-dimensional datasets, X, and Y,
PLS extracts the 'directions of covariance', i.e. the components of each
datasets that explain the most shared variance between both datasets.
This is apparent on the **scatterplot matrix** display: components 1 in
dataset X and dataset Y are maximally correlated (points lie around the
first diagonal). This is also true for components 2 in both dataset,
however, the correlation across datasets for different components is
weak: the point cloud is very spherical.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

# #############################################################################
# Dataset based latent variables model

n = 500
# 2 latents vars:
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)

latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

X_train = X[:n // 2]
Y_train = Y[:n // 2]
X_test = X[n // 2:]
Y_test = Y[n // 2:]

print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))

# #############################################################################
# Canonical (symmetric) PLS

# Transform data
# ~~~~~~~~~~~~~~
plsca = PLSCanonical(n_components=2)
plsca.fit(X_train, Y_train)
X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

# Scatter plot of scores
# ~~~~~~~~~~~~~~~~~~~~~~
# 1) On diagonal plot X vs Y scores on each components
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train",
            marker="o", c="b", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test",
            marker="o", c="r", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('Comp. 1: X vs Y (test corr = %.2f)' %
          np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

plt.subplot(224)
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train",
            marker="o", c="b", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test",
            marker="o", c="r", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
          np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

# 2) Off diagonal plot components 1 vs 2 for X and Y
plt.subplot(222)
plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train",
            marker="*", c="b", s=50)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test",
            marker="*", c="r", s=50)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title('X comp. 1 vs X comp. 2 (test corr = %.2f)'
          % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())

plt.subplot(223)
plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train",
            marker="*", c="b", s=50)

```
206 - /tmp/repos/scikit-learn/examples/plot_multilabel.py
```python
# Authors: Vlad Niculae, Mathieu Blondel
# License: BSD 3 clause
"""
=========================
Multilabel classification
=========================

This example simulates a multi-label document classification problem. The
dataset is generated randomly based on the following process:

    - pick the number of labels: n ~ Poisson(n_labels)
    - n times, choose a class c: c ~ Multinomial(theta)
    - pick the document length: k ~ Poisson(length)
    - k times, choose a word: w ~ Multinomial(theta_c)

In the above process, rejection sampling is used to make sure that n is more
than 2, and that the document length is never zero. Likewise, we reject classes
which have already been chosen.  The documents that are assigned to both
classes are plotted surrounded by two colored circles.

The classification is performed by projecting to the first two principal
components found by PCA and CCA for visualisation purposes, followed by using
the :class:`sklearn.multiclass.OneVsRestClassifier` metaclassifier using two
SVCs with linear kernels to learn a discriminative model for each class.
Note that PCA is used to perform an unsupervised dimensionality reduction,
while CCA is used to perform a supervised one.

Note: in the plot, "unlabeled samples" does not mean that we don't know the
labels (as in semi-supervised learning) but that the samples simply do *not*
have a label.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")
```
207 - /tmp/repos/scikit-learn/sklearn/decomposition/online_lda.py
```python
"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""

# Author: Chyi-Kwei Yau
# Author: Matthew D. Hoffman (original onlineldavb implementation)

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
import warnings

from ..base import BaseEstimator, TransformerMixin
from ..utils import (check_random_state, check_array,
                     gen_batches, gen_even_slices, _get_n_jobs)
from ..utils.fixes import logsumexp
from ..utils.validation import check_non_negative
from ..externals.joblib import Parallel, delayed
from ..externals.six.moves import xrange
from ..exceptions import NotFittedError

from ._online_lda import (mean_change, _dirichlet_expectation_1d,
                          _dirichlet_expectation_2d)

EPS = np.finfo(np.float).eps



```
208 - /tmp/repos/scikit-learn/examples/decomposition/plot_ica_blind_source_separation.py
```python
"""
=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 3 instruments playing simultaneously and 3 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument. Importantly, PCA fails
at recovering our `instruments` since the related signals reflect
non-Gaussian processes.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
```
209 - /tmp/repos/scikit-learn/examples/cluster/plot_optics.py
```python
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()

```
210 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/grower.py
```python


    def _validate_parameters(self, X_binned, max_leaf_nodes, max_depth,
                             min_samples_leaf, min_gain_to_split,
                             l2_regularization, min_hessian_to_split):
        """Validate parameters passed to __init__.

        Also validate parameters passed to splitter.
        """
        if X_binned.dtype != np.uint8:
            raise NotImplementedError(
                "X_binned must be of type uint8.")
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency.")
        if max_leaf_nodes is not None and max_leaf_nodes <= 1:
            raise ValueError('max_leaf_nodes={} should not be'
                             ' smaller than 2'.format(max_leaf_nodes))
        if max_depth is not None and max_depth <= 1:
            raise ValueError('max_depth={} should not be'
                             ' smaller than 2'.format(max_depth))
        if min_samples_leaf < 1:
            raise ValueError('min_samples_leaf={} should '
                             'not be smaller than 1'.format(min_samples_leaf))
        if min_gain_to_split < 0:
            raise ValueError('min_gain_to_split={} '
                             'must be positive.'.format(min_gain_to_split))
        if l2_regularization < 0:
            raise ValueError('l2_regularization={} must be '
                             'positive.'.format(l2_regularization))
        if min_hessian_to_split < 0:
            raise ValueError('min_hessian_to_split={} '
                             'must be positive.'.format(min_hessian_to_split))

    def grow(self):
        """Grow the tree, from root to leaves."""
        while self.splittable_nodes:
            self.split_next()

    def _intilialize_root(self, gradients, hessians, hessians_are_constant):
        """Initialize root node and finalize it if needed."""
        n_samples = self.X_binned.shape[0]
        depth = 0
        sum_gradients = sum_parallel(gradients)
        if self.histogram_builder.hessians_are_constant:
            sum_hessians = hessians[0] * n_samples
        else:
            sum_hessians = sum_parallel(hessians)
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitter.partition,
            sum_gradients=sum_gradients,
            sum_hessians=sum_hessians
        )

        self.root.partition_start = 0
        self.root.partition_stop = n_samples

        if self.root.n_samples < 2 * self.min_samples_leaf:
            # Do not even bother computing any splitting statistics.
            self._finalize_leaf(self.root)
            return
        if sum_hessians < self.splitter.min_hessian_to_split:
            self._finalize_leaf(self.root)
            return

        self.root.histograms = self.histogram_builder.compute_histograms_brute(
            self.root.sample_indices)
        self._compute_best_split_and_push(self.root)

    def _compute_best_split_and_push(self, node):
        """Compute the best possible split (SplitInfo) of a given node.

        Also push it in the heap of splittable nodes if gain isn't zero.
        The gain of a node is 0 if either all the leaves are pure
        (best gain = 0), or if no split would satisfy the constraints,
        (min_hessians_to_split, min_gain_to_split, min_samples_leaf)
        """

        node.split_info = self.splitter.find_node_split(
            node.n_samples, node.histograms, node.sum_gradients,
            node.sum_hessians)

        if node.split_info.gain <= 0:  # no valid split
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    
```
211 - /tmp/repos/scikit-learn/examples/applications/plot_face_recognition.py
```python
"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))

```
212 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python
\
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

        - 'recursion' is only supported for objects inheriting from
          `BaseGradientBoosting`, but is more efficient in terms of speed.
          With this method, ``X`` is optional and is only used to build the
          grid and the partial dependences are computed using the training
          data. This method does not account for the ``init`` predicor of
          the boosting process, which may lead to incorrect values (see
          warning below. With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities.

        - 'brute' is supported for any estimator, but is more
          computationally intensive.

        - If 'auto', then 'recursion' will be used for
          ``BaseGradientBoosting`` estimators with ``init=None``, and
          'brute' for all other.

        Unlike the 'brute' method, 'recursion' does not account for the
        ``init`` predictor of the boosting process. In practice this still
        produces the same plots, up to a constant offset in the target
        response.
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
    dependence values are incorrect for 'recursion'.
    """
```
213 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
"""Compute Non-negative Matrix Factorization with Multiplicative Update

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant input matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    max_iter : integer, default: 200
        Number of iterations.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    l1_reg_W : double, default: 0.
        L1 regularization parameter for W.

    l1_reg_H : double, default: 0.
        L1 regularization parameter for H.

    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.

    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : integer, default: 0
        The verbosity level.

    Returns
    -------
    W : array, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1. / (2. - beta_loss)
    elif beta_loss > 2:
        gamma = 1. / (beta_loss - 1.)
    else:
        gamma = 1.

    # used for the convergence criterion
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    
```
214 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.")
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})



```
215 - /tmp/repos/scikit-learn/sklearn/utils/arpack.py
```python
# Remove this module in version 0.21

from scipy.sparse.linalg import eigs as _eigs, eigsh as _eigsh, svds as _svds

from .deprecation import deprecated


@deprecated("sklearn.utils.arpack.eigs was deprecated in version 0.19 and "
            "will be removed in 0.21. Use scipy.sparse.linalg.eigs instead.")
def eigs(A, *args, **kwargs):
    return _eigs(A, *args, **kwargs)


@deprecated("sklearn.utils.arpack.eigsh was deprecated in version 0.19 and "
            "will be removed in 0.21. Use scipy.sparse.linalg.eigsh instead.")
def eigsh(A, *args, **kwargs):
    return _eigsh(A, *args, **kwargs)


@deprecated("sklearn.utils.arpack.svds was deprecated in version 0.19 and "
            "will be removed in 0.21. Use scipy.sparse.linalg.svds instead.")
def svds(A, *args, **kwargs):
    return _svds(A, *args, **kwargs)
```
216 - /tmp/repos/scikit-learn/examples/svm/plot_separating_hyperplane_unbalanced.py
```python
"""
=================================================
SVM: Separating hyperplane for unbalanced classes
=================================================

Find the optimal separating hyperplane using an SVC for classes that
are unbalanced.

We first find the separating plane with a plain SVC and then plot
(dashed) the separating hyperplane with automatically correction for
unbalanced classes.

.. currentmodule:: sklearn.linear_model

.. note::

    This example will also work by replacing ``SVC(kernel="linear")``
    with ``SGDClassifier(loss="hinge")``. Setting the ``loss`` parameter
    of the :class:`SGDClassifier` equal to ``hinge`` will yield behaviour
    such as that of a SVC with a linear kernel.

    For example try instead of the ``SVC``::

        clf = SGDClassifier(n_iter=100, alpha=0.01)

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create two clusters of random points
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

# plot separating hyperplanes and samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.legend()

# plot the decision functions for both classifiers
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()
```
**217 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
```python
"""Perform Fast Independent Component Analysis.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    n_components : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.

    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.

    whiten : boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
        Otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.

    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:

        def my_g(x):
            return x ** 3, 3 * x ** 2

    fun_args : dictionary, optional
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}

    max_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : (n_components, n_components) array, optional
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    return_X_mean : bool, optional
        If True, X_mean is returned too.

    compute_sources : bool, optional
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    K : array, shape (n_components, n_features) | None.
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::

            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I

    S : array, shape (n_samples, n_components) | None
        Estimated source matrix

    X_mean : array, shape (n_features, )
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----

    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    Implemented using FastICA:
    `A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430`

    """
    random_state = check_random_state(random_state)
    
```
218 - /tmp/repos/scikit-learn/sklearn/metrics/cluster/supervised.py
```python
"""Utilities to evaluate the clustering performance of models.

Functions named as *_score return a scalar value to maximize: the higher the
better.
"""

# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Wei LI <kuantkid@gmail.com>
#          Diego Molla <dmolla-aliod@gmail.com>
#          Arnaud Fouchet <foucheta@gmail.com>
#          Thierry Guillemot <thierry.guillemot.work@gmail.com>
#          Gregory Stupp <stuppie@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause

from __future__ import division

from math import log

import numpy as np
from scipy import sparse as sp

from .expected_mutual_info_fast import expected_mutual_information
from ...utils.validation import check_array
from ...utils.fixes import comb


def comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


# clustering measures


```
219 - /tmp/repos/scikit-learn/examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py
```python
"""
=====================================================
Multiclass sparse logisitic regression on newgroups20
=====================================================

Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression
to classify documents from the newgroups20 dataset. Multinomial logistic
regression yields more accurate results and is faster to train on the larger
scale dataset.

Here we use the l1 sparsity that trims the weights of not informative
features to zero. This is good if the goal is to extract the strongly
discriminative vocabulary of each class. If the goal is to get the best
predictive accuracy, it is better to use the non sparsity-inducing l2 penalty
instead.

A more traditional (and possibly better) way to predict on a sparse subset of
input features would be to use univariate feature selection followed by a
traditional (l2-penalised) logistic regression model.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print(__doc__)
# Author: Arthur Mensch

t0 = time.clock()

# We use SAGA solver
solver = 'saga'

# Turn down for faster run time
n_samples = 10000

# Memorized fetch_rcv1 for faster access
dataset = fetch_20newsgroups_vectorized('all')
X = dataset.data
y = dataset.target
X = X[:n_samples]
y = y[:n_samples]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, n_features, n_classes))

models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 3]},
          'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}}

for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params['iters']:
        print('[model=%s, solver=%s] Number of epochs: %s' %
              (model_params['name'], solver, this_max_iter))
        lr = LogisticRegression(solver=solver,
                                multi_class=model,
                                C=1,
                                penalty='l1',
                                fit_intercept=True,
                                max_iter=this_max_iter,
                                random_state=42,
                                )
        t1 = time.clock()
        lr.fit(X_train, y_train)
        train_time = time.clock() - t1

        y_pred = lr.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        density = np.mean(lr.coef_ != 0, axis=1) * 100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]['times'] = times
    models[model]['densities'] = densities
    models[model]['accuracies'] = accuracies
    print('Test accuracy for model %s: %.4f' % (model, accuracies[-1]))
    print('%% non-zero coefficients for model %s, '
          'per class:\n %s' % (model, densities[-1]))
    print('Run time (%i epochs) for model %s:'
          '%.2f' % (model_params['iters'][-1], model, times[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]['name']
    times = models[model]['times']
    accuracies = models[model]['accuracies']
    ax.plot(times, accuracies, marker='o',
            label='Model: %s' % name)
    ax.set_xlabel('Train time (s)')
    ax.set_ylabel('Test accuracy')
ax.legend()
fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
             'Dataset %s' % '20newsgroups')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = time.clock() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```
**220 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py**:
```python
"""Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, optional, default: 0.5
        Damping factor (between 0.5 and 1) is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, optional, default: 200
        Maximum number of iterations.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : boolean, optional, default: True
        Make a copy of input data.

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : string, optional, default=``euclidean``
        Which affinity to use. At the moment ``precomputed`` and
        ``euclidean`` are supported. ``euclidean`` uses the
        negative squared euclidean distance between points.

    verbose : boolean, optional, default: False
        Whether to be verbose.


    Attributes
    ----------
    cluster_centers_indices_ : array, shape (n_clusters,)
        Indices of cluster centers

    cluster_centers_ : array, shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : array, shape (n_samples,)
        Labels of each point

    affinity_matrix_ : array, shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
    array and all training samples will be labelled as ``-1``. In addition,
    ``predict`` will then label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """

    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    
```
221 - /tmp/repos/scikit-learn/sklearn/metrics/cluster/__init__.py
```python
"""
The :mod:`sklearn.metrics.cluster` submodule contains evaluation metrics for
cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the 'quality' of the model itself.
"""
from .supervised import adjusted_mutual_info_score
from .supervised import normalized_mutual_info_score
from .supervised import adjusted_rand_score
from .supervised import completeness_score
from .supervised import contingency_matrix
from .supervised import expected_mutual_information
from .supervised import homogeneity_completeness_v_measure
from .supervised import homogeneity_score
from .supervised import mutual_info_score
from .supervised import v_measure_score
from .supervised import fowlkes_mallows_score
from .supervised import entropy
from .unsupervised import silhouette_samples
from .unsupervised import silhouette_score
from .unsupervised import calinski_harabaz_score
from .bicluster import consensus_score

__all__ = ["adjusted_mutual_info_score", "normalized_mutual_info_score",
           "adjusted_rand_score", "completeness_score", "contingency_matrix",
           "expected_mutual_information", "homogeneity_completeness_v_measure",
           "homogeneity_score", "mutual_info_score", "v_measure_score",
           "fowlkes_mallows_score", "entropy", "silhouette_samples",
           "silhouette_score", "calinski_harabaz_score", "consensus_score"]
```
222 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py
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

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause

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
223 - /tmp/repos/scikit-learn/examples/classification/plot_lda.py
```python
"""
====================================================================
Normal and Shrinkage Linear Discriminant Analysis for classification
====================================================================

Shows how shrinkage improves classification.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
         label="Linear Discriminant Analysis with shrinkage", color='navy')
plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
         label="Linear Discriminant Analysis", color='gold')

plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')

plt.legend(loc=1, prop={'size': 12})
plt.suptitle('Linear Discriminant Analysis vs. \
shrinkage Linear Discriminant Analysis (1 discriminative feature)')
plt.show()

```
224 - /tmp/repos/scikit-learn/examples/ensemble/plot_bias_variance.py
```python
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y


X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

plt.figure(figsize=(10, 8))

# Loop over estimators to compare
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

    plt.plot(X_test, np.mean(y_predict, axis=1), "c",
             label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlim([-5, 5])
    plt.title(name)

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))

    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")

    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if n == n_estimators - 1:

        plt.legend(loc=(1.1, .5))

plt.subplots_adjust(right=.75)
plt.show()

```
225 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree

        .. versionadded:: 0.18

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        check_is_fitted(self, 'tree_')

        return self.tree_.compute_feature_importances()


# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
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

    max_features : int, float, string or None, optional (default=None)
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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

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

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
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

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

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
    
```
226 - /tmp/repos/scikit-learn/sklearn/utils/sparsetools/__init__.py
```python
# Remove in version 0.21

from scipy.sparse.csgraph import connected_components as \
     scipy_connected_components

from sklearn.utils.deprecation import deprecated


@deprecated("sklearn.utils.sparsetools.connected_components was deprecated in "
            "version 0.19 and will be removed in 0.21. Use "
            "scipy.sparse.csgraph.connected_components instead.")
def connected_components(*args, **kwargs):
    return scipy_connected_components(*args, **kwargs)
```
227 - /tmp/repos/scikit-learn/sklearn/utils/fixes.py
```python



euler_gamma = getattr(np, 'euler_gamma',
                      0.577215664901532860606512090082402431)

np_version = _parse_version(np.__version__)
sp_version = _parse_version(scipy.__version__)


# Remove when minimum required NumPy >= 1.10
try:
    if (not np.allclose(np.divide(.4, 1, casting="unsafe"),
                        np.divide(.4, 1, casting="unsafe", dtype=np.float64))
            or not np.allclose(np.divide(.4, 1), .4)):
        raise TypeError('Divide not working with dtype: '
                        'https://github.com/numpy/numpy/issues/3484')
    divide = np.divide

except TypeError:
    # Compat for old versions of np.divide that do not provide support for
    # the dtype args
    def divide(x1, x2, out=None, dtype=None):
        out_orig = out
        if out is None:
            out = np.asarray(x1, dtype=dtype)
            if out is x1:
                out = x1.copy()
        else:
            if out is not x1:
                out[:] = x1
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype)
        out /= x2
        if out_orig is None and np.isscalar(x1):
            out = np.asscalar(out)
        return out


try:
    with warnings.catch_warnings(record=True):
        # Don't raise the numpy deprecation warnings that appear in
        # 1.9, but avoid Python bug due to simplefilter('ignore')
        warnings.simplefilter('always')
        sp.csr_matrix([1.0, 2.0, 3.0]).max(axis=0)
except (TypeError, AttributeError):
    # in scipy < 0.14.0, sparse matrix min/max doesn't accept `axis` argument
    # the following code is taken from the scipy 0.14 codebase

    def _minor_reduce(X, ufunc):
        major_index = np.flatnonzero(np.diff(X.indptr))
        value = ufunc.reduceat(X.data, X.indptr[major_index])
        return major_index, value

    def _min_or_max_axis(X, axis, min_or_max):
        N = X.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = X.shape[1 - axis]
        mat = X.tocsc() if axis == 0 else X.tocsr()
        mat.sum_duplicates()
        major_index, value = _minor_reduce(mat, min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)
        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        from scipy.sparse import coo_matrix
        if axis == 0:
            res = coo_matrix((value, (np.zeros(len(value)), major_index)),
                             dtype=X.dtype, shape=(1, M))
        else:
            res = coo_matrix((value, (major_index, np.zeros(len(value)))),
                             dtype=X.dtype, shape=(M, 1))
        return res.A.ravel()

    def _sparse_min_or_max(X, axis, min_or_max):
        if axis is None:
            if 0 in X.shape:
                raise ValueError("zero-size array to reduction operation")
            zero = X.dtype.type(0)
            if X.nnz == 0:
                return zero
            m = min_or_max.reduce(X.data.ravel())
            if X.nnz != np.product(X.shape):
                m = min_or_max(zero, m)
            return m
        if axis < 0:
            axis += 2
        if (axis == 0) or (axis == 1):
            return _min_or_max_axis(X, axis, min_or_max)
        else:
            raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

    def sparse_min_max(X, axis):
        return (_sparse_min_or_max(X, axis, np.minimum),
                _sparse_min_or_max(X, axis, np.maximum))

else:
    def sparse_min_max(X, axis):
        return (X.min(axis=axis).toarray().ravel(),
                X.max(axis=axis).toarray().ravel())



```
228 - /tmp/repos/scikit-learn/sklearn/inspection/partial_dependence.py
```python
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
229 - /tmp/repos/scikit-learn/examples/classification/plot_classifier_comparison.py
```python
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

```
230 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
```python
"""Solvers for Ridge and LogisticRegression using SAG algorithm"""

# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#
# License: BSD 3 clause

import warnings

import numpy as np

from .base import make_dataset
from .sag_fast import sag
from ..exceptions import ConvergenceWarning
from ..utils import check_array
from ..utils.extmath import row_norms


def get_auto_step_size(max_squared_sum, alpha_scaled, loss, fit_intercept,
                       n_samples=None,
                       is_saga=False):
    """Compute automatic step size for SAG solver

    The step size is set to 1 / (alpha_scaled + L + fit_intercept) where L is
    the max sum of squares for over all samples.

    Parameters
    ----------
    max_squared_sum : float
        Maximum squared sum of X over samples.

    alpha_scaled : float
        Constant that multiplies the regularization term, scaled by
        1. / n_samples, the number of samples.

    loss : string, in {"log", "squared"}
        The loss function used in SAG solver.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) will be
        added to the decision function.

    n_samples : int, optional
        Number of rows in X. Useful if is_saga=True.

    is_saga : boolean, optional
        Whether to return step size for the SAGA algorithm or the SAG
        algorithm.

    Returns
    -------
    step_size : float
        Step size used in SAG solver.

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202
    """
    if loss in ('log', 'multinomial'):
        L = (0.25 * (max_squared_sum + int(fit_intercept)) + alpha_scaled)
    elif loss == 'squared':
        # inverse Lipschitz constant for squared loss
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
    else:
        raise ValueError("Unknown loss function for SAG solver, got %s "
                         "instead of 'log' or 'squared'" % loss)
    if is_saga:
        # SAGA theoretical step size is 1/3L or 1 / (2 * (L + mu n))
        # See Defazio et al. 2014
        mun = min(2 * n_samples * alpha_scaled, L)
        step = 1. / (2 * L + mun)
    else:
        # SAG theoretical step size is 1/16L but it is recommended to use 1 / L
        # see http://www.birs.ca//workshops//2014/14w5003/files/schmidt.pdf,
        # slide 65
        step = 1. / L
    return step


def sag_solver(X, y, sample_weight=None, loss='log', alpha=1., beta=0.,
               max_iter=1000, tol=0.001, verbose=0, random_state=None,
               check_input=True, max_squared_sum=None,
               warm_start_mem=None,
               is_saga=False):
    """SAG solver for Ridge and LogisticRegression

    SAG stands for Stochastic Average Gradient: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a constant learning rate.

    IMPORTANT NOTE: 'sag' solver converges faster on columns that are on the
    same scale. You can normalize the data by using
    sklearn.preprocessing.StandardScaler on your data before passing it to the
    fit method.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values for the features. It will
    fit the data according to squared loss or log loss.

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using the squared euclidean norm L2.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data

    y : numpy array, shape (n_samples,)
        Target values. With loss='multinomial', y must be label encoded
        (see preprocessing.LabelEncoder).

    sample_weight : array-like, shape (n_samples,), optional
        Weights applied to individual samples (1. for unweighted).

    loss : 'log' | 'squared' | 'multinomial'
        Loss function that will be optimized:
        -'log' is the binary logistic loss, as used in LogisticRegression.
        -'squared' is the squared loss, as used in Ridge.
        -'multinomial' is the multinomial logistic loss, as used in
         LogisticRegression.

        .. versionadded:: 0.18
           *loss='multinomial'*

    alpha : float, optional
        L2 regularization term in the objective function
        ``(0.5 * alpha * || W ||_F^2)``. Defaults to 1.

    beta : float, optional
        L1 regularization term in the objective function
        ``(beta * || W ||_1)``. Only applied if ``is_saga`` is set to True.
        Defaults to 0.

    max_iter : int, optional
        The max number of passes over the training data if the stopping
        criteria is not reached. Defaults to 1000.

    tol : double, optional
        The stopping criteria for the weights. The iterations will stop when
        max(change in weights) / max(weights) < tol. Defaults to .001

    verbose : integer, optional
        The verbosity level.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. If None, it will be computed,
        going through all the samples. The value should be precomputed
        to speed up cross validation.

    warm_start_mem : dict, optional
        The initialization parameters used for warm starting. Warm starting is
        currently used in LogisticRegression but not in Ridge.
        It contains:
            - 'coef': the weight vector, with the intercept in last line
                if the intercept is fitted.
            - 'gradient_memory': the scalar gradient for all seen samples.
            - 'sum_gradient': the sum of gradient over all seen samples,
                for each feature.
            - 'intercept_sum_gradient': the sum of gradient over all seen
                samples, for the intercept.
            - 'seen': array of boolean describing the seen samples.
            - 'num_seen': the number of seen samples.

    is_saga : boolean, optional
        Whether to use the SAGA algorithm or the SAG algorithm. SAGA behaves
        better in the first epochs, and allow for l1 regularisation.

    Returns
    -------
    coef_ : array, shape (n_features)
        Weight vector.

    n_iter_ : int
        The number of full pass on all samples.

    warm_start_mem : dict
        Contains a 'coef' key with the fitted result, and possibly the
        fitted intercept at the end of the array. Contains also other keys
        used for warm starting.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> X = np.random.randn(n_samples, n_features)
    >>> y = np.random.randn(n_samples)
    >>> clf = linear_model.Ridge(solver='sag')
    >>> clf.fit(X, y)
    ... #doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='sag', tol=0.001)

    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.LogisticRegression(solver='sag')
    >>> clf.fit(X, y)
    ... #doctest: +NORMALIZE_WHITESPACE
    LogisticRegression(C=1.0, class_weight=None, dual=False,
        fit_intercept=True, intercept_scaling=1, max_iter=100,
        multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
        solver='sag', tol=0.0001, verbose=0, warm_start=False)

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202

    See also
    --------
    Ridge, SGDRegressor, ElasticNet, Lasso, SVR, and
    LogisticRegression, SGDClassifier, LinearSVC, Perceptron
    
```
231 - /tmp/repos/scikit-learn/examples/cluster/plot_agglomerative_clustering.py
```python
"""
Agglomerative clustering with and without structure
===================================================

This example shows the effect of imposing a connectivity graph to capture
local structure in the data. The graph is simply the graph of 20 nearest
neighbors.

Two consequences of imposing a connectivity can be seen. First clustering
with a connectivity matrix is much faster.

Second, when using a connectivity matrix, single, average and complete
linkage are unstable and tend to create a few clusters that grow very
quickly. Indeed, average and complete linkage fight this percolation behavior
by considering all the distances between two clusters when merging them (
while single linkage exaggerates the behaviour by considering only the
shortest distance between clusters). The connectivity graph breaks this
mechanism for average and complete linkage, making them resemble the more
brittle single linkage. This effect is more pronounced for very sparse graphs
(try decreasing the number of neighbors in kneighbors_graph) and with
complete linkage. In particular, having a very small number of neighbors in
the graph, imposes a geometry that is close to that of single linkage,
which is well known to have this percolation instability. """
# Authors: Gael Varoquaux, Nelle Varoquaux
# License: BSD 3 clause

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Generate sample data
n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)


X = np.concatenate((x, y))
X += .7 * np.random.randn(2, n_samples)
X = X.T

# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data
knn_graph = kneighbors_graph(X, 30, include_self=False)

for connectivity in (None, knn_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average',
                                         'complete',
                                         'ward',
                                         'single')):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                        cmap=plt.cm.spectral)
            plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)


plt.show()
```
232 - /tmp/repos/scikit-learn/examples/preprocessing/plot_all_scaling.py
```python



########################################################################
# .. _results:
#
# Original data
# -------------
#
# Each transformation is plotted showing two transformed features, with the
# left plot showing the entire dataset, and the right zoomed-in to show the
# dataset without the marginal outliers. A large majority of the samples are
# compacted to a specific range, [0, 10] for the median income and [0, 6] for
# the number of households. Note that there are some marginal outliers (some
# blocks have more than 1200 households). Therefore, a specific pre-processing
# can be very beneficial depending of the application. In the following, we
# present some insights and behaviors of those pre-processing methods in the
# presence of marginal outliers.

make_plot(0)

#######################################################################
# StandardScaler
# --------------
#
# ``StandardScaler`` removes the mean and scales the data to unit variance.
# However, the outliers have an influence when computing the empirical mean and
# standard deviation which shrink the range of the feature values as shown in
# the left figure below. Note in particular that because the outliers on each
# feature have different magnitudes, the spread of the transformed data on
# each feature is very different: most of the data lie in the [-2, 4] range for
# the transformed median income feature while the same data is squeezed in the
# smaller [-0.2, 0.2] range for the transformed number of households.
#
# ``StandardScaler`` therefore cannot guarantee balanced feature scales in the
# presence of outliers.

make_plot(1)

##########################################################################
# MinMaxScaler
# ------------
#
# ``MinMaxScaler`` rescales the data set such that all feature values are in
# the range [0, 1] as shown in the right panel below. However, this scaling
# compress all inliers in the narrow range [0, 0.005] for the transformed
# number of households.
#
# As ``StandardScaler``, ``MinMaxScaler`` is very sensitive to the presence of
# outliers.

make_plot(2)

#############################################################################
# MaxAbsScaler
# ------------
#
# ``MaxAbsScaler`` differs from the previous scaler such that the absolute
# values are mapped in the range [0, 1]. On positive only data, this scaler
# behaves similarly to ``MinMaxScaler`` and therefore also suffers from the
# presence of large outliers.

make_plot(3)

##############################################################################
# RobustScaler
# ------------
#
# Unlike the previous scalers, the centering and scaling statistics of this
# scaler are based on percentiles and are therefore not influenced by a few
# number of very large marginal outliers. Consequently, the resulting range of
# the transformed feature values is larger than for the previous scalers and,
# more importantly, are approximately similar: for both features most of the
# transformed values lie in a [-2, 3] range as seen in the zoomed-in figure.
# Note that the outliers themselves are still present in the transformed data.
# If a separate outlier clipping is desirable, a non-linear transformation is
# required (see below).

make_plot(4)

##############################################################################
# PowerTransformer (Box-Cox)
# --------------------------
#
# ``PowerTransformer`` applies a power transformation to each
# feature to make the data more Gaussian-like. Currently,
# ``PowerTransformer`` implements the Box-Cox transform. The Box-Cox transform
# finds the optimal scaling factor to stabilize variance and mimimize skewness
# through maximum likelihood estimation. By default, ``PowerTransformer`` also
# applies zero-mean, unit variance normalization to the transformed output.
# Note that Box-Cox can only be applied to positive, non-zero data. Income and
# number of households happen to be strictly positive, but if negative values
# are present, a constant can be added to each feature to shift it into the
# positive range - this is known as the two-parameter Box-Cox transform.

make_plot(5)

##############################################################################
# QuantileTransformer (Gaussian output)
# -------------------------------------
#
# ``QuantileTransformer`` has an additional ``output_distribution`` parameter
# allowing to match a Gaussian distribution instead of a uniform distribution.
# Note that this non-parametetric transformer introduces saturation artifacts
# for extreme values.

make_plot(6)

###################################################################
# QuantileTransformer (uniform output)
# ------------------------------------
#
# ``QuantileTransformer`` applies a non-linear transformation such that the
# probability density function of each feature will be mapped to a uniform
# distribution. In this case, all the data will be mapped in the range [0, 1],
# even the outliers which cannot be distinguished anymore from the inliers.
#
# As ``RobustScaler``, ``QuantileTransformer`` is robust to outliers in the

```
233 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        dot_vals = np.multiply(W[ii, :], H.T[jj, :]).sum(axis=1)
        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def _compute_regularization(alpha, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H"""
    alpha_H = 0.
    alpha_W = 0.
    if regularization in ('both', 'components'):
        alpha_H = float(alpha)
    if regularization in ('both', 'transformation'):
        alpha_W = float(alpha)

    l1_reg_W = alpha_W * l1_ratio
    l1_reg_H = alpha_H * l1_ratio
    l2_reg_W = alpha_W * (1. - l1_ratio)
    l2_reg_H = alpha_H * (1. - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _check_string_param(solver, regularization, beta_loss, init):
    allowed_solver = ('cd', 'mu')
    if solver not in allowed_solver:
        raise ValueError(
            'Invalid solver parameter: got %r instead of one of %r' %
            (solver, allowed_solver))

    allowed_regularization = ('both', 'components', 'transformation', None)
    if regularization not in allowed_regularization:
        raise ValueError(
            'Invalid regularization parameter: got %r instead of one of %r' %
            (regularization, allowed_regularization))

    # 'mu' is the only solver that handles other beta losses than 'frobenius'
    if solver != 'mu' and beta_loss not in (2, 'frobenius'):
        raise ValueError(
            'Invalid beta_loss parameter: solver %r does not handle beta_loss'
            ' = %r' % (solver, beta_loss))

    if solver == 'mu' and init == 'nndsvd':
        warnings.warn("The multiplicative update ('mu') solver cannot update "
                      "zeros present in the initialization, and so leads to "
                      "poorer results when used jointly with init='nndsvd'. "
                      "You may try init='nndsvda' or init='nndsvdar' instead.",
                      UserWarning)

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None):
    
```
234 - /tmp/repos/scikit-learn/examples/applications/plot_outlier_detection_housing.py
```python
plt.scatter(X1[:, 0], X1[:, 1], color='black')
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate("several confounded points", xy=(24, 19),
             xycoords="data", textcoords="data",
             xytext=(13, 10), bbox=bbox_args, arrowprops=arrow_args)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend((legend1_values_list[0].collections[0],
            legend1_values_list[1].collections[0],
            legend1_values_list[2].collections[0]),
           (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2]),
           loc="upper center",
           prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("accessibility to radial highways")
plt.xlabel("pupil-teacher ratio by town")

legend2_values_list = list(legend2.values())
legend2_keys_list = list(legend2.keys())

plt.figure(2)  # "banana" shape
plt.title("Outlier detection on a real data set (boston housing)")
plt.scatter(X2[:, 0], X2[:, 1], color='black')
plt.xlim((xx2.min(), xx2.max()))
plt.ylim((yy2.min(), yy2.max()))
plt.legend((legend2_values_list[0].collections[0],
            legend2_values_list[1].collections[0],
            legend2_values_list[2].collections[0]),
           (legend2_keys_list[0], legend2_keys_list[1], legend2_keys_list[2]),
           loc="upper center",
           prop=matplotlib.font_manager.FontProperties(size=12))
plt.ylabel("% lower status of the population")
plt.xlabel("average number of rooms per dwelling")

plt.show()

```
235 - /tmp/repos/scikit-learn/examples/model_selection/plot_precision_recall.py
```python
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,
                                                    random_state=random_state)

# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)

###############################################################################
# Compute the average precision score
# ...................................
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

###############################################################################
# Plot the Precision-Recall curve
# ................................
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.externals.funcsigs import signature

precision, recall, _ = precision_recall_curve(y_test, y_score)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

###############################################################################
# In multi-label settings
# ------------------------
#
# Create multi-label data, fit, and predict
# ...........................................
#
# We create a multi-label dataset, to illustrate the precision-recall in
# multi-label settings

from sklearn.preprocessing import label_binarize

# Use label_binarize to be multi-label like settings
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)


###############################################################################
# The average precision score in multi-label settings
# ....................................................
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

###############################################################################
# Plot the micro-averaged Precision-Recall curve
# ...............................................
#

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

###############################################################################
# Plot Precision-Recall curve for each class and iso-f1 curves
# .............................................................
#
from itertools import cycle
# setup plot details

```
236 - /tmp/repos/scikit-learn/examples/decomposition/plot_incremental_pca.py
```python
"""

===============
Incremental PCA
===============

Incremental principal component analysis (IPCA) is typically used as a
replacement for principal component analysis (PCA) when the dataset to be
decomposed is too large to fit in memory. IPCA builds a low-rank approximation
for the input data using an amount of memory which is independent of the
number of input data samples. It is still dependent on the input data features,
but changing the batch size allows for control of memory usage.

This example serves as a visual check that IPCA is able to find a similar
projection of the data to PCA (to a sign flip), while only processing a
few samples at a time. This can be considered a "toy example", as IPCA is
intended for large datasets which do not fit in main memory, requiring
incremental approaches.

"""
print(__doc__)

# Authors: Kyle Kastner
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ['navy', 'turquoise', 'darkorange']

for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    color=color, lw=2, label=target_name)

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error "
                  "%.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()
```
237 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
def _multiplicative_update_w(X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
                             H_sum=None, HHt=None, XHt=None, update_H=True):
    """update W in Multiplicative Update NMF"""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        # Denominator
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        # if X is sparse, compute WH only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1. < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2. < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T)

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    return delta_W, H_sum, HHt, XHt



```
238 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
```python
X, y = indexable(X, y)
    # Make a list since we will be iterating multiple times over the folds
    cv = list(check_cv(cv, X, y, classifier=is_classifier(estimator)))
    scorer = check_scoring(estimator, scoring=scoring)

    # HACK as long as boolean indices are allowed in cv generators
    if cv[0][0].dtype == bool:
        new_cv = []
        for i in range(len(cv)):
            new_cv.append((np.nonzero(cv[i][0])[0], np.nonzero(cv[i][1])[0]))
        cv = new_cv

    n_max_training_samples = len(cv[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)
    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel(delayed(_incremental_fit_estimator)(
            clone(estimator), X, y, classes, train, test, train_sizes_abs,
            scorer, verbose) for train, test in cv)
    else:
        out = parallel(delayed(_fit_and_score)(
            clone(estimator), X, y, scorer, train[:n_train_samples], test,
            verbose, parameters=None, fit_params=None, return_train_score=True,
            error_score=error_score)
            for train, test in cv for n_train_samples in train_sizes_abs)
        out = np.array(out)[:, :2]
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)

    out = np.asarray(out).transpose((2, 1, 0))

    return train_sizes_abs, out[0], out[1]



```
239 - /tmp/repos/scikit-learn/examples/ensemble/plot_ensemble_oob.py
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

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

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
240 - /tmp/repos/scikit-learn/sklearn/cluster/hierarchical.py
```python
"""
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.

    affinity : string or callable, default: "euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or 'precomputed'.
        If linkage is "ward", only "euclidean" is accepted.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, optional
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is None, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : bool or 'auto' (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. This option is
        useful only when specifying a connectivity matrix. Note also that
        when varying the number of clusters and using caching, it may
        be advantageous to compute the full tree.

    linkage : {"ward", "complete", "average", "single"}, optional \
            (default="ward")
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.

    pooling_func : callable, default='deprecated'
        Ignored.

        .. deprecated:: 0.20
            ``pooling_func`` has been deprecated in 0.20 and will be removed
            in 0.22.

    Attributes
    ----------
    labels_ : array [n_samples]
        cluster labels for each point

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_components_ : int
        The estimated number of connected components in the graph.

    children_ : array-like, shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`

    """

    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=None,
                 connectivity=None, compute_full_tree='auto',
                 linkage='ward', pooling_func='deprecated'):
        self.n_clusters = n_clusters
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.pooling_func = pooling_func

    
```
241 - /tmp/repos/scikit-learn/examples/linear_model/plot_sgd_comparison.py
```python
"""
==================================
Comparing various online solvers
==================================

An example showing how different online solvers perform
on the hand-written digits dataset.

"""
# Author: Rob Zinkov <rob at zinkov dot com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
digits = datasets.load_digits()
X, y = digits.data, digits.target

classifiers = [
    ("SGD", SGDClassifier(max_iter=100)),
    ("ASGD", SGDClassifier(average=True, max_iter=100)),
    ("Perceptron", Perceptron(tol=1e-3)),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
```
242 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
"""GraphicalLasso: sparse inverse covariance estimation with an l1-penalized
estimator.
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause
# Copyright: INRIA
import warnings
import operator
import sys
import time

import numpy as np
from scipy import linalg

from .empirical_covariance_ import (empirical_covariance, EmpiricalCovariance,
                                    log_likelihood)

from ..exceptions import ConvergenceWarning
from ..utils.validation import check_random_state, check_array
from ..utils import deprecated
from ..linear_model import lars_path
from ..linear_model import cd_fast
from ..model_selection import check_cv, cross_val_score
from ..externals.joblib import Parallel, delayed
import collections


# Helper functions to compute the objective and dual objective functions
# of the l1-penalized estimator
def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = - 2. * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum()
                     - np.abs(np.diag(precision_)).sum())
    return cost


def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum()
                    - np.abs(np.diag(precision_)).sum())
    return gap


def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : 2D array, (n_features, n_features)
        The sample covariance matrix

    Notes
    -----

    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.

    """
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


# The g-lasso algorithm

def graphical_lasso(emp_cov, alpha, cov_init=None, mode='cd', tol=1e-4,
                    enet_tol=1e-4, max_iter=100, verbose=False,
                    return_costs=False, eps=np.finfo(np.float64).eps,
                    return_n_iter=False):
    
```
243 - /tmp/repos/scikit-learn/examples/model_selection/plot_confusion_matrix.py
```python
"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

```
244 - /tmp/repos/scikit-learn/sklearn/cross_decomposition/__init__.py
```python
from .pls_ import *  # noqa
from .cca_ import *  # noqa
```
245 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/grower.py
```python
def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self, bin_thresholds=None):
        """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        bin_thresholds : array-like of floats, optional (default=None)
            The actual thresholds values of each bin.

        Returns
        -------
        A TreePredictor object.
        """
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        _fill_predictor_node_array(predictor_nodes, self.root,
                                   bin_thresholds=bin_thresholds)
        return TreePredictor(predictor_nodes)


def _fill_predictor_node_array(predictor_nodes, grower_node,
                               bin_thresholds, next_free_idx=0):
    """Helper used in make_predictor to set the TreePredictor fields."""
    node = predictor_nodes[next_free_idx]
    node['count'] = grower_node.n_samples
    node['depth'] = grower_node.depth
    if grower_node.split_info is not None:
        node['gain'] = grower_node.split_info.gain
    else:
        node['gain'] = -1

    if grower_node.value is not None:
        # Leaf node
        node['is_leaf'] = True
        node['value'] = grower_node.value
        return next_free_idx + 1
    else:
        # Decision node
        split_info = grower_node.split_info
        feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
        node['feature_idx'] = feature_idx
        node['bin_threshold'] = bin_idx
        if bin_thresholds is not None:
            threshold = bin_thresholds[feature_idx][bin_idx]
            node['threshold'] = threshold
        next_free_idx += 1

        node['left'] = next_free_idx
        next_free_idx = _fill_predictor_node_array(
            predictor_nodes, grower_node.left_child,
            bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)

        node['right'] = next_free_idx
        return _fill_predictor_node_array(
            predictor_nodes, grower_node.right_child,
            bin_thresholds=bin_thresholds, next_free_idx=next_free_idx)

```
246 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def plot_results(results_df, plot_name):
    if results_df is None:
        return None

    plt.figure(figsize=(16, 6))
    colors = 'bgr'
    markers = 'ovs'
    ax = plt.subplot(1, 3, 1)
    for i, init in enumerate(np.unique(results_df['init'])):
        plt.subplot(1, 3, i + 1, sharex=ax, sharey=ax)
        for j, method in enumerate(np.unique(results_df['method'])):
            mask = np.logical_and(results_df['init'] == init,
                                  results_df['method'] == method)
            selected_items = results_df[mask]

            plt.plot(selected_items['time'], selected_items['loss'],
                     color=colors[j % len(colors)], ls='-',
                     marker=markers[j % len(markers)],
                     label=method)

        plt.legend(loc=0, fontsize='x-small')
        plt.xlabel("Time (s)")
        plt.ylabel("loss")
        plt.title("%s" % init)
    plt.suptitle(plot_name, fontsize=16)


@ignore_warnings(category=ConvergenceWarning)
# use joblib to cache the results.
# X_shape is specified in arguments for avoiding hashing X
@mem.cache(ignore=['X', 'W0', 'H0'])
def bench_one(name, X, W0, H0, X_shape, clf_type, clf_params, init,
              n_components, random_state):
    W = W0.copy()
    H = H0.copy()

    clf = clf_type(**clf_params)
    st = time()
    W = clf.fit_transform(X, W=W, H=H)
    end = time()
    H = clf.components_

    this_loss = _beta_divergence(X, W, H, 2.0, True)
    duration = end - st
    return this_loss, duration


def run_bench(X, clfs, plot_name, n_components, tol, alpha, l1_ratio):
    start = time()
    results = []
    for name, clf_type, iter_range, clf_params in clfs:
        print("Training %s:" % name)
        for rs, init in enumerate(('nndsvd', 'nndsvdar', 'random')):
            print("    %s %s: " % (init, " " * (8 - len(init))), end="")
            W, H = _initialize_nmf(X, n_components, init, 1e-6, rs)

            for max_iter in iter_range:
                clf_params['alpha'] = alpha
                clf_params['l1_ratio'] = l1_ratio
                clf_params['max_iter'] = max_iter
                clf_params['tol'] = tol
                clf_params['random_state'] = rs
                clf_params['init'] = 'custom'
                clf_params['n_components'] = n_components

                this_loss, duration = bench_one(name, X, W, H, X.shape,
                                                clf_type, clf_params,
                                                init, n_components, rs)

                init_name = "init='%s'" % init
                results.append((name, this_loss, duration, init_name))
                # print("loss: %.6f, time: %.3f sec" % (this_loss, duration))
                print(".", end="")
                sys.stdout.flush()
            print(" ")

    # Use a panda dataframe to organize the results
    results_df = pandas.DataFrame(results,
                                  columns="method loss time init".split())
    print("Total time = %0.3f sec\n" % (time() - start))

    # plot the results
    plot_results(results_df, plot_name)
    return results_df


def load_20news():
    print("Loading 20 newsgroups dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(dataset.data)
    return tfidf


def load_faces():
    print("Loading Olivetti face dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    return faces.data



```
247 - /tmp/repos/scikit-learn/sklearn/metrics/cluster/bicluster.py
```python
from __future__ import division

import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils.validation import check_consistent_length, check_array

__all__ = ["consensus_score"]


def _check_rows_and_columns(a, b):
    """Unpacks the row and column arrays and checks their shape."""
    check_consistent_length(*a)
    check_consistent_length(*b)
    checks = lambda x: check_array(x, ensure_2d=False)
    a_rows, a_cols = map(checks, a)
    b_rows, b_cols = map(checks, b)
    return a_rows, a_cols, b_rows, b_cols


def _jaccard(a_rows, a_cols, b_rows, b_cols):
    """Jaccard coefficient on the elements of the two biclusters."""
    intersection = ((a_rows * b_rows).sum() *
                    (a_cols * b_cols).sum())

    a_size = a_rows.sum() * a_cols.sum()
    b_size = b_rows.sum() * b_cols.sum()

    return intersection / (a_size + b_size - intersection)


def _pairwise_similarity(a, b, similarity):
    """Computes pairwise similarity matrix.

    result[i, j] is the Jaccard coefficient of a's bicluster i and b's
    bicluster j.

    """
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)
    n_a = a_rows.shape[0]
    n_b = b_rows.shape[0]
    result = np.array(list(list(similarity(a_rows[i], a_cols[i],
                                           b_rows[j], b_cols[j])
                                for j in range(n_b))
                           for i in range(n_a)))
    return result


def consensus_score(a, b, similarity="jaccard"):
    """The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed. Then the
    best matching between sets is found using the Hungarian algorithm.
    The final score is the sum of similarities divided by the size of
    the larger set.

    Read more in the :ref:`User Guide <biclustering>`.

    Parameters
    ----------
    a : (rows, columns)
        Tuple of row and column indicators for a set of biclusters.

    b : (rows, columns)
        Another set of biclusters like ``a``.

    similarity : string or function, optional, default: "jaccard"
        May be the string "jaccard" to use the Jaccard coefficient, or
        any function that takes four arguments, each of which is a 1d
        indicator vector: (a_rows, a_columns, b_rows, b_columns).

    References
    ----------

    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.

    """
    if similarity == "jaccard":
        similarity = _jaccard
    matrix = _pairwise_similarity(a, b, similarity)
    indices = linear_assignment(1. - matrix)
    n_a = len(a[0])
    n_b = len(b[0])
    return matrix[indices[:, 0], indices[:, 1]].sum() / max(n_a, n_b)

```
248 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
```python
"""Learning curve.

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :func:`sklearn.model_selection.learning_curve` instead.

    Determines cross-validated training and test scores for different training
    set sizes.

    A cross-validation generator splits the whole dataset k times in training
    and test data. Subsets of the training set with varying sizes will be used
    to train the estimator and a score for each training subset size and the
    test set will be computed. Afterwards, the scores will be averaged over
    all k runs for each training subset size.

    Read more in the :ref:`User Guide <learning_curves>`.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    exploit_incremental_learning : boolean, optional, default: False
        If the estimator supports incremental learning, this will be
        used to speed up fitting for different training set sizes.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Returns
    -------
    train_sizes_abs : array, shape = (n_unique_ticks,), dtype int
        Numbers of training examples that has been used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    Notes
    -----
    See :ref:`examples/model_selection/plot_learning_curve.py
    <sphx_glr_auto_examples_model_selection_plot_learning_curve.py>`
    """
    if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError("An estimator must support the partial_fit interface "
                         "to exploit incremental learning")

    
```
249 - /tmp/repos/scikit-learn/examples/classification/plot_lda_qda.py
```python


for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(store_covariances=True)
    y_pred = qda.fit(X, y).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')
plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant'
             'Analysis')
plt.show()

```
