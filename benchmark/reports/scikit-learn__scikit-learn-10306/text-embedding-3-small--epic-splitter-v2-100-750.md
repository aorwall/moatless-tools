0 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
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
```
1 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """
```
2 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
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
```
3 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
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
```
4 - /tmp/repos/scikit-learn/sklearn/exceptions.py
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
```
5 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
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
6 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
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



```
7 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
# assert_raises_regexp is deprecated in Python 3.4 in favor of
# assert_raises_regex but lets keep the backward compat in scikit-learn with
# the old name for now
assert_raises_regexp = assert_raises_regex


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



```
8 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
# assert_raises_regexp is deprecated in Python 3.4 in favor of
# assert_raises_regex but lets keep the backward compat in scikit-learn with
# the old name for now
assert_raises_regexp = assert_raises_regex


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



```
9 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
# assert_raises_regexp is deprecated in Python 3.4 in favor of
# assert_raises_regex but lets keep the backward compat in scikit-learn with
# the old name for now
assert_raises_regexp = assert_raises_regex


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



```
10 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
11 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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
12 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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
**13 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class _CFSubcluster(object):


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
```
14 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
**15 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
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
```
16 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
17 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
18 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
19 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
20 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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

    Parameters
    ----------
    func
    *args
    **kw
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
21 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
22 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    # warn_on_dtype deprecation
    if warn_on_dtype is not None:
        warnings.warn(
            "'warn_on_dtype' is deprecated in version 0.21 and will be "
            "removed in 0.23. Don't set `warn_on_dtype` to remove this "
            "warning.",
            DeprecationWarning)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        dtypes_orig = np.array(array.dtypes)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

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
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    
```
23 - /tmp/repos/scikit-learn/sklearn/exceptions.py
```python
class NonBLASDotWarning(EfficiencyWarning):
    """Warning used when the dot operation does not use BLAS.

    This warning is used to notify the user that BLAS was not used for dot
    operation and hence the efficiency may be affected.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation, extends EfficiencyWarning.
    """
```
24 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
25 - /tmp/repos/scikit-learn/sklearn/utils/deprecation.py
```python
import sys
import warnings
import functools

__all__ = ["deprecated", "DeprecationDict"]
```
26 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
        X_m -= X_m.min()
        X_b -= X_b.min()

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    problems = [(X_b, y_b)]
    tags = _safe_tags(classifier_orig)
    if not tags['binary_only']:
        problems.append((X_m, y_m))

    for (X, y) in problems:
        
```
27 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
        X_m -= X_m.min()
        X_b -= X_b.min()

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
28 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
# ...


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
        X_m -= X_m.min()
        X_b -= X_b.min()

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        
```
29 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
30 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
31 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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
32 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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
33 - /tmp/repos/scikit-learn/sklearn/utils/_joblib.py
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
34 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
35 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
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

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
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

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    
```
36 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
            H_sum, HHt, XHt, update_H)
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H,
                                               l2_reg_H, gamma)
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, n_iter
# ...
```
37 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
            H_sum, HHt, XHt, update_H)
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H,
                                               l2_reg_H, gamma)
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, n_iter
# ...
```
38 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
            H_sum, HHt, XHt, update_H)
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H,
                                               l2_reg_H, gamma)
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, n_iter
# ...
```
39 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
40 - /tmp/repos/scikit-learn/sklearn/externals/joblib/memory.py
```python
class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


def _get_func_fullname(func):
    """Compute the part of part associated with a function.

    See code of_cache_key_to_dir() for details
    """
    modules, funcname = get_func_name(func)
    modules.append(funcname)
    return os.path.join(*modules)
```
41 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
class BaseLibSVM(six.with_metaclass(ABCMeta, BaseEstimator)):


    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).'
                          '  Consider pre-processing your data with'
                          ' StandardScaler or MinMaxScaler.'
                          % self.max_iter, ConvergenceWarning)
```
42 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
43 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python



try:
    import pytest

    skip_if_32bit = pytest.mark.skipif(8 * struct.calcsize("P") == 32,
                                       reason='skipped on 32bit platforms')
    skip_travis = pytest.mark.skipif(os.environ.get('TRAVIS') == 'true',
                                     reason='skip on travis')

    #  Decorator for tests involving both BLAS calls and multiprocessing.
    #
    #  Under POSIX (e.g. Linux or OSX), using multiprocessing in conjunction
    #  with some implementation of BLAS (or other libraries that manage an
    #  internal posix thread pool) can cause a crash or a freeze of the Python
    #  process.
    #
    #  In practice all known packaged distributions (from Linux distros or
    #  Anaconda) of BLAS under Linux seems to be safe. So we this problem seems
    #  to only impact OSX users.
    #
    #  This wrapper makes it possible to skip tests that can possibly cause
    #  this crash under OS X with.
    #
    #  Under Python 3.4+ it is possible to use the `forkserver` start method
    #  for multiprocessing to avoid this issue. However it can cause pickling
    #  errors on interactively defined functions. It therefore not enabled by
    #  default.

    if_safe_multiprocessing_with_blas = pytest.mark.skipif(
            sys.platform == 'darwin',
            reason="Possible multi-process bug with some BLAS")
except ImportError:
    pass


def clean_warning_registry():
    """Clean Python warning registry for easier testing of warning messages.

    We may not need to do this any more when getting rid of Python 2, not
    entirely sure. See https://bugs.python.org/issue4180 and
    https://bugs.python.org/issue21724 for more details.

    """
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


def check_skip_network():
    if int(os.environ.get('SKLEARN_SKIP_NETWORK_TESTS', 0)):
        raise SkipTest("Text tutorial requires large dataset download")


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
        self.mmap_mode = mmap_mode
        self.data = data

    def __enter__(self):
        data_read_only, self.temp_folder = create_memmap_backed_data(
            self.data, mmap_mode=self.mmap_mode, return_folder=True)
        return data_read_only

    def __exit__(self, exc_type, exc_val, exc_tb):
        _delete_folder(self.temp_folder)



```
44 - /tmp/repos/scikit-learn/sklearn/externals/_joblib/memory.py
```python
class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


def _get_func_fullname(func):
    """Compute the part of part associated with a function.

    See code of_cache_key_to_dir() for details
    """
    modules, funcname = get_func_name(func)
    modules.append(funcname)
    return os.path.join(*modules)
```
45 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
46 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
47 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
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
```
48 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""Utilities for input validation"""

# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
# License: BSD 3 clause

import warnings
import numbers

import numpy as np
import scipy.sparse as sp
from numpy.core.numeric import ComplexWarning

from ..externals import six
from ..utils.fixes import signature
from .. import get_config as _get_config
from ..exceptions import NonBLASDotWarning
from ..exceptions import NotFittedError
from ..exceptions import DataConversionWarning
from ..externals.joblib import Memory


FLOAT_DTYPES = (np.float64, np.float32, np.float16)

# Silenced by default to reduce verbosity. Turn on at runtime for
# performance profiling.
warnings.simplefilter('ignore', NonBLASDotWarning)
```
49 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
50 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
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

    
```
51 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
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
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
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

    
```
52 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
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

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

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
**53 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):


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
54 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

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

    
```
55 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
56 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
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

            if not isinstance(n_alphas, Sequence):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0),
                                     n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print('[GraphicalLassoCV] Done refinement % 2i out of'
                      ' %i: % 3is' % (i + 1, n_refinements, time.time() - t0))

        path = list(zip(*path))
        
```
57 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
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

            if not isinstance(n_alphas, Sequence):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0),
                                     n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print('[GraphicalLassoCV] Done refinement % 2i out of'
                      ' %i: % 3is' % (i + 1, n_refinements, time.time() - t0))

        path = list(zip(*path))
        
```
58 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
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



```
59 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
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

    
```
60 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
61 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
62 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
63 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
64 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
65 - /tmp/repos/scikit-learn/sklearn/utils/deprecation.py
```python
class DeprecationDict(dict):
    """A dict which raises a warning when some keys are looked up

    Note, this does not raise a warning for __contains__ and iteration.

    It also will raise a warning even after the key has been manually set by
    the user.
    """
    def __init__(self, *args, **kwargs):
        self._deprecations = {}
        super(DeprecationDict, self).__init__(*args, **kwargs)
```
66 - /tmp/repos/scikit-learn/examples/cluster/plot_linkage_comparison.py
```python
# ..."""
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
                " > 1. Completing it to avoid stopping the tree early."
```
67 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
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

    
```
68 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
69 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ('WardAgglomeration', "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter
```
70 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
71 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py
```python
# Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
# ...
```
72 - /tmp/repos/scikit-learn/sklearn/impute/_iterative.py
```python
from time import time
from distutils.version import LooseVersion
from collections import namedtuple
import warnings

import scipy
from scipy import stats
import numpy as np

from ..base import clone, BaseEstimator, TransformerMixin
from ..exceptions import ConvergenceWarning
from ..preprocessing import normalize
from ..utils import check_array, check_random_state, safe_indexing
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
from ..utils import is_scalar_nan

from ._base import (_get_mask, MissingIndicator, SimpleImputer,
                    _check_inputs_dtype)


_ImputerTriplet = namedtuple('_ImputerTriplet', ['feat_idx',
                                                 'neighbor_feat_idx',
                                                 'estimator'])
```
73 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
74 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
for i in range(n_refinements):
            with warnings.catch_warnings():
                # No need to see the convergence warnings on this grid:
                # they will always be points that will not converge
                # during the cross-validation
                warnings.simplefilter('ignore', ConvergenceWarning)
                # Compute the cross-validated loss on the current grid

                # NOTE: Warm-restarting graph_lasso_path has been tried, and
                # this did not allow to gain anything (same execution time with
                # or without).
                this_path = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )(delayed(graph_lasso_path)(X[train], alphas=alphas,
                                            X_test=X[test], mode=self.mode,
                                            tol=self.tol,
                                            enet_tol=self.enet_tol,
                                            max_iter=int(.1 * self.max_iter),
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
                print('[GraphLassoCV] Done refinement % 2i out of %i: % 3is'
                      % (i + 1, n_refinements, time.time() - t0))

        path = list(zip(*path))
        
```
75 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
76 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sample_weights_invariance(name, estimator_orig):
    # check that the estimators yield same results for
    # unit weights and no weights
    if (has_fit_parameter(estimator_orig, "sample_weight") and
            not (hasattr(estimator_orig, "_pairwise")
                 and estimator_orig._pairwise)):
        # We skip pairwise because the data is not pairwise

        estimator1 = clone(estimator_orig)
        estimator2 = clone(estimator_orig)
        set_random_state(estimator1, random_state=0)
        set_random_state(estimator2, random_state=0)

        X = np.array([[1, 3], [1, 3], [1, 3], [1, 3],
                      [2, 1], [2, 1], [2, 1], [2, 1],
                      [3, 3], [3, 3], [3, 3], [3, 3],
                      [4, 1], [4, 1], [4, 1], [4, 1]], dtype=np.dtype('float'))
        y = np.array([1, 1, 1, 1, 2, 2, 2, 2,
                      1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype('int'))
        y = enforce_estimator_tags_y(estimator1, y)

        estimator1.fit(X, y=y, sample_weight=np.ones(shape=len(y)))
        estimator2.fit(X, y=y, sample_weight=None)

        for method in ["predict", "transform"]:
            if hasattr(estimator_orig, method):
                X_pred1 = getattr(estimator1, method)(X)
                X_pred2 = getattr(estimator2, method)(X)
                if sparse.issparse(X_pred1):
                    X_pred1 = X_pred1.toarray()
                    X_pred2 = X_pred2.toarray()
                assert_allclose(X_pred1, X_pred2,
                                err_msg="For %s sample_weight=None is not"
                                        " equivalent to sample_weight=ones"
                                        % name)
```
77 - /tmp/repos/scikit-learn/sklearn/utils/deprecation.py
```python
class DeprecationDict(dict):


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
78 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
79 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    
```
80 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
81 - /tmp/repos/scikit-learn/examples/cluster/plot_kmeans_assumptions.py
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
82 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array, "__array__"):
        dtypes_orig = np.array(array.dtypes)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

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

    
```
83 - /tmp/repos/scikit-learn/sklearn/cluster/optics_.py
```python
for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        # just continue if steep_index has been a part of a discovered xward
        # area.
        if steep_index < index:
            continue

        mib = max(mib, np.max(reachability_plot[index:steep_index + 1]))

        # steep downward areas
        if steep_downward[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xi_complement,
                                       reachability_plot)
            D_start = steep_index
            D_end = _extend_region(steep_downward, upward,
                                   D_start, min_samples)
            D = {'start': D_start, 'end': D_end, 'mib': 0.}
            sdas.append(D)
            index = D_end + 1
            mib = reachability_plot[index]

        # steep upward areas
        else:
            sdas = _update_filter_sdas(sdas, mib, xi_complement,
                                       reachability_plot)
            U_start = steep_index
            U_end = _extend_region(steep_upward, downward, U_start,
                                   min_samples)
            index = U_end + 1
            mib = reachability_plot[index]

            U_clusters = []
            for D in sdas:
                c_start = D['start']
                c_end = U_end

                # line (**), sc2*
                if reachability_plot[c_end + 1] * xi_complement < D['mib']:
                    continue

                # Definition 11: criterion 4
                D_max = reachability_plot[D['start']]
                if D_max * xi_complement >= reachability_plot[c_end + 1]:
                    # Find the first index from the left side which is almost
                    # at the same level as the end of the detected cluster.
                    while (reachability_plot[c_start + 1] >
                           reachability_plot[c_end + 1]
                           and c_start < D['end']):
                        c_start += 1
                elif reachability_plot[c_end + 1] * xi_complement >= D_max:
                    # Find the first index from the right side which is almost
                    # at the same level as the beginning of the detected
                    # cluster.
                    # Our implementation corrects a mistake in the original
                    # paper, i.e., in Definition 11 4c, r(x) < r(sD) should be
                    # r(x) > r(sD).
                    while (reachability_plot[c_end - 1] > D_max
                           and c_end > U_start):
                        c_end -= 1

                # predecessor correction
                if predecessor_correction:
                    c_start, c_end = _correct_predecessor(reachability_plot,
                                                          predecessor_plot,
                                                          ordering,
                                                          c_start,
                                                          c_end)
                if c_start is None:
                    continue

                # Definition 11: criterion 3.a
                if c_end - c_start + 1 < min_cluster_size:
                    continue

                # Definition 11: criterion 1
                if c_start > D['end']:
                    continue

                # Definition 11: criterion 2
                if c_end < U_start:
                    continue

                U_clusters.append((c_start, c_end))

            # add smaller clusters first.
            U_clusters.reverse()
            clusters.extend(U_clusters)

    return np.array(clusters)
# ...
```
84 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
85 - /tmp/repos/scikit-learn/sklearn/__init__.py
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
```
86 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
87 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
88 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
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
# ...
```
89 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
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
# ...
```
90 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array
# ...
```
91 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
# ...# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
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



```
92 - /tmp/repos/scikit-learn/sklearn/metrics/cluster/bicluster.py
```python
from __future__ import division

import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils.validation import check_consistent_length, check_array

__all__ = ["consensus_score"]
```
93 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array
# ...
```
**94 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class _CFSubcluster(object):


    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)
```
**95 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py**:
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
```
96 - /tmp/repos/scikit-learn/examples/cluster/plot_linkage_comparison.py
```python
,
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
**97 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):


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
```
98 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
99 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array
# ...
```
100 - /tmp/repos/scikit-learn/sklearn/linear_model/stochastic_gradient.py
```python
# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com> (main author)
#          Mathieu Blondel (partial_fit support)
#
# License: BSD 3 clause
"""Classification and regression using Stochastic Gradient Descent (SGD)."""

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod

from ..externals.joblib import Parallel, delayed

from .base import LinearClassifierMixin, SparseCoefMixin
from .base import make_dataset
from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array, check_random_state, check_X_y
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.validation import check_is_fitted
from ..exceptions import ConvergenceWarning
from ..externals import six

from .sgd_fast import plain_sgd, average_sgd
from ..utils import compute_class_weight
from ..utils import deprecated
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import ModifiedHuber
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive
from .sgd_fast import SquaredEpsilonInsensitive


LEARNING_RATE_TYPES = {"constant": 1, "optimal": 2, "invscaling": 3,
                       "pa1": 4, "pa2": 5}

PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

DEFAULT_EPSILON = 0.1
```
101 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order='C')
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
                        _, _, coefs = lars_path_gram(
                            Xy=row, Gram=sub_covariance, n_samples=row.size,
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
            if not np.isfinite(precision_.sum()):
                raise FloatingPointError('The system is too ill-conditioned '
                                         'for this solver')
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
    
```
102 - /tmp/repos/scikit-learn/examples/applications/plot_outlier_detection_housing.py
```python
# ..."""
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
plt.xlabel("average number of rooms per dwelling"
```
103 - /tmp/repos/scikit-learn/sklearn/covariance/elliptic_envelope.py
```python
class EllipticEnvelope(MinCovDet, OutlierMixin):


    @property
    def threshold_(self):
        warnings.warn("threshold_ attribute is deprecated in 0.20 and will"
                      " be removed in 0.22.", DeprecationWarning)
        return self.offset_
```
104 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
105 - /tmp/repos/scikit-learn/sklearn/svm/classes.py
```python
import warnings
import numpy as np

from .base import _fit_liblinear, BaseSVC, BaseLibSVM
from ..base import BaseEstimator, RegressorMixin, OutlierMixin
from ..linear_model.base import LinearClassifierMixin, SparseCoefMixin, \
    LinearModel
from ..utils import check_X_y
from ..utils.validation import _num_samples
from ..utils.multiclass import check_classification_targets
```
106 - /tmp/repos/scikit-learn/examples/cluster/plot_cluster_comparison.py
```python
# ..."""
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
                " may not work as expected."
```
107 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
108 - /tmp/repos/scikit-learn/sklearn/preprocessing/_encoders.py
```python
self._categories = getattr(self, '_categories', None)
        self._categorical_features = getattr(self, '_categorical_features',
                                             None)

        # user manually set the categories or second fit -> never legacy mode
        if self.categories is not None or self._categories is not None:
            self._legacy_mode = False
            if self.categories is not None:
                self._categories = self.categories

        # categories not set -> infer if we need legacy mode or not
        elif self.n_values is not None and self.n_values != 'auto':
            msg = (
                "Passing 'n_values' is deprecated in version 0.20 and will be "
                "removed in 0.22. You can use the 'categories' keyword "
                "instead. 'n_values=n' corresponds to 'categories=[range(n)]'."
            )
            warnings.warn(msg, DeprecationWarning)
            self._legacy_mode = True

        else:  # n_values = 'auto'
            if self.handle_unknown == 'ignore':
                # no change in behaviour, no need to raise deprecation warning
                self._legacy_mode = False
                self._categories = 'auto'
                if self.n_values == 'auto':
                    # user manually specified this
                    msg = (
                        "Passing 'n_values' is deprecated in version 0.20 and "
                        "will be removed in 0.22. n_values='auto' can be "
                        "replaced with categories='auto'."
                    )
                    warnings.warn(msg, DeprecationWarning)
            else:

                # check if we have integer or categorical input
                try:
                    X = check_array(X, dtype=np.int)
                except ValueError:
                    self._legacy_mode = False
                    self._categories = 'auto'
                else:
                    msg = (
                        "The handling of integer data will change in version "
                        "0.22. Currently, the categories are determined "
                        "based on the range [0, max(values)], while in the "
                        "future they will be determined based on the unique "
                        "values.\nIf you want the future behaviour and "
                        "silence this warning, you can specify "
                        "\"categories='auto'\".\n"
                        "In case you used a LabelEncoder before this "
                        "OneHotEncoder to convert the categories to integers, "
                        "then you can now use the OneHotEncoder directly."
                    )
                    warnings.warn(msg, FutureWarning)
                    self._legacy_mode = True
                    self.n_values = 'auto'

        # if user specified categorical_features -> always use legacy mode
        
```
109 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
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
**110 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):


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
```
**111 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):


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
112 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _boston_subset(n_samples=50)
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)
```
113 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py
```python
# collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
# ...
```
114 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array
# ...
```
115 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
```
**116 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
```python
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
```
117 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
**118 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
# ...


class Birch(BaseEstimator, TransformerMixin, ClusterMixin):
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
    >>> brc = Birch(n_clusters=None)
    >>> brc.fit(X)
    Birch(n_clusters=None)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

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
    
```
119 - /tmp/repos/scikit-learn/sklearn/exceptions.py
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
```
120 - /tmp/repos/scikit-learn/sklearn/learning_curve.py
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
```
**121 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
# ...


class Birch(BaseEstimator, TransformerMixin, ClusterMixin):
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
    
```
122 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python

```
123 - /tmp/repos/scikit-learn/examples/cluster/plot_cluster_comparison.py
```python
,
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
124 - /tmp/repos/scikit-learn/doc/conftest.py
```python
def setup_unsupervised_learning():
    # ignore deprecation warnings from scipy.misc.face
    warnings.filterwarnings('ignore', 'The binary mode of fromstring',
                            DeprecationWarning)
```
125 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.

    Parameters
    ----------
    estimator : object
        The estimator
    random_state : int, RandomState instance or None, optional, default=0
        Pseudo random number generator state.  If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


try:
    import pytest

    skip_if_32bit = pytest.mark.skipif(_IS_32BIT,
                                       reason='skipped on 32bit platforms')
    skip_travis = pytest.mark.skipif(os.environ.get('TRAVIS') == 'true',
                                     reason='skip on travis')
    fails_if_pypy = pytest.mark.xfail(IS_PYPY, raises=NotImplementedError,
                                      reason='not compatible with PyPy')
    skip_if_no_parallel = pytest.mark.skipif(not joblib.parallel.mp,
                                             reason="joblib is in serial mode")

    #  Decorator for tests involving both BLAS calls and multiprocessing.
    #
    #  Under POSIX (e.g. Linux or OSX), using multiprocessing in conjunction
    #  with some implementation of BLAS (or other libraries that manage an
    #  internal posix thread pool) can cause a crash or a freeze of the Python
    #  process.
    #
    #  In practice all known packaged distributions (from Linux distros or
    #  Anaconda) of BLAS under Linux seems to be safe. So we this problem seems
    #  to only impact OSX users.
    #
    #  This wrapper makes it possible to skip tests that can possibly cause
    #  this crash under OS X with.
    #
    #  Under Python 3.4+ it is possible to use the `forkserver` start method
    #  for multiprocessing to avoid this issue. However it can cause pickling
    #  errors on interactively defined functions. It therefore not enabled by
    #  default.

    if_safe_multiprocessing_with_blas = pytest.mark.skipif(
            sys.platform == 'darwin',
            reason="Possible multi-process bug with some BLAS")
except ImportError:
    pass


def clean_warning_registry():
    """Clean Python warning registry for easier testing of warning messages.

    When changing warning filters this function is not necessary with
    Python3.5+, as __warningregistry__ will be re-set internally.
    See https://bugs.python.org/issue4180 and
    https://bugs.python.org/issue21724 for more details.

    """
    for mod in sys.modules.values():
        registry = getattr(mod, "__warningregistry__", None)
        if registry is not None:
            registry.clear()


def check_skip_network():
    if int(os.environ.get('SKLEARN_SKIP_NETWORK_TESTS', 0)):
        raise SkipTest("Text tutorial requires large dataset download")



```
126 - /tmp/repos/scikit-learn/sklearn/cluster/optics_.py
```python
if (len(node_1.points) < min_cluster_size and
            node_list.count((node_1, local_max_1)) > 0):
        # cluster 1 is too small
        node_list.remove((node_1, local_max_1))
    if (len(node_2.points) < min_cluster_size and
            node_list.count((node_2, local_max_2)) > 0):
        # cluster 2 is too small
        node_list.remove((node_2, local_max_2))
    if not node_list:
        # parent_node will be a leaf
        node.assign_split_point(-1)
        return

    # Check if nodes can be moved up one level - the new cluster created
    # is too "similar" to its parent, given the similarity threshold.
    bypass_node = 0
    if parent_node is not None:
        if ((node.end - node.start) / (parent_node.end - parent_node.start) >
                similarity_threshold):

            parent_node.children.remove(node)
            bypass_node = 1

    for nl in node_list:
        if bypass_node == 1:
            parent_node.add_child(nl[0])
            _cluster_tree(nl[0], parent_node, nl[1],
                          reachability_plot, reachability_ordering,
                          min_cluster_size, maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)
        else:
            node.add_child(nl[0])
            _cluster_tree(nl[0], node, nl[1], reachability_plot,
                          reachability_ordering, min_cluster_size,
                          maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)
# ...
```
127 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
128 - /tmp/repos/scikit-learn/sklearn/metrics/classification.py
```python
if labels is not None:
        warnings.warn("The labels parameter is unused. It was"
                      " deprecated in version 0.21 and"
                      " will be removed in version 0.23",
                      DeprecationWarning)

    if sample_weight is None:
        weight_average = 1.
    else:
        weight_average = np.mean(sample_weight)

    if y_type.startswith('multilabel'):
        n_differences = count_nonzero(y_true - y_pred,
                                      sample_weight=sample_weight)
        return (n_differences /
                (y_true.shape[0] * y_true.shape[1] * weight_average))

    elif y_type in ["binary", "multiclass"]:
        return _weighted_sum(y_true != y_pred, sample_weight, normalize=True)
    else:
        raise ValueError("{0} is not supported".format(y_type))
# ...
```
129 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def is_abstract(c):
        if not(hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    if include_other is not None:
        warnings.warn("include_other was deprecated in version 0.21,"
                      " has no effect and will be removed in 0.23",
                      DeprecationWarning)

    if include_dont_test is not None:
        warnings.warn("include_dont_test was deprecated in version 0.21,"
                      " has no effect and will be removed in 0.23",
                      DeprecationWarning)

    if include_meta_estimators is not None:
        warnings.warn("include_meta_estimators was deprecated in version 0.21,"
                      " has no effect and will be removed in 0.23",
                      DeprecationWarning)

    all_classes = []
    # get parent folder
    path = sklearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='sklearn.', onerror=lambda x: None):
        if ".tests." in modname or "externals" in modname:
            continue
        if IS_PYPY and ('_svmlight_format' in modname or
                        'feature_extraction._hashing' in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] != 'BaseEstimator')]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin,
                   'regressor': RegressorMixin,
                   'transformer': TransformerMixin,
                   'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))



```
130 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    if ("n_iter" in params and estimator.__class__.__name__ != "TSNE"
            and not isinstance(estimator, BaseSGD)):
        estimator.set_params(n_iter=5)
    if "max_iter" in params:
        warnings.simplefilter("ignore", ConvergenceWarning)
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
131 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
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
132 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
133 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
134 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
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
```
135 - /tmp/repos/scikit-learn/sklearn/cross_validation.py
```python
"""
The :mod:`sklearn.cross_validation` module includes utilities for cross-
validation and performance evaluation.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import warnings
from itertools import chain, combinations
from math import ceil, floor, factorial
import numbers
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse as sp

from .base import is_classifier, clone
from .utils import indexable, check_random_state, safe_indexing
from .utils.validation import (_is_arraylike, _num_samples,
                               column_or_1d)
from .utils.multiclass import type_of_target
from .externals.joblib import Parallel, delayed, logger
from .externals.six import with_metaclass
from .externals.six.moves import zip
from .metrics.scorer import check_scoring
from .gaussian_process.kernels import Kernel as GPKernel
from .exceptions import FitFailedWarning


warnings.warn("This module was deprecated in version 0.18 in favor of the "
              "model_selection module into which all the refactored classes "
              "and functions are moved. Also note that the interface of the "
              "new CV iterators are different from that of this module. "
              "This module will be removed in 0.20.", DeprecationWarning)


__all__ = ['KFold',
           'LabelKFold',
           'LeaveOneLabelOut',
           'LeaveOneOut',
           'LeavePLabelOut',
           'LeavePOut',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'PredefinedSplit',
           'LabelShuffleSplit',
           'check_cv',
           'cross_val_score',
           'cross_val_predict',
           'permutation_test_score',
           'train_test_split']
```
**136 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py**:
```python
if (n_samples == 1 or
            _equal_similarities_and_preferences(S, preference)):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn("All samples have mutually equal similarities. "
                      "Returning arbitrary cluster center(s).")
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return ((np.arange(n_samples), np.arange(n_samples), 0)
                    if return_n_iter
                    else (np.arange(n_samples), np.arange(n_samples)))
        else:
            return ((np.array([0]), np.array([0] * n_samples), 0)
                    if return_n_iter
                    else (np.array([0]), np.array([0] * n_samples)))

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
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

    
```
137 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
138 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array
# ...
```
139 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=DeprecationWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array([[3, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 1]])
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = [1, 1, 1, 2, 2, 2]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)
```
140 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python


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

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        
```
141 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python


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

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        
```
142 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
143 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
class _IgnoreWarnings:
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
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning
        self.log[:] = []


def assert_raise_message(exceptions, message, function, *args, **kwargs):
    """Helper function to test the message raised in an exception.

    Given an exception, a callable to raise the exception, and
    a message string, tests that the correct exception is raised and
    that the message is a substring of the error thrown. Used to test
    that the specific message thrown during an exception is correct.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        An Exception object.

    message : str
        The error message or a substring of the error message.

    function : callable
        Callable object to raise error.

    *args : the positional arguments to `function`.

    **kwargs : the keyword arguments to `function`.
    """
    try:
        function(*args, **kwargs)
    except exceptions as e:
        error_message = str(e)
        if message not in error_message:
            raise AssertionError("Error message does not include the expected"
                                 " string: %r. Observed error message: %r" %
                                 (message, error_message))
    else:
        # concatenate exception names
        if isinstance(exceptions, tuple):
            names = " or ".join(e.__name__ for e in exceptions)
        else:
            names = exceptions.__name__

        raise AssertionError("%s not raised by %s" %
                             (names, function.__name__))



```
144 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
145 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
class IsolationForest(BaseBagging, OutlierMixin):


    @property
    def threshold_(self):
        warnings.warn("threshold_ attribute is deprecated in 0.20 and will"
                      " be removed in 0.22.", DeprecationWarning)
        if self.contamination == 'auto':
            return self._threshold_
        return self.offset_
```
146 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python


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
        least_squares, _ = solve_cholesky(L[:n_active, :n_active],
                                          sign_active[:n_active],
                                          lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, _ = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        
```
147 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
148 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
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
# ...
```
149 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py
```python
def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions
```
150 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            weight_sums = np.zeros(self.n_clusters, dtype=sample_weight.dtype)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            # Compute the label assignment on the init dataset
            _mini_batch_step(
                X_valid, sample_weight_valid,
                x_squared_norms[validation_indices], cluster_centers,
                weight_sums, old_center_buffer, False, distances=None,
                verbose=self.verbose)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, sample_weight_valid,
                                         x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = weight_sums
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.randint(
                0, n_samples, self.batch_size)

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], sample_weight[minibatch_indices],
                x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + int(self.counts_.min())) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = \
                    self._labels_inertia_minibatch(X, sample_weight)

        return self

    
```
151 - /tmp/repos/scikit-learn/sklearn/cluster/spectral.py
```python
# do this max_svd_restarts times.
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
# ...
```
152 - /tmp/repos/scikit-learn/sklearn/__init__.py
```python
"""
import sys
import re
import warnings
import logging
import os

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
__version__ = '0.22.dev0'


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
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
    from .utils._show_versions import show_versions

    __check_build  # avoid flakes unused variable error

    __all__ = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
               'datasets', 'decomposition', 'dummy', 'ensemble', 'exceptions',
               'experimental', 'externals', 'feature_extraction',
               'feature_selection', 'gaussian_process', 'inspection',
               'isotonic', 'kernel_approximation', 'kernel_ridge',
               'linear_model', 'manifold', 'metrics', 'mixture',
               'model_selection', 'multiclass', 'multioutput',
               'naive_bayes', 'neighbors', 'neural_network', 'pipeline',
               'preprocessing', 'random_projection', 'semi_supervised',
               'svm', 'tree', 'discriminant_analysis', 'impute', 'compose',
               # Non-modules:
               'clone', 'get_config', 'set_config', 'config_context',
               'show_versions']


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs
```
153 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
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
154 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            weight_sums = np.zeros(self.n_clusters, dtype=sample_weight.dtype)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            # Compute the label assignment on the init dataset
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X_valid, sample_weight_valid,
                x_squared_norms[validation_indices], cluster_centers,
                weight_sums, old_center_buffer, False, distances=None,
                verbose=self.verbose)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, sample_weight_valid,
                                         x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = weight_sums
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.randint(
                0, n_samples, self.batch_size)

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], sample_weight[minibatch_indices],
                x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + int(self.counts_.min())) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = \
                    self._labels_inertia_minibatch(X, sample_weight)

        return self

    
```
155 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
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
```
156 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
# ..."""Testing utilities."""

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

```
157 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/grower.py
```python
class TreeGrower:


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
```
158 - /tmp/repos/scikit-learn/sklearn/svm/base.py
```python
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import warnings
from abc import ABCMeta, abstractmethod

from . import libsvm, liblinear
from . import libsvm_sparse
from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelEncoder
from ..utils.multiclass import _ovr_decision_function
from ..utils import check_array, check_consistent_length, check_random_state
from ..utils import column_or_1d, check_X_y
from ..utils import compute_class_weight
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..externals import six
from ..exceptions import ConvergenceWarning
from ..exceptions import NotFittedError


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
```
159 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
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
```
160 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
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
```
161 - /tmp/repos/scikit-learn/examples/cluster/plot_kmeans_stability_low_dim_dense.py
```python
# ..."""
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
    plt.title("Example cluster allocation with a single random init
```
**162 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
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
**163 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
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
164 - /tmp/repos/scikit-learn/examples/covariance/plot_outlier_detection.py
```python
# ..."""
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
    plt.suptitle("Outlier detection"
```
165 - /tmp/repos/scikit-learn/sklearn/linear_model/coordinate_descent.py
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
```
166 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
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
        
```
167 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
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
        
```
168 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class BaseHistGradientBoosting(BaseEstimator, ABC):


    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # A higher score is always better. Higher tol means that it will be
        # harder for subsequent iteration to be considered an improvement upon
        # the reference score, and therefore it is more likely to early stop
        # because of the lack of significant improvement.
        tol = 0 if self.tol is None else self.tol
        reference_score = scores[-reference_position] + tol
        recent_scores = scores[-reference_position + 1:]
        recent_improvements = [score > reference_score
                               for score in recent_scores]
        return not any(recent_improvements)
```
169 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
170 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py
```python
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
                                        **SOLVE_TRIANGULAR_ARGS)

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
```
**171 - /tmp/repos/scikit-learn/sklearn/decomposition/fastica_.py**:
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
```
172 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            counts = np.zeros(self.n_clusters, dtype=np.int32)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            # Compute the label assignment on the init dataset
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X_valid, x_squared_norms[validation_indices],
                cluster_centers, counts, old_center_buffer, False,
                distances=None, verbose=self.verbose)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = counts
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.randint(
                0, n_samples, self.batch_size)

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + self.counts_.min()) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = self._labels_inertia_minibatch(X)

        return self

    
```
173 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
for init_idx in range(n_init):
            if self.verbose:
                print("Init %d/%d with method: %s"
                      % (init_idx + 1, n_init, self.init))
            counts = np.zeros(self.n_clusters, dtype=np.int32)

            # TODO: once the `k_means` function works with sparse input we
            # should refactor the following init to use it instead.

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            cluster_centers = _init_centroids(
                X, self.n_clusters, self.init,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                init_size=init_size)

            # Compute the label assignment on the init dataset
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X_valid, x_squared_norms[validation_indices],
                cluster_centers, counts, old_center_buffer, False,
                distances=None, verbose=self.verbose)

            # Keep only the best cluster centers across independent inits on
            # the common validation set
            _, inertia = _labels_inertia(X_valid, x_squared_norms_valid,
                                         cluster_centers)
            if self.verbose:
                print("Inertia for init %d/%d: %f"
                      % (init_idx + 1, n_init, inertia))
            if best_inertia is None or inertia < best_inertia:
                self.cluster_centers_ = cluster_centers
                self.counts_ = counts
                best_inertia = inertia

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        # Perform the iterative optimization until the final convergence
        # criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            minibatch_indices = random_state.randint(
                0, n_samples, self.batch_size)

            # Perform the actual update step on the minibatch data
            batch_inertia, centers_squared_diff = _mini_batch_step(
                X[minibatch_indices], x_squared_norms[minibatch_indices],
                self.cluster_centers_, self.counts_,
                old_center_buffer, tol > 0.0, distances=distances,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + self.counts_.min()) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose)

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self, iteration_idx, n_iter, tol, n_samples,
                    centers_squared_diff, batch_inertia, convergence_context,
                    verbose=self.verbose):
                break

        self.n_iter_ = iteration_idx + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = self._labels_inertia_minibatch(X)

        return self

    
```
174 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
```python
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
175 - /tmp/repos/scikit-learn/sklearn/feature_selection/variance_threshold.py
```python
class VarianceThreshold(BaseEstimator, SelectorMixin):


    def _get_support_mask(self):
        check_is_fitted(self, 'variances_')

        return self.variances_ > self.threshold
```
176 - /tmp/repos/scikit-learn/sklearn/neighbors/lof.py
```python
# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.stats import scoreatpercentile

from .base import NeighborsBase
from .base import KNeighborsMixin
from .base import UnsupervisedMixin
from ..base import OutlierMixin

from ..utils.validation import check_is_fitted
from ..utils import check_array

__all__ = ["LocalOutlierFactor"]
```
177 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
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
178 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
        with assert_raises(ValueError, msg="The classifer {} does not"
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
        
```
179 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
        with assert_raises(ValueError, msg="The classifer {} does not"
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
        
```
180 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
        
```
**181 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
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
        check_is_fitted(self, ['subcluster_centers_', 'partial_fit_'],
                        all_or_any=any)

        if (hasattr(self, 'subcluster_centers_') and
                X.shape[1] != self.subcluster_centers_.shape[1]):
            raise ValueError(
                "Training data and predicted data do "
                "not have same number of features.")

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

    
```
182 - /tmp/repos/scikit-learn/examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py
```python
# ..."""
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
plt.title("Feature Agglomeration"
```
183 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
**184 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
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

    
```
185 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
# ...
```
186 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
# ...
```
187 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
# ...
```
188 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
# ...
```
189 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
```python
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
**190 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
\
             optional (default='lbfgs').

        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
          'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
          handle multinomial loss; 'liblinear' is limited to one-versus-rest
          schemes.
        - 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
        - 'liblinear' and 'saga' also handle L1 penalty
        - 'saga' also supports 'elasticnet' penalty
        - 'liblinear' does not handle no penalty

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.

    max_iter : int, optional (default=100)
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int or None, optional (default=None)
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.

    l1_ratio : float or None, optional (default=None)
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'`. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Attributes
    ----------

    classes_ : array, shape (n_classes, )
        A list of class labels known to the classifier.

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

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...

    See also
    --------
    SGDClassifier : incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
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
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

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
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    
```
191 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array
# ...
```
192 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    boston = load_boston()
    X, y = boston.data, boston.target
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    assert_raises_regex(ValueError, msg, e.fit, X, y)
```
193 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
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

    def _do_mstep(self, X, z, params):
        """Maximize the variational lower bound

        Update each of the parameters to maximize the lower bound."""
        self._monitor(X, z, "z")
        self._update_concentration(z)
        self._monitor(X, z, "gamma")
        if 'm' in params:
            self._update_means(X, z)
        self._monitor(X, z, "mu")
        if 'c' in params:
            self._update_precisions(X, z)
        self._monitor(X, z, "a and b", end=True)

    def _initialize_gamma(self):
        "Initializes the concentration parameters"
        self.gamma_ = self.alpha * np.ones((self.n_components, 3))

    def _bound_concentration(self):
        """The variational lower bound for the concentration parameter."""
        logprior = gammaln(self.alpha) * self.n_components
        logprior += np.sum((self.alpha - 1) * (
            digamma(self.gamma_.T[2]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        logprior += np.sum(- gammaln(self.gamma_.T[1] + self.gamma_.T[2]))
        logprior += np.sum(gammaln(self.gamma_.T[1]) +
                           gammaln(self.gamma_.T[2]))
        logprior -= np.sum((self.gamma_.T[1] - 1) * (
            digamma(self.gamma_.T[1]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        logprior -= np.sum((self.gamma_.T[2] - 1) * (
            digamma(self.gamma_.T[2]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        return logprior

    
```
194 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
# ...
```
195 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
# ...
```
196 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
# ...
```
197 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
# ...
```
198 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
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
```
199 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
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
# ...
```
200 - /tmp/repos/scikit-learn/sklearn/decomposition/sparse_pca.py
```python
def _check_normalize_components(normalize_components, estimator_name):
    if normalize_components != 'deprecated':
        if normalize_components:
            warnings.warn(
                "'normalize_components' has been deprecated in 0.22 and "
                "will be removed in 0.24. Remove the parameter from the "
                " constructor.", DeprecationWarning
            )
        else:
            raise NotImplementedError(
                "normalize_components=False is not supported starting from "
                "0.22. Remove this parameter from the constructor."
            )
```
201 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
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
# ...
```
202 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
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
# ...
```
203 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
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
```
**204 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py**:
```python
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
# ...
```
205 - /tmp/repos/scikit-learn/sklearn/externals/joblib/__init__.py
```python
# Import necessary to preserve backward compatibility of pickles
import sys
import warnings

from joblib import *


msg = ("sklearn.externals.joblib is deprecated in 0.21 and will be removed "
       "in 0.23. Please import this functionality directly from joblib, "
       "which can be installed with: pip install joblib. If this warning is "
       "raised when loading pickled models, you may need to re-serialize "
       "those models with scikit-learn 0.21+.")

if not hasattr(sys, "_is_pytest_session"):
    warnings.warn(msg, category=DeprecationWarning)
```
206 - /tmp/repos/scikit-learn/sklearn/neighbors/nca.py
```python
# coding: utf-8
"""
Neighborhood Component Analysis
"""

# Authors: William de Vazelhes <wdevazelhes@gmail.com>
#          John Chiotellis <ioannis.chiotellis@in.tum.de>
# License: BSD 3 clause

from __future__ import print_function

from warnings import warn
import numpy as np
import sys
import time
import numbers
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
207 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
208 - /tmp/repos/scikit-learn/examples/cluster/plot_birch_vs_minibatchkmeans.py
```python
# ..."""
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
209 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
def _yield_checks(name, estimator):
    tags = _safe_tags(estimator)
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_sample_weights_pandas_series
    yield check_sample_weights_list
    yield check_sample_weights_invariance
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    if not tags["no_validation"]:
        yield check_complex_data
        yield check_dtype_object
        yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION:
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if not tags["allow_nan"] and not tags["no_validation"]:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    yield check_estimators_overwrite_params
    if hasattr(estimator, 'sparsify'):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle
```
210 - /tmp/repos/scikit-learn/sklearn/discriminant_analysis.py
```python
class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    @property
    @deprecated("Attribute covariances_ was deprecated in version"
                " 0.19 and will be removed in 0.21. Use "
                "covariance_ instead")
    def covariances_(self):
        return self.covariance_
```
211 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
212 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        X = pairwise_estimator_convert_X(X, classifier)
        y = enforce_estimator_tags_y(classifier, y)

        set_random_state(classifier)
        # raises error on malformed input for fit
        if not tags["no_validation"]:
            with assert_raises(
                ValueError,
                msg="The classifier {} does not "
                    "raise an error when incorrect/malformed input "
                    "data for fit is passed. The number of training "
                    "examples is not the same as the number of labels. "
                    "Perhaps use check_X_y in fit.".format(name)):
                classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_samples,)
        # training set performance
        if not tags['poor_score']:
            assert accuracy_score(y, y_pred) > 0.83

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)")
        msg = ("The classifier {} does not raise an error when the number of "
               "features in {} is different from the number of features in "
               "fit.")

        if not tags["no_validation"]:
            if _is_pairwise(classifier):
                with assert_raises(ValueError,
                                   msg=msg_pairwise.format(name, "predict")):
                    classifier.predict(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError,
                                   msg=msg.format(name, "predict")):
                    classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if not tags["multioutput_only"]:
                        assert decision.shape == (n_samples,)
                    else:
                        assert decision.shape == (n_samples, 1)
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert decision.shape == (n_samples, n_classes)
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not tags["no_validation"]:
                    if _is_pairwise(classifier):
                        with assert_raises(ValueError, msg=msg_pairwise.format(
                                name, "decision_function")):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        with assert_raises(ValueError, msg=msg.format(
                                name, "decision_function")):
                            classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        
```
**213 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class _CFNode(object):


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
```
214 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
```
215 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
class _PGNMF(NMF):


    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        X = check_array(X, accept_sparse=('csr', 'csc'))
        check_non_negative(X, "NMF (input X)")

        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features

        if (not isinstance(n_components, INTEGER_TYPES) or
                n_components <= 0):
            raise ValueError("Number of components must be a positive integer;"
                             " got (n_components=%r)" % n_components)
        if not isinstance(self.max_iter, INTEGER_TYPES) or self.max_iter < 0:
            raise ValueError("Maximum number of iterations must be a positive "
                             "integer; got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        # check W and H, or initialize them
        if self.init == 'custom' and update_H:
            _check_init(H, (n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, n_components), "NMF (input W)")
        elif not update_H:
            _check_init(H, (n_components, n_features), "NMF (input H)")
            W = np.zeros((n_samples, n_components))
        else:
            W, H = _initialize_nmf(X, n_components, init=self.init,
                                   random_state=self.random_state)

        if update_H:  # fit_transform
            W, H, n_iter = _fit_projected_gradient(
                X, W, H, self.tol, self.max_iter, self.nls_max_iter,
                self.alpha, self.l1_ratio)
        else:  # transform
            Wt, _, n_iter = _nls_subproblem(X.T, H.T, W.T, self.tol,
                                            self.nls_max_iter,
                                            alpha=self.alpha,
                                            l1_ratio=self.l1_ratio)
            W = Wt.T

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn("Maximum number of iteration %d reached. Increase it"
                          " to improve convergence." % self.max_iter,
                          ConvergenceWarning)

        return W, H, n_iter
```
216 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
                assert_equal(param_value, init_param.default, init_param.name)
```
217 - /tmp/repos/scikit-learn/examples/decomposition/plot_ica_vs_pca.py
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
**218 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py**:
```python
class _CFNode(object):


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
219 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
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

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    if init == "warn":
        if n_components < n_features:
            warnings.warn("The default value of init will change from "
                          "random to None in 0.23 to make it consistent "
                          "with decomposition.NMF.", FutureWarning)
        init = "random"

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((n_samples, n_components), avg)
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
# ...
```
220 - /tmp/repos/scikit-learn/sklearn/cluster/hierarchical.py
```python
\
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

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                connectivity=None, linkage='ward', memory=None, n_clusters=2,
                pooling_func='deprecated')
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

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
221 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
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
```
**222 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            w0, loss, info = optimize.fmin_l_bfgs_b(
                func, w0, fprime=None,
                args=(X, target, 1. / C, sample_weight),
                iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            if info["warnflag"] == 1:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.", ConvergenceWarning)
            # In scipy <= 1.0.0, nit may exceed maxiter.
            # See https://github.com/scipy/scipy/issues/7854.
            n_iter_i = min(info['nit'], max_iter)
        elif solver == 'newton-cg':
            args = (X, target, 1. / C, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(np.float64)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            else:
                alpha = 1. / C
                beta = 0.
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            multi_w0 = np.reshape(w0, (classes.size, -1))
            if classes.size == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0)
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter
# ...
```
**223 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            w0, loss, info = optimize.fmin_l_bfgs_b(
                func, w0, fprime=None,
                args=(X, target, 1. / C, sample_weight),
                iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            if info["warnflag"] == 1 and verbose > 0:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.", ConvergenceWarning)
            n_iter_i = info['nit'] - 1
        elif solver == 'newton-cg':
            args = (X, target, 1. / C, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(np.float64)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            else:
                alpha = 1. / C
                beta = 0.
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            multi_w0 = np.reshape(w0, (classes.size, -1))
            if classes.size == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0)
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter
# ...
```
224 - /tmp/repos/scikit-learn/sklearn/decomposition/online_lda.py
```python
class LatentDirichletAllocation(BaseEstimator, TransformerMixin):


    def _check_params(self):
        """Check model parameters."""
        if self.n_topics is not None:
            self._n_components = self.n_topics
            warnings.warn("n_topics has been renamed to n_components in "
                          "version 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        else:
            self._n_components = self.n_components

        if self._n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self._n_components)

        if self.total_samples <= 0:
            raise ValueError("Invalid 'total_samples' parameter: %r"
                             % self.total_samples)

        if self.learning_offset < 0:
            raise ValueError("Invalid 'learning_offset' parameter: %r"
                             % self.learning_offset)

        if self.learning_method not in ("batch", "online", None):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)
```
225 - /tmp/repos/scikit-learn/sklearn/cluster/hierarchical.py
```python
class AgglomerativeClustering(BaseEstimator, ClusterMixin):


    @deprecated("The ``n_components_`` attribute was deprecated "
                "in favor of ``n_connected_components_`` in 0.21 "
                "and will be removed in 0.23.")
    @property
    def n_components_(self):
        return self.n_connected_components_
```
226 - /tmp/repos/scikit-learn/sklearn/cluster/k_means_.py
```python
if effective_n_jobs(n_jobs) == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, n_clusters, max_iter=max_iter, init=init,
                verbose=verbose, precompute_distances=precompute_distances,
                tol=tol, x_squared_norms=x_squared_norms,
                random_state=random_state)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_single)(X, sample_weight, n_clusters,
                                   max_iter=max_iter, init=init,
                                   verbose=verbose, tol=tol,
                                   precompute_distances=precompute_distances,
                                   x_squared_norms=x_squared_norms,
                                   # Change seed to ensure variety
                                   random_state=seed)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    distinct_clusters = len(set(best_labels))
    if distinct_clusters < n_clusters:
        warnings.warn("Number of distinct clusters ({}) found smaller than "
                      "n_clusters ({}). Possibly due to duplicate points "
                      "in X.".format(distinct_clusters, n_clusters),
                      ConvergenceWarning, stacklevel=2)

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia
# ...
```
227 - /tmp/repos/scikit-learn/sklearn/tree/_reingold_tilford.py
```python
def move_subtree(wl, wr, shift):
    subtrees = wr.number - wl.number
    # print(wl.tree, "is conflicted with", wr.tree, 'moving', subtrees,
    # 'shift', shift)
    # print wl, wr, wr.number, wl.number, shift, subtrees, shift/subtrees
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift
```
228 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res
# ...
```
229 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res
# ...
```
230 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res
# ...
```
231 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
# ...


def _fit_multiplicative_update(X, W, H, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                               update_H=True, verbose=0):
    
```
232 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
# ...


def _fit_multiplicative_update(X, W, H, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                               update_H=True, verbose=0):
    
```
233 - /tmp/repos/scikit-learn/sklearn/decomposition/nmf.py
```python
# ...


def _fit_multiplicative_update(X, W, H, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                               update_H=True, verbose=0):
    
```
234 - /tmp/repos/scikit-learn/examples/cluster/plot_adjusted_for_chance_measures.py
```python

```
235 - /tmp/repos/scikit-learn/examples/cluster/plot_adjusted_for_chance_measures.py
```python

```
236 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1),
                                      np.ones(n_samples))
            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                if _is_pairwise(classifier_orig):
                    with assert_raises(ValueError, msg=msg_pairwise.format(
                            name, "predict_proba")):
                        classifier.predict_proba(X.reshape(-1, 1))
                else:
                    with assert_raises(ValueError, msg=msg.format(
                            name, "predict_proba")):
                        classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))
# ...
```
237 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
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
238 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
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
# ...
```
239 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
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
# ...
```
240 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
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
# ...
```
241 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
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
# ...
```
242 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
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
# ...
```
243 - /tmp/repos/scikit-learn/examples/cluster/plot_inductive_clustering.py
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
244 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
for alpha in alphas:
        try:
            # Capture the errors, and move on
            covariance_, precision_ = graphical_lasso(
                emp_cov, alpha=alpha, cov_init=covariance_, mode=mode, tol=tol,
                enet_tol=enet_tol, max_iter=max_iter, verbose=inner_verbose)
            covariances_.append(covariance_)
            precisions_.append(precision_)
            if X_test is not None:
                this_score = log_likelihood(test_emp_cov, precision_)
        except FloatingPointError:
            this_score = -np.inf
            covariances_.append(np.nan)
            precisions_.append(np.nan)
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose == 1:
            sys.stderr.write('.')
        elif verbose > 1:
            if X_test is not None:
                print('[graphical_lasso_path] alpha: %.2e, score: %.2e'
                      % (alpha, this_score))
            else:
                print('[graphical_lasso_path] alpha: %.2e' % alpha)
    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_
# ...
```
245 - /tmp/repos/scikit-learn/sklearn/covariance/graph_lasso_.py
```python
for alpha in alphas:
        try:
            # Capture the errors, and move on
            covariance_, precision_ = graphical_lasso(
                emp_cov, alpha=alpha, cov_init=covariance_, mode=mode, tol=tol,
                enet_tol=enet_tol, max_iter=max_iter, verbose=inner_verbose)
            covariances_.append(covariance_)
            precisions_.append(precision_)
            if X_test is not None:
                this_score = log_likelihood(test_emp_cov, precision_)
        except FloatingPointError:
            this_score = -np.inf
            covariances_.append(np.nan)
            precisions_.append(np.nan)
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose == 1:
            sys.stderr.write('.')
        elif verbose > 1:
            if X_test is not None:
                print('[graphical_lasso_path] alpha: %.2e, score: %.2e'
                      % (alpha, this_score))
            else:
                print('[graphical_lasso_path] alpha: %.2e' % alpha)
    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_
# ...
```
**246 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py**:
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
        
```
**247 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py**:
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
        
```
248 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py
```python
fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
# ...
```
249 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class BaseHistGradientBoosting(BaseEstimator, ABC):


    def _check_early_stopping_scorer(self, X_binned_small_train, y_small_train,
                                     X_binned_val, y_val):
        """Check if fitting should be early-stopped based on scorer.

        Scores are computed on validation data or on training data.
        """
        self.train_score_.append(
            self.scorer_(self, X_binned_small_train, y_small_train)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                self.scorer_(self, X_binned_val, y_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)
```
