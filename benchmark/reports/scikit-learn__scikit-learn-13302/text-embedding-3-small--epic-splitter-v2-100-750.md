0 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
```python
# ...


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
1 - /tmp/repos/scikit-learn/sklearn/linear_model/sag.py
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
**2 - /tmp/repos/scikit-learn/sklearn/linear_model/ridge.py**:
```python
class _BaseRidge(six.with_metaclass(ABCMeta, LinearModel)):


    def fit(self, X, y, sample_weight=None):

        if self.solver in ('sag', 'saga'):
            _dtype = np.float64
        else:
            # all other solvers work at both float precision levels
            _dtype = [np.float64, np.float32]

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=_dtype,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        # temporary fix for fitting the intercept with sparse data using 'sag'
        if sparse.issparse(X) and self.fit_intercept:
            self.coef_, self.n_iter_, self.intercept_ = ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=True)
            self.intercept_ += y_offset
        else:
            self.coef_, self.n_iter_ = ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=False)
            self._set_intercept(X_offset, y_offset, X_scale)

        return self
```
