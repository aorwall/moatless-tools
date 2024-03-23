**0 - /tmp/repos/scikit-learn/sklearn/linear_model/coordinate_descent.py**:
```python
"""
    path = staticmethod(enet_path)

    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, positive=False, random_state=None,
                 selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection


###############################################################################
# Multi Task ElasticNet and Lasso models (with joint feature selection)


class MultiTaskElasticNet(Lasso):
    r"""Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1/L2 term. Defaults to 1.0

    l1_ratio : float
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    intercept_ : array, shape (n_tasks,)
        Independent term in decision function.

    coef_ : array, shape (n_tasks, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is \
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
    ... #doctest: +NORMALIZE_WHITESPACE
    MultiTaskElasticNet(alpha=0.1, copy_X=True, fit_intercept=True,
            l1_ratio=0.5, max_iter=1000, normalize=False, random_state=None,
            selection='cyclic', tol=0.0001, warm_start=False)
    >>> print(clf.coef_)
    [[ 0.45663524  0.45612256]
     [ 0.45663524  0.45612256]]
    >>> print(clf.intercept_)
    [ 0.0872422  0.0872422]

    See also
    --------
    MultiTaskElasticNet : Multi-task L1/L2 ElasticNet with built-in
        cross-validation.
    ElasticNet
    MultiTaskLasso

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    
```
