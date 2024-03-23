0 - /tmp/repos/scikit-learn/sklearn/semi_supervised/label_propagation.py
```python
class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix.

    gamma : float
        Parameter for rbf kernel

    n_neighbors : integer > 0
        Parameter for knn kernel

    alpha : float
        Clamping factor.

        .. deprecated:: 0.19
            This parameter will be removed in 0.21.
            'alpha' is fixed to zero in 'LabelPropagation'.

    max_iter : integer
        Change maximum number of iterations allowed

    tol : float
        Convergence tolerance: threshold to consider the system at steady
        state

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        Input array.

    classes_ : array, shape = [n_classes]
        The distinct labels used in classifying instances.

    label_distributions_ : array, shape = [n_samples, n_classes]
        Categorical distribution for each item.

    transduction_ : array, shape = [n_samples]
        Label assigned to each item via the transduction.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LabelPropagation(...)

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    See Also
    --------
    LabelSpreading : Alternate label propagation strategy more robust to noise
    """

    _variant = 'propagation'

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7,
                 alpha=None, max_iter=1000, tol=1e-3, n_jobs=1):
        super(LabelPropagation, self).__init__(
            kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha,
            max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        if self.kernel == 'knn':
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        if sparse.isspmatrix(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            affinity_matrix /= normalizer[:, np.newaxis]
        return affinity_matrix

    def fit(self, X, y):
        if self.alpha is not None:
            warnings.warn(
                "alpha is deprecated since 0.19 and will be removed in 0.21.",
                DeprecationWarning
            )
            self.alpha = None
        return super(LabelPropagation, self).fit(X, y)



```
**1 - /tmp/repos/scikit-learn/sklearn/semi_supervised/_label_propagation.py**:
```python
class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix.

    gamma : float
        Parameter for rbf kernel

    n_neighbors : integer > 0
        Parameter for knn kernel

    max_iter : integer
        Change maximum number of iterations allowed

    tol : float
        Convergence tolerance: threshold to consider the system at steady
        state

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        Input array.

    classes_ : array, shape = [n_classes]
        The distinct labels used in classifying instances.

    label_distributions_ : array, shape = [n_samples, n_classes]
        Categorical distribution for each item.

    transduction_ : array, shape = [n_samples]
        Label assigned to each item via the transduction.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelPropagation(...)

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    See Also
    --------
    LabelSpreading : Alternate label propagation strategy more robust to noise
    """

    _variant = 'propagation'

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7,
                 max_iter=1000, tol=1e-3, n_jobs=None):
        super().__init__(kernel=kernel, gamma=gamma,
                         n_neighbors=n_neighbors, max_iter=max_iter,
                         tol=tol, n_jobs=n_jobs, alpha=None)

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        if self.kernel == 'knn':
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        if sparse.isspmatrix(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            affinity_matrix /= normalizer[:, np.newaxis]
        return affinity_matrix

    def fit(self, X, y):
        return super().fit(X, y)



```
