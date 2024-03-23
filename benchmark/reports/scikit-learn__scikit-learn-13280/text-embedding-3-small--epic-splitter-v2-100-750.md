0 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python
class OneVsRestClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):


    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise ValueError(("Base estimator {0}, doesn't have "
                                 "partial_fit method").format(self.estimator))
            self.estimators_ = [clone(self.estimator) for _ in range
                                (self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(("Mini-batch contains {0} while classes " +
                             "must be subset of {1}").format(np.unique(y),
                                                             self.classes_))

        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, X, column)
            for estimator, column in izip(self.estimators_, columns))

        return self
```
1 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python
class OneVsOneClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):


    @if_delegate_has_method(delegate='estimator')
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data. Chunks
        of data can be passed in several iteration, where the first call
        should have an array of all target variables.


        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
            Multi-class targets.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            self.estimators_ = [clone(self.estimator) for i in
                                range(self.n_classes_ *
                                      (self.n_classes_ - 1) // 2)]

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError("Mini-batch contains {0} while it "
                             "must be subset of {1}".format(np.unique(y),
                                                            self.classes_))

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)
        combinations = itertools.combinations(range(self.n_classes_), 2)
        self.estimators_ = Parallel(
            n_jobs=self.n_jobs)(
                delayed(_partial_fit_ovo_binary)(
                    estimator, X, y, self.classes_[i], self.classes_[j])
                for estimator, (i, j) in izip(self.estimators_,
                                              (combinations)))

        self.pairwise_indices_ = None

        return self
```
**2 - /tmp/repos/scikit-learn/sklearn/naive_bayes.py**:
```python
class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    intercept_ : array, shape (n_classes, )
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    coef_ : array, shape (n_classes, n_features)
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]

    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
```
