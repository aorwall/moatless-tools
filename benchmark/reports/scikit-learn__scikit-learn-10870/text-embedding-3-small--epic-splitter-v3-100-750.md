0 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to decrease at
        each iteration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        n_features, = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = (_compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        if self.covariance_type == 'tied':
            log_wishart = self.n_components * np.float64(_log_wishart_norm(
                self.degrees_of_freedom_, log_det_precisions_chol, n_features))
        else:
            log_wishart = np.sum(_log_wishart_norm(
                self.degrees_of_freedom_, log_det_precisions_chol, n_features))

        if self.weight_concentration_prior_type == 'dirichlet_process':
            log_norm_weight = -np.sum(betaln(self.weight_concentration_[0],
                                             self.weight_concentration_[1]))
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (-np.sum(np.exp(log_resp) * log_resp) -
                log_wishart - log_norm_weight -
                0.5 * n_features * np.sum(np.log(self.mean_precision_)))
```
**1 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
```
**2 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)
```
3 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

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
```
4 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
@deprecated("The `VBGMM` class is not working correctly and it's better "
            "to use `sklearn.mixture.BayesianGaussianMixture` class with "
            "parameter `weight_concentration_prior_type="
            "'dirichlet_distribution'` instead. "
            "VBGMM is deprecated in 0.18 and will be removed in 0.20.")
class VBGMM(_DPGMMBase):

    def _monitor(self, X, z, n, end=False):
        """Monitor the lower bound during iteration

        Debug method to help see exactly when it is failing to converge as
        expected.

        Note: this is very expensive and should not be used by default."""
        if self.verbose > 0:
            print("Bound after updating %8s: %f" % (n, self.lower_bound(X, z)))
            if end:
                print("Cluster proportions:", self.gamma_)
                print("covariance_type:", self.covariance_type)

    def _set_weights(self):
        self.weights_[:] = self.gamma_
        self.weights_ /= np.sum(self.weights_)
```
5 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

    def _checkcovariance_prior_parameter(self, X):
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                'full': np.atleast_2d(np.cov(X.T)),
                'tied': np.atleast_2d(np.cov(X.T)),
                'diag': np.var(X, axis=0, ddof=1),
                'spherical': np.var(X, axis=0, ddof=1).mean()
            }[self.covariance_type]

        elif self.covariance_type in ['full', 'tied']:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features, n_features),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_matrix(self.covariance_prior_,
                                    self.covariance_type)
        elif self.covariance_type == 'diag':
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features,),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_positivity(self.covariance_prior_,
                                        self.covariance_type)
        # spherical case
        elif self.covariance_prior > 0.:
            self.covariance_prior_ = self.covariance_prior
        else:
            raise ValueError("The parameter 'spherical covariance_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.covariance_prior)
```
6 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
class _DPGMMBase(_GMMBase):

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
**7 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)
```
8 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
class _GMMBase(BaseEstimator):

    def _fit(self, X, y=None, do_prediction=False):
        # ... other code

        for init in range(self.n_init):
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state).fit(X).cluster_centers_
                if self.verbose > 1:
                    print('\tMeans have been initialized.')

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)
                if self.verbose > 1:
                    print('\tWeights have been initialized.')

            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = \
                    distribute_covar_matrix_to_match_covariance_type(
                        cv, self.covariance_type, self.n_components)
                if self.verbose > 1:
                    print('\tCovariance matrices have been initialized.')

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            for i in range(self.n_iter):
                if self.verbose > 0:
                    print('\tEM iteration ' + str(i + 1))
                    start_iter_time = time()
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self.score_samples(X)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if self.verbose > 1:
                        print('\t\tChange: ' + str(change))
                    if change < self.tol:
                        self.converged_ = True
                        if self.verbose > 0:
                            print('\t\tEM algorithm converged.')
                        break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)
                if self.verbose > 1:
                    print('\t\tEM iteration ' + str(i + 1) + ' took {0:.5f}s'.format(
                        time() - start_iter_time))

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
                    if self.verbose > 1:
                        print('\tBetter parameters were found.')

            if self.verbose > 1:
                print('\tInitialization ' + str(init + 1) + ' took {0:.5f}s'.format(
                    time() - start_init_time))

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:  # self.n_iter == 0 occurs when using GMM within HMM
            # Need to make sure that there are responsibilities to output
            # Output zeros because it was just a quick initialization
            responsibilities = np.zeros((X.shape[0], self.n_components))

        return responsibilities
```
**9 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init
```
**10 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
```
11 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of::

            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).

    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float | None, optional.
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed. Smaller
        values concentrate the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it's set to 1.

    mean_prior : array-like, shape (n_features,), optional
        The prior on the mean distribution (Gaussian).
        If it is None, it's set to the mean of X.

    degrees_of_freedom_prior : float | None, optional.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, optional
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like, shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed.
        Smaller values concentrate the means of each clusters around
        `mean_prior`.

    mean_precision_ : array-like, shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    means_prior_ : array-like, shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like, shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <http://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
    """
```
12 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

    def _check_is_fitted(self):
        check_is_fitted(self, ['weight_concentration_', 'mean_precision_',
                               'means_', 'degrees_of_freedom_',
                               'covariances_', 'precisions_',
                               'precisions_cholesky_'])
```
13 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

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
```
14 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

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
```
15 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

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
**16 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2
```
**17 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
###############################################################################
```
**18 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
###############################################################################
```
19 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_sin.py
```python
# Parameters
n_samples = 100

# Generate random sample following a sine curve
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(X.shape[0]):
    x = i * step - 6.
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))

plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.04, top=0.95, hspace=.2, wspace=.05,
                    left=.03, right=.97)

# Fit a Gaussian mixture with EM using ten components
gmm = mixture.GaussianMixture(n_components=10, covariance_type='full',
                              max_iter=100).fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Expectation-maximization')

dpgmm = mixture.BayesianGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e-2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="random", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             "Bayesian Gaussian mixture models with a Dirichlet process prior "
             r"for $\gamma_0=0.01$.")

X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(X_s, y_s, dpgmm.n_components, 0,
             "Gaussian mixture with a Dirichlet process prior "
             r"for $\gamma_0=0.01$ sampled with $2000$ samples.")

dpgmm = mixture.BayesianGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e+2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="kmeans", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 2,
             "Bayesian Gaussian mixture models with a Dirichlet process prior "
             r"for $\gamma_0=100$")

X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(X_s, y_s, dpgmm.n_components, 1,
             "Gaussian mixture with a Dirichlet process prior "
             r"for $\gamma_0=100$ sampled with $2000$ samples.")

plt.show()
```
20 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

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
```
21 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)
```
22 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
class _GMMBase(BaseEstimator):
    """Gaussian Mixture Model.

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.

    Read more in the :ref:`User Guide <gmm>`.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. The best results is kept.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars. Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars. Defaults to 'wmc'.

    verbose : int, default: 0
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the change and time needed for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------

    DPGMM : Infinite gaussian mixture model, using the Dirichlet
        process, fit with a variational algorithm


    VBGMM : Finite gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn import mixture
    >>> np.random.seed(1)
    >>> g = mixture.GMM(n_components=2)
    >>> # Generate random observations with two modes centered on 0
    >>> # and 10 to use for training.
    >>> obs = np.concatenate((np.random.randn(100, 1),
    ...                       10 + np.random.randn(300, 1)))
    >>> g.fit(obs)  # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, tol=0.001, verbose=0)
    >>> np.round(g.weights_, 2)
    array([ 0.75,  0.25])
    >>> np.round(g.means_, 2)
    array([[ 10.05],
           [  0.06]])
    >>> np.round(g.covars_, 2) # doctest: +SKIP
    array([[[ 1.02]],
           [[ 0.96]]])
    >>> g.predict([[0], [2], [9], [10]]) # doctest: +ELLIPSIS
    array([1, 1, 0, 0]...)
    >>> np.round(g.score([[0], [2], [9], [10]]), 2)
    array([-2.19, -4.58, -1.75, -1.21])
    >>> # Refit the model on new data (initial parameters remain the
    >>> # same), this time with an even split between the two modes.
    >>> g.fit(20 * [[0]] + 20 * [[10]])  # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, tol=0.001, verbose=0)
    >>> np.round(g.weights_, 2)
    array([ 0.5,  0.5])

    """
```
23 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):
```
24 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
class _GMMBase(BaseEstimator):

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.verbose = verbose

        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')
```
25 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):

    def _set_parameters(self, params):
        (self.weight_concentration_, self.mean_precision_, self.means_,
         self.degrees_of_freedom_, self.covariances_,
         self.precisions_cholesky_) = params

        # Weights computation
        if self.weight_concentration_prior_type == "dirichlet_process":
            weight_dirichlet_sum = (self.weight_concentration_[0] +
                                    self.weight_concentration_[1])
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0] / weight_dirichlet_sum *
                np.hstack((1, np.cumprod(tmp[:-1]))))
            self.weights_ /= np.sum(self.weights_)
        else:
            self. weights_ = (self.weight_concentration_ /
                              np.sum(self.weight_concentration_))

        # Precisions matrices computation
        if self.covariance_type == 'full':
            self.precisions_ = np.array([
                np.dot(prec_chol, prec_chol.T)
                for prec_chol in self.precisions_cholesky_])

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2
```
**26 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights
```
27 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
```python
@deprecated("GaussianProcess was deprecated in version 0.18 and will be "
            "removed in 0.20. Use the GaussianProcessRegressor instead.")
class GaussianProcess(BaseEstimator, RegressorMixin):

    def _arg_max_reduced_likelihood_function(self):
        # ... other code

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
28 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
class _DPGMMBase(_GMMBase):

    def _fit(self, X, y=None):
        # ... other code

        if 'c' in self.init_params or not hasattr(self, 'precs_'):
            if self.covariance_type == 'spherical':
                self.dof_ = np.ones(self.n_components)
                self.scale_ = np.ones(self.n_components)
                self.precs_ = np.ones((self.n_components, n_features))
                self.bound_prec_ = 0.5 * n_features * (
                    digamma(self.dof_) - np.log(self.scale_))
            elif self.covariance_type == 'diag':
                self.dof_ = 1 + 0.5 * n_features
                self.dof_ *= np.ones((self.n_components, n_features))
                self.scale_ = np.ones((self.n_components, n_features))
                self.precs_ = np.ones((self.n_components, n_features))
                self.bound_prec_ = 0.5 * (np.sum(digamma(self.dof_) -
                                                 np.log(self.scale_), 1))
                self.bound_prec_ -= 0.5 * np.sum(self.precs_, 1)
            elif self.covariance_type == 'tied':
                self.dof_ = 1.
                self.scale_ = np.identity(n_features)
                self.precs_ = np.identity(n_features)
                self.det_scale_ = 1.
                self.bound_prec_ = 0.5 * wishart_log_det(
                    self.dof_, self.scale_, self.det_scale_, n_features)
                self.bound_prec_ -= 0.5 * self.dof_ * np.trace(self.scale_)
            elif self.covariance_type == 'full':
                self.dof_ = (1 + self.n_components + n_samples)
                self.dof_ *= np.ones(self.n_components)
                self.scale_ = [2 * np.identity(n_features)
                               for _ in range(self.n_components)]
                self.precs_ = [np.identity(n_features)
                               for _ in range(self.n_components)]
                self.det_scale_ = np.ones(self.n_components)
                self.bound_prec_ = np.zeros(self.n_components)
                for k in range(self.n_components):
                    self.bound_prec_[k] = wishart_log_det(
                        self.dof_[k], self.scale_[k], self.det_scale_[k],
                        n_features)
                    self.bound_prec_[k] -= (self.dof_[k] *
                                            np.trace(self.scale_[k]))
                self.bound_prec_ *= 0.5

        # EM algorithms
        current_log_likelihood = None
        # reset self.converged_ to False
        self.converged_ = False

        for i in range(self.n_iter):
            prev_log_likelihood = current_log_likelihood
            # Expectation step
            curr_logprob, z = self.score_samples(X)

            current_log_likelihood = (
                curr_logprob.mean() + self._logprior(z) / n_samples)

            # Check for convergence.
            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

            # Maximization step
            self._do_mstep(X, z, self.params)

        if self.n_iter == 0:
            # Need to make sure that there is a z value to output
            # Output zeros because it was just a quick initialization
            z = np.zeros((X.shape[0], self.n_components))

        self._set_weights()

        return z
```
**29 - /tmp/repos/scikit-learn/sklearn/mixture/base.py**:
```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        # Check all the parameters values of the derived class
        self._check_parameters(X)
```
