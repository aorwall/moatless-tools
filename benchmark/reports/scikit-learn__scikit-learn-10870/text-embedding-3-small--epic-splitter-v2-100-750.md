0 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
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
**1 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
"""

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
**2 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
"""

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
3 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
4 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python


    def _bound_proportions(self, z):
        logprior = 0.
        dg = digamma(self.gamma_)
        dg -= digamma(np.sum(self.gamma_))
        logprior += np.sum(dg.reshape((-1, 1)) * z.T)
        z_non_zeros = z[z > np.finfo(np.float32).eps]
        logprior -= np.sum(z_non_zeros * np.log(z_non_zeros))
        return logprior

    def _bound_concentration(self):
        logprior = gammaln(np.sum(self.gamma_)) - gammaln(self.n_components
                                                          * self.alpha_)
        logprior -= np.sum(gammaln(self.gamma_) - gammaln(self.alpha_))
        sg = digamma(np.sum(self.gamma_))
        logprior += np.sum((self.gamma_ - self.alpha_)
                           * (digamma(self.gamma_) - sg))
        return logprior

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
5 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
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
```
**6 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
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
7 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
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

        
```
8 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
9 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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

    
```
10 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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

    
```
**11 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):


    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm
```
12 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
13 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
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
14 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
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
15 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
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
**16 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
class GaussianMixture(BaseMixture):


    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)
```
17 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
# ...


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

    
```
**18 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
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
**19 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
# ...


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
    
```
20 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
class BayesianGaussianMixture(BaseMixture):


    def _check_is_fitted(self):
        check_is_fitted(self, ['weight_concentration_', 'mean_precision_',
                               'means_', 'degrees_of_freedom_',
                               'covariances_', 'precisions_',
                               'precisions_cholesky_'])
```
21 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_sin.py
```python
# ..."""
=================================
Gaussian Mixture Model Sine Curve
=================================

This example demonstrates the behavior of Gaussian mixture models fit on data
that was not sampled from a mixture of Gaussian random variables. The dataset
is formed by 100 points loosely spaced following a noisy sine curve. There is
therefore no ground truth value for the number of Gaussian components.

The first model is a classical Gaussian Mixture Model with 10 components fit
with the Expectation-Maximization algorithm.

The second model is a Bayesian Gaussian Mixture Model with a Dirichlet process
prior fit with variational inference. The low value of the concentration prior
makes the model favor a lower number of active components. This models
"decides" to focus its modeling power on the big picture of the structure of
the dataset: groups of points with alternating directions modeled by
non-diagonal covariance matrices. Those alternating directions roughly capture
the alternating nature of the original sine signal.

The third model is also a Bayesian Gaussian mixture model with a Dirichlet
process prior but this time the value of the concentration prior is higher
giving the model more liberty to model the fine-grained structure of the data.
The result is a mixture with a larger number of active components that is
similar to the first model where we arbitrarily decided to fix the number of
components to 10.

Which model is the best is a matter of subjective judgement: do we want to
favor models that only capture the big picture to summarize and explain most of
the structure of the data while ignoring the details or do we prefer models
that closely follow the high density regions of the signal?

The last two panels show how we can sample from the last two models. The
resulting samples distributions do not look exactly like the original data
distribution. The difference primarily stems from the approximation error we
made by using a model that assumes that the data was generated by a finite
number of Gaussian components instead of a continuous noisy sine curve.

"""

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, index, title):
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


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
             r"for $\gamma_0=100$ sampled with $2000$ samples."
```
22 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
# ...


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
    
```
23 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
24 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
25 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
"""
Gaussian Mixture Models.

This implementation corresponds to frequentist (non-Bayesian) formulation
of Gaussian Mixture Models.
"""

# Author: Ron Weiss <ronweiss@gmail.com>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Bertrand Thirion <bertrand.thirion@inria.fr>

# Important note for the deprecation cleaning of 0.20 :
# All the functions and classes of this file have been deprecated in 0.18.
# When you remove this file please also remove the related files
# - 'sklearn/mixture/dpgmm.py'
# - 'sklearn/mixture/test_dpgmm.py'
# - 'sklearn/mixture/test_gmm.py'
from time import time

import numpy as np
from scipy import linalg

from ..base import BaseEstimator
from ..utils import check_random_state, check_array, deprecated
from ..utils.fixes import logsumexp
from ..utils.validation import check_is_fitted
from .. import cluster

from sklearn.externals.six.moves import zip

EPS = np.finfo(float).eps


_covar_mstep_funcs = {'spherical': _covar_mstep_spherical,
                      'diag': _covar_mstep_diag,
                      'tied': _covar_mstep_tied,
                      'full': _covar_mstep_full,
                      }
```
**26 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
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
27 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
def _bound_means(self):
        "The variational lower bound for the mean parameters"
        logprior = 0.
        logprior -= 0.5 * squared_norm(self.means_)
        logprior -= 0.5 * self.means_.shape[1] * self.n_components
        return logprior

    def _bound_precisions(self):
        """Returns the bound term related to precisions"""
        logprior = 0.
        if self.covariance_type == 'spherical':
            logprior += np.sum(gammaln(self.dof_))
            logprior -= np.sum(
                (self.dof_ - 1) * digamma(np.maximum(0.5, self.dof_)))
            logprior += np.sum(- np.log(self.scale_) + self.dof_
                               - self.precs_[:, 0])
        elif self.covariance_type == 'diag':
            logprior += np.sum(gammaln(self.dof_))
            logprior -= np.sum(
                (self.dof_ - 1) * digamma(np.maximum(0.5, self.dof_)))
            logprior += np.sum(- np.log(self.scale_) + self.dof_ - self.precs_)
        elif self.covariance_type == 'tied':
            logprior += _bound_wishart(self.dof_, self.scale_, self.det_scale_)
        elif self.covariance_type == 'full':
            for k in range(self.n_components):
                logprior += _bound_wishart(self.dof_[k],
                                           self.scale_[k],
                                           self.det_scale_[k])
        return logprior

    def _bound_proportions(self, z):
        """Returns the bound term related to proportions"""
        dg12 = digamma(self.gamma_.T[1] + self.gamma_.T[2])
        dg1 = digamma(self.gamma_.T[1]) - dg12
        dg2 = digamma(self.gamma_.T[2]) - dg12

        cz = stable_cumsum(z[:, ::-1], axis=-1)[:, -2::-1]
        logprior = np.sum(cz * dg2[:-1]) + np.sum(z * dg1)
        del cz  # Save memory
        z_non_zeros = z[z > np.finfo(np.float32).eps]
        logprior -= np.sum(z_non_zeros * np.log(z_non_zeros))
        return logprior

    def _logprior(self, z):
        logprior = self._bound_concentration()
        logprior += self._bound_means()
        logprior += self._bound_precisions()
        logprior += self._bound_proportions(z)
        return logprior

    def lower_bound(self, X, z):
        """returns a lower bound on model evidence based on X and membership"""
        check_is_fitted(self, 'means_')

        if self.covariance_type not in ['full', 'tied', 'diag', 'spherical']:
            raise NotImplementedError("This ctype is not implemented: %s"
                                      % self.covariance_type)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        c = np.sum(z * _bound_state_log_lik(X, self._initial_bound +
                                            self.bound_prec_, self.precs_,
                                            self.means_, self.covariance_type))

        return c + self._logprior(z)

    
```
28 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
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

    def _get_parameters(self):
        return (self.weight_concentration_,
                self.mean_precision_, self.means_,
                self.degrees_of_freedom_, self.covariances_,
                self.precisions_cholesky_)

    
```
29 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
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

    def _get_parameters(self):
        return (self.weight_concentration_,
                self.mean_precision_, self.means_,
                self.degrees_of_freedom_, self.covariances_,
                self.precisions_cholesky_)

    
```
30 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
31 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
32 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
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
# ...
```
33 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
34 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
class _DPGMMBase(_GMMBase):


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
```
35 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
```python
def _check_params(self, n_samples=None):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.corr))

        # Check storage mode
        if self.storage_mode != 'full' and self.storage_mode != 'light':
            raise ValueError("Storage mode should either be 'full' or "
                             "'light', %s was given." % self.storage_mode)

        # Check correlation parameters
        self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            self.thetaL = np.atleast_2d(self.thetaL)
            self.thetaU = np.atleast_2d(self.thetaU)
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        self.nugget = np.asarray(self.nugget)
        if np.any(self.nugget) < 0.:
            raise ValueError("nugget must be positive or zero.")
        if (n_samples is not None
                and self.nugget.shape not in [(), (n_samples,)]):
            raise ValueError("nugget must be either a scalar "
                             "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)
```
36 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
class _DPGMMBase(_GMMBase):


    def _bound_means(self):
        "The variational lower bound for the mean parameters"
        logprior = 0.
        logprior -= 0.5 * squared_norm(self.means_)
        logprior -= 0.5 * self.means_.shape[1] * self.n_components
        return logprior
```
**37 - /tmp/repos/scikit-learn/sklearn/mixture/base.py**:
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
