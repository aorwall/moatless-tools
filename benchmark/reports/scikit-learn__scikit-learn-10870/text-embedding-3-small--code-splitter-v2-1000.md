**0 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
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

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    
```
1 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
2 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
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

    
```
3 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
def _sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                     random_state=None):
    rng = check_random_state(random_state)
    n_dim = len(mean)
    rand = rng.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    if covariance_type == 'spherical':
        rand *= np.sqrt(covar)
    elif covariance_type == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        s, U = linalg.eigh(covar)
        s.clip(0, out=s)  # get rid of tiny negatives
        np.sqrt(s, out=s)
        U *= s
        rand = np.dot(U, rand)

    return (rand.T + mean).T


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
4 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_sin.py
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
5 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
"""Bayesian Gaussian Mixture Model."""
# Author: Wei Xue <xuewei4d@gmail.com>
#         Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import math
import numpy as np
from scipy.special import betaln, digamma, gammaln

from .base import BaseMixture, _check_shape
from .gaussian_mixture import _check_precision_matrix
from .gaussian_mixture import _check_precision_positivity
from .gaussian_mixture import _compute_log_det_cholesky
from .gaussian_mixture import _compute_precision_cholesky
from .gaussian_mixture import _estimate_gaussian_parameters
from .gaussian_mixture import _estimate_log_gaussian_prob
from ..utils import check_array
from ..utils.validation import check_is_fitted


def _log_dirichlet_norm(dirichlet_concentration):
    """Compute the log of the Dirichlet distribution normalization term.

    Parameters
    ----------
    dirichlet_concentration : array-like, shape (n_samples,)
        The parameters values of the Dirichlet distribution.

    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return (gammaln(np.sum(dirichlet_concentration)) -
            np.sum(gammaln(dirichlet_concentration)))


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like, shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precision_chol : array-like, shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like, shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(degrees_of_freedom * log_det_precisions_chol +
             degrees_of_freedom * n_features * .5 * math.log(2.) +
             np.sum(gammaln(.5 * (degrees_of_freedom -
                                  np.arange(n_features)[:, np.newaxis])), 0))


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
6 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
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
7 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
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
8 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python



@deprecated("The class GMM is deprecated in 0.18 and will be "
            " removed in 0.20. Use class GaussianMixture instead.")
class GMM(_GMMBase):
    """
    Legacy Gaussian Mixture Model

    .. deprecated:: 0.18
        This class will be removed in 0.20.
        Use :class:`sklearn.mixture.GaussianMixture` instead.

    """

    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 verbose=0):
        super(GMM, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            random_state=random_state, tol=tol, min_covar=min_covar,
            n_iter=n_iter, n_init=n_init, params=params,
            init_params=init_params, verbose=verbose)

#########################################################################
# some helper routines
#########################################################################


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model."""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if cv.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob



```
**9 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
"""Gaussian Mixture Model."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np

from scipy import linalg

from .base import BaseMixture, _check_shape
from ..externals.six.moves import zip
from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..utils.extmath import row_norms


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


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)



```
10 - /tmp/repos/scikit-learn/sklearn/mixture/bayesian_mixture.py
```python
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

    def _initialize(self, X, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        nk, xk, sk = _estimate_gaussian_parameters(X, resp, self.reg_covar,
                                                   self.covariance_type)

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_weights(self, nk):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)
        """
        if self.weight_concentration_prior_type == 'dirichlet_process':
            # For dirichlet process weight_concentration will be a tuple
            # containing the two parameters of the beta distribution
            self.weight_concentration_ = (
                1. + nk,
                (self.weight_concentration_prior_ +
                 np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            self.weight_concentration_ = self.weight_concentration_prior_ + nk

    def _estimate_means(self, nk, xk):
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)
        """
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = ((self.mean_precision_prior_ * self.mean_prior_ +
                        nk[:, np.newaxis] * xk) /
                       self.mean_precision_[:, np.newaxis])

    
```
11 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
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
**12 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


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
13 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_sin.py
```python
"""
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
```
14 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm_selection.py
```python
"""
================================
Gaussian Mixture Model Selection
================================

This example shows that model selection can be performed with
Gaussian Mixture Models using information-theoretic criteria (BIC).
Model selection concerns both the covariance type
and the number of components in the model.
In that case, AIC also provides the right result (not shown to save time),
but BIC is better suited if the problem is to identify the right model.
Unlike Bayesian procedures, such inferences are prior-free.

In that case, the model with 2 components and full covariance
(which corresponds to the true generative model) is selected.
"""

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, 2 components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
```
15 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
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

@deprecated("The function log_multivariate_normal_density is deprecated in 0.18"
            " and will be removed in 0.20.")
def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'

    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)


@deprecated("The function sample_gaussian is deprecated in 0.18"
            " and will be removed in 0.20."
            " Use numpy.random.multivariate_normal instead.")
def sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                    random_state=None):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean of the distribution.

    covar : array_like
        Covariance of the distribution. The shape depends on `covariance_type`:
            scalar if 'spherical',
            (n_features) if 'diag',
            (n_features, n_features)  if 'tied', or 'full'

    covariance_type : string, optional
        Type of the covariance parameters. Must be one of
        'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.

    n_samples : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    X : array
        Randomly generated sample. The shape depends on `n_samples`:
        (n_features,) if `1`
        (n_features, n_samples) otherwise
    """
    return _sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                            random_state=None)



```
16 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
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
17 - /tmp/repos/scikit-learn/examples/mixture/plot_concentration_prior.py
```python
"""
========================================================================
Concentration Prior Type Analysis of Variation Bayesian Gaussian Mixture
========================================================================

This example plots the ellipsoids obtained from a toy dataset (mixture of three
Gaussians) fitted by the ``BayesianGaussianMixture`` class models with a
Dirichlet distribution prior
(``weight_concentration_prior_type='dirichlet_distribution'``) and a Dirichlet
process prior (``weight_concentration_prior_type='dirichlet_process'``). On
each figure, we plot the results for three different values of the weight
concentration prior.

The ``BayesianGaussianMixture`` class can adapt its number of mixture
components automatically. The parameter ``weight_concentration_prior`` has a
direct link with the resulting number of components with non-zero weights.
Specifying a low value for the concentration prior will make the model put most
of the weight on few components set the remaining components weights very close
to zero. High values of the concentration prior will allow a larger number of
components to be active in the mixture.

The Dirichlet process prior allows to define an infinite number of components
and automatically selects the correct number of components: it activates a
component only if it is necessary.

On the contrary the classical finite mixture model with a Dirichlet
distribution prior will favor more uniformly weighted components and therefore
tends to divide natural clusters into unnecessary sub-components.
"""
# Author: Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.mixture import BayesianGaussianMixture

print(__doc__)


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker='o', color=colors[y], alpha=0.8)
    ax1.set_xlim(-2., 2.)
    ax1.set_ylim(-3., 3.)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_,
                  estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction='out')
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    ax2.set_xlim(-.6, 2 * n_components - .4)
    ax2.set_ylim(0., 1.1)
    ax2.tick_params(axis='y', which='both', left='off',
                    right='off', labelleft='off')
    ax2.tick_params(axis='x', which='both', top='off')

    if plot_title:
        ax1.set_ylabel('Estimated Mixtures')
        ax2.set_ylabel('Weight of each component')
```
18 - /tmp/repos/scikit-learn/examples/mixture/plot_gmm.py
```python
"""
=================================
Gaussian Mixture Model Ellipsoids
=================================

Plot the confidence ellipsoids of a mixture of two Gaussians
obtained with Expectation Maximisation (``GaussianMixture`` class) and
Variational Inference (``BayesianGaussianMixture`` class models with
a Dirichlet process prior).

Both models have access to five components with which to fit the data. Note
that the Expectation Maximisation model will necessarily use all five
components while the Variational Inference model will effectively only use as
many as are needed for a good fit. Here we can see that the Expectation
Maximisation model splits some components arbitrarily, because it is trying to
fit too many components, while the Dirichlet Process model adapts it number of
state automatically.

This example doesn't show it, as we're in a low-dimensional space, but
another advantage of the Dirichlet process model is that it can fit
full covariance matrices effectively even when there are less examples
per cluster than there are dimensions in the data, due to
regularization properties of the inference algorithm.
"""

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()

```
19 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
"""Bayesian Gaussian Mixture Models and
Dirichlet Process Gaussian Mixture Models"""
from __future__ import print_function

# Author: Alexandre Passos (alexandre.tp@gmail.com)
#         Bertrand Thirion <bertrand.thirion@inria.fr>
#
# Based on mixture.py by:
#         Ron Weiss <ronweiss@gmail.com>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#

# Important note for the deprecation cleaning of 0.20 :
# All the function and classes of this file have been deprecated in 0.18.
# When you remove this file please also remove the related files
# - 'sklearn/mixture/gmm.py'
# - 'sklearn/mixture/test_dpgmm.py'
# - 'sklearn/mixture/test_gmm.py'

import numpy as np
from scipy.special import digamma as _digamma, gammaln as _gammaln
from scipy import linalg
from scipy.linalg import pinvh
from scipy.spatial.distance import cdist

from ..externals.six.moves import xrange
from ..utils import check_random_state, check_array, deprecated
from ..utils.fixes import logsumexp
from ..utils.extmath import squared_norm, stable_cumsum
from ..utils.validation import check_is_fitted
from .. import cluster
from .gmm import _GMMBase


@deprecated("The function digamma is deprecated in 0.18 and "
            "will be removed in 0.20. Use scipy.special.digamma instead.")
def digamma(x):
    return _digamma(x + np.finfo(np.float32).eps)


@deprecated("The function gammaln is deprecated in 0.18 and "
            "will be removed in 0.20. Use scipy.special.gammaln instead.")
def gammaln(x):
    return _gammaln(x + np.finfo(np.float32).eps)


@deprecated("The function log_normalize is deprecated in 0.18 and "
            "will be removed in 0.20.")
def log_normalize(v, axis=0):
    """Normalized probabilities from unnormalized log-probabilities"""
    v = np.rollaxis(v, axis)
    v = v.copy()
    v -= v.max(axis=0)
    out = logsumexp(v)
    v = np.exp(v - out)
    v += np.finfo(np.float32).eps
    v /= np.sum(v, axis=0)
    return np.swapaxes(v, 0, axis)


@deprecated("The function wishart_log_det is deprecated in 0.18 and "
            "will be removed in 0.20.")
def wishart_log_det(a, b, detB, n_features):
    """Expected value of the log of the determinant of a Wishart

    The expected value of the logarithm of the determinant of a
    wishart-distributed random variable with the specified parameters."""
    l = np.sum(digamma(0.5 * (a - np.arange(-1, n_features - 1))))
    l += n_features * np.log(2)
    return l + detB


@deprecated("The function wishart_logz is deprecated in 0.18 and "
            "will be removed in 0.20.")
def wishart_logz(v, s, dets, n_features):
    "The logarithm of the normalization constant for the wishart distribution"
    z = 0.
    z += 0.5 * v * n_features * np.log(2)
    z += (0.25 * (n_features * (n_features - 1)) * np.log(np.pi))
    z += 0.5 * v * np.log(dets)
    z += np.sum(gammaln(0.5 * (v - np.arange(n_features) + 1)))
    return z


def _bound_wishart(a, B, detB):
    """Returns a function of the dof, scale matrix and its determinant
    used as an upper bound in variational approximation of the evidence"""
    n_features = B.shape[0]
    logprior = wishart_logz(a, B, detB, n_features)
    logprior -= wishart_logz(n_features,
                             np.identity(n_features),
                             1, n_features)
    logprior += 0.5 * (a - 1) * wishart_log_det(a, B, detB, n_features)
    logprior += 0.5 * a * np.trace(B)
    return logprior


##############################################################################
# Variational bound on the log likelihood of each class
##############################################################################



```
20 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
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


@deprecated("The `DPGMM` class is not working correctly and it's better "
            "to use `sklearn.mixture.BayesianGaussianMixture` class with "
            "parameter `weight_concentration_prior_type='dirichlet_process'` "
            "instead. DPGMM is deprecated in 0.18 and will be "
            "removed in 0.20.")
class DPGMM(_DPGMMBase):
    """Dirichlet Process Gaussian Mixture Models

    .. deprecated:: 0.18
        This class will be removed in 0.20.
        Use :class:`sklearn.mixture.BayesianGaussianMixture` with
        parameter ``weight_concentration_prior_type='dirichlet_process'``
        instead.

    """

    def __init__(self, n_components=1, covariance_type='diag', alpha=1.0,
                 random_state=None, tol=1e-3, verbose=0, min_covar=None,
                 n_iter=10, params='wmc', init_params='wmc'):
        super(DPGMM, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            alpha=alpha, random_state=random_state, tol=tol, verbose=verbose,
            min_covar=min_covar, n_iter=n_iter, params=params,
            init_params=init_params)


@deprecated("The `VBGMM` class is not working correctly and it's better "
            "to use `sklearn.mixture.BayesianGaussianMixture` class with "
            "parameter `weight_concentration_prior_type="
            "'dirichlet_distribution'` instead. "
            "VBGMM is deprecated in 0.18 and will be removed in 0.20.")
class VBGMM(_DPGMMBase):
    
```
21 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
def _get_covars(self):
        return [pinvh(c) for c in self._get_precisions()]

    def _set_covars(self, covars):
        raise NotImplementedError("""The variational algorithm does
        not support setting the covariance parameters.""")

    def score_samples(self, X):
        """Return the likelihood of the data under the model.

        Compute the bound on log probability of X under the model
        and return the posterior distribution (responsibilities) of
        each mixture component for each element of X.

        This is done by computing the parameters for the mean-field of
        z for each observation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'gamma_')

        X = check_array(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        sd = digamma(self.gamma_.T[1] + self.gamma_.T[2])
        dgamma1 = digamma(self.gamma_.T[1]) - sd
        dgamma2 = np.zeros(self.n_components)
        dgamma2[0] = digamma(self.gamma_[0, 2]) - digamma(self.gamma_[0, 1] +
                                                          self.gamma_[0, 2])
        for j in range(1, self.n_components):
            dgamma2[j] = dgamma2[j - 1] + digamma(self.gamma_[j - 1, 2])
            dgamma2[j] -= sd[j - 1]
        dgamma = dgamma1 + dgamma2
        # Free memory and developers cognitive load:
        del dgamma1, dgamma2, sd

        if self.covariance_type not in ['full', 'tied', 'diag', 'spherical']:
            raise NotImplementedError("This ctype is not implemented: %s"
                                      % self.covariance_type)
        p = _bound_state_log_lik(X, self._initial_bound + self.bound_prec_,
                                 self.precs_, self.means_,
                                 self.covariance_type)
        z = p + dgamma
        z = log_normalize(z, axis=-1)
        bound = np.sum(z * p, axis=-1)
        return bound, z

    def _update_concentration(self, z):
        """Update the concentration parameters for each cluster"""
        sz = np.sum(z, axis=0)
        self.gamma_.T[1] = 1. + sz
        self.gamma_.T[2].fill(0)
        for i in range(self.n_components - 2, -1, -1):
            self.gamma_[i, 2] = self.gamma_[i + 1, 2] + sz[i]
        self.gamma_.T[2] += self.alpha

    def _update_means(self, X, z):
        """Update the variational distributions for the means"""
        n_features = X.shape[1]
        for k in range(self.n_components):
            if self.covariance_type in ['spherical', 'diag']:
                num = np.sum(z.T[k].reshape((-1, 1)) * X, axis=0)
                num *= self.precs_[k]
                den = 1. + self.precs_[k] * np.sum(z.T[k])
                self.means_[k] = num / den
            elif self.covariance_type in ['tied', 'full']:
                if self.covariance_type == 'tied':
                    cov = self.precs_
                else:
                    cov = self.precs_[k]
                den = np.identity(n_features) + cov * np.sum(z.T[k])
                num = np.sum(z.T[k].reshape((-1, 1)) * X, axis=0)
                num = np.dot(cov, num)
                self.means_[k] = linalg.lstsq(den, num)[0]

    
```
22 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
**23 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
```python
def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

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

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

```
24 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
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
25 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
def _fit(self, X, y=None):
        """Estimate model parameters with the variational algorithm.

        For a full derivation and description of the algorithm see
        doc/modules/dp-derivation.rst
        or
        http://scikit-learn.org/stable/modules/dp-derivation.html

        A initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating
        the object. Likewise, if you just would like to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """
        self.alpha_ = float(self.alpha) / self.n_components
        return super(VBGMM, self)._fit(X, y)

    def score_samples(self, X):
        """Return the likelihood of the data under the model.

        Compute the bound on log probability of X under the model
        and return the posterior distribution (responsibilities) of
        each mixture component for each element of X.

        This is done by computing the parameters for the mean-field of
        z for each observation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'gamma_')

        X = check_array(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        dg = digamma(self.gamma_) - digamma(np.sum(self.gamma_))

        if self.covariance_type not in ['full', 'tied', 'diag', 'spherical']:
            raise NotImplementedError("This ctype is not implemented: %s"
                                      % self.covariance_type)
        p = _bound_state_log_lik(X, self._initial_bound + self.bound_prec_,
                                 self.precs_, self.means_,
                                 self.covariance_type)

        z = p + dg
        z = log_normalize(z, axis=-1)
        bound = np.sum(z * p, axis=-1)
        return bound, z

    def _update_concentration(self, z):
        for i in range(self.n_components):
            self.gamma_[i] = self.alpha_ + np.sum(z.T[i])

    def _initialize_gamma(self):
        self.gamma_ = self.alpha_ * np.ones(self.n_components)

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
26 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
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

    def _set_weights(self):
        for i in xrange(self.n_components):
            self.weights_[i] = self.gamma_[i, 1] / (self.gamma_[i, 1]
                                                    + self.gamma_[i, 2])
        self.weights_ /= np.sum(self.weights_)

    def _fit(self, X, y=None):
        
```
**27 - /tmp/repos/scikit-learn/sklearn/mixture/gaussian_mixture.py**:
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


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar



```
28 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
"""Variational Inference for the Infinite Gaussian Mixture Model.

    DPGMM stands for Dirichlet Process Gaussian Mixture Model, and it
    is an infinite mixture model with the Dirichlet Process as a prior
    distribution on the number of clusters. In practice the
    approximate inference algorithm uses a truncated distribution with
    a fixed maximum number of components, but almost always the number
    of components actually used depends on the data.

    Stick-breaking Representation of a Gaussian mixture model
    probability distribution. This class allows for easy and efficient
    inference of an approximate posterior distribution over the
    parameters of a Gaussian mixture model with a variable number of
    components (smaller than the truncation parameter n_components).

    Initialization is with normally-distributed means and identity
    covariance, for proper convergence.

    Read more in the :ref:`User Guide <dpgmm>`.

    Parameters
    ----------
    n_components : int, default 1
        Number of mixture components.

    covariance_type : string, default 'diag'
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    alpha : float, default 1
        Real number representing the concentration parameter of
        the dirichlet process. Intuitively, the Dirichlet Process
        is as likely to start a new cluster for a point as it is
        to add that point to a cluster with alpha elements. A
        higher alpha means more clusters, as the expected number
        of clusters is ``alpha*log(N)``.

    tol : float, default 1e-3
        Convergence threshold.

    n_iter : int, default 10
        Maximum number of iterations to perform before convergence.

    params : string, default 'wmc'
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.

    init_params : string, default 'wmc'
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    verbose : int, default 0
        Controls output verbosity.

    Attributes
    ----------
    covariance_type : string
        String describing the type of covariance parameters used by
        the DP-GMM.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_components : int
        Number of mixture components.

    weights_ : array, shape (`n_components`,)
        Mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    precs_ : array
        Precision (inverse covariance) parameters for each mixture
        component.  The shape depends on `covariance_type`::

            (`n_components`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)                  if 'tied',
            (`n_components`, `n_features`)                if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------
    GMM : Finite Gaussian mixture model fit with EM

    VBGMM : Finite Gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.
    """
    def __init__(self, n_components=1, covariance_type='diag', alpha=1.0,
                 random_state=None, tol=1e-3, verbose=0, min_covar=None,
                 n_iter=10, params='wmc', init_params='wmc'):
        self.alpha = alpha
        super(_DPGMMBase, self).__init__(n_components, covariance_type,
                                         random_state=random_state,
                                         tol=tol, min_covar=min_covar,
                                         n_iter=n_iter, params=params,
                                         init_params=init_params,
                                         verbose=verbose)

    def _get_precisions(self):
        """Return precisions as a full matrix."""
        if self.covariance_type == 'full':
            return self.precs_
        elif self.covariance_type in ['diag', 'spherical']:
            return [np.diag(cov) for cov in self.precs_]
        elif self.covariance_type == 'tied':
            return [self.precs_] * self.n_components

    
```
29 - /tmp/repos/scikit-learn/sklearn/mixture/dpgmm.py
```python
"""Variational Inference for the Gaussian Mixture Model

    .. deprecated:: 0.18
        This class will be removed in 0.20.
        Use :class:`sklearn.mixture.BayesianGaussianMixture` with parameter
        ``weight_concentration_prior_type='dirichlet_distribution'`` instead.

    Variational inference for a Gaussian mixture model probability
    distribution. This class allows for easy and efficient inference
    of an approximate posterior distribution over the parameters of a
    Gaussian mixture model with a fixed number of components.

    Initialization is with normally-distributed means and identity
    covariance, for proper convergence.

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, default 1
        Number of mixture components.

    covariance_type : string, default 'diag'
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    alpha : float, default 1
        Real number representing the concentration parameter of
        the dirichlet distribution. Intuitively, the higher the
        value of alpha the more likely the variational mixture of
        Gaussians model will use all components it can.

    tol : float, default 1e-3
        Convergence threshold.

    n_iter : int, default 10
        Maximum number of iterations to perform before convergence.

    params : string, default 'wmc'
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.

    init_params : string, default 'wmc'
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    verbose : int, default 0
        Controls output verbosity.

    Attributes
    ----------
    covariance_type : string
        String describing the type of covariance parameters used by
        the DP-GMM.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_features : int
        Dimensionality of the Gaussians.

    n_components : int (read-only)
        Number of mixture components.

    weights_ : array, shape (`n_components`,)
        Mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    precs_ : array
        Precision (inverse covariance) parameters for each mixture
        component.  The shape depends on `covariance_type`::

            (`n_components`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)                  if 'tied',
            (`n_components`, `n_features`)                if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False
        otherwise.

    See Also
    --------
    GMM : Finite Gaussian mixture model fit with EM
    DPGMM : Infinite Gaussian mixture model, using the dirichlet
        process, fit with a variational algorithm
    """

    def __init__(self, n_components=1, covariance_type='diag', alpha=1.0,
                 random_state=None, tol=1e-3, verbose=0,
                 min_covar=None, n_iter=10, params='wmc', init_params='wmc'):
        super(VBGMM, self).__init__(
            n_components, covariance_type, random_state=random_state,
            tol=tol, verbose=verbose, min_covar=min_covar,
            n_iter=n_iter, params=params, init_params=init_params)
        self.alpha = alpha

    
```
30 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
"""

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

    def _get_covars(self):
        """Covariance parameters for each mixture component.

        The shape depends on ``cvtype``::

            (n_states, n_features)                if 'spherical',
            (n_features, n_features)              if 'tied',
            (n_states, n_features)                if 'diag',
            (n_states, n_features, n_features)    if 'full'

        """
        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == 'tied':
            return [self.covars_] * self.n_components
        elif self.covariance_type == 'spherical':
            return [np.diag(cov) for cov in self.covars_]

    def _set_covars(self, covars):
        """Provide values for covariance."""
        covars = np.asarray(covars)
        _validate_covars(covars, self.covariance_type, self.n_components)
        self.covars_ = covars

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'means_')

        X = check_array(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        lpr = (log_multivariate_normal_density(X, self.means_, self.covars_,
                                               self.covariance_type) +
               np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    
```
31 - /tmp/repos/scikit-learn/sklearn/mixture/gmm.py
```python
if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:  # self.n_iter == 0 occurs when using GMM within HMM
            # Need to make sure that there are responsibilities to output
            # Output zeros because it was just a quick initialization
            responsibilities = np.zeros((X.shape[0], self.n_components))

        return responsibilities

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self._fit(X, y)
        return self

    def _do_mstep(self, X, responsibilities, params, min_covar=0):
        """Perform the Mstep of the EM algorithm and return the cluster weights.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic : float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic : float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()
```
32 - /tmp/repos/scikit-learn/sklearn/manifold/mds.py
```python
if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None

    if n_jobs == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed)
            for seed in seeds)
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class MDS(BaseEstimator):
    
```
33 - /tmp/repos/scikit-learn/examples/mixture/plot_concentration_prior.py
```python


# Parameters of the dataset
random_state, n_components, n_features = 2, 3, 2
colors = np.array(['#0072B2', '#F0E442', '#D55E00'])

covars = np.array([[[.7, .0], [.0, .1]],
                   [[.5, .0], [.0, .1]],
                   [[.5, .0], [.0, .1]]])
samples = np.array([200, 500, 200])
means = np.array([[.0, -.70],
                  [.0, .0],
                  [.0, .70]])

# mean_precision_prior= 0.8 to minimize the influence of the prior
estimators = [
    ("Finite mixture with a Dirichlet distribution\nprior and "
     r"$\gamma_0=$", BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_distribution",
        n_components=2 * n_components, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state), [0.001, 1, 1000]),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
     BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=2 * n_components, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state), [1, 1000, 100000])]

# Generate data
rng = np.random.RandomState(random_state)
X = np.vstack([
    rng.multivariate_normal(means[j], covars[j], samples[j])
    for j in range(n_components)])
y = np.concatenate([j * np.ones(samples[j], dtype=int)
                    for j in range(n_components)])

# Plot results in two different figures
for (title, estimator, concentrations_prior) in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    plt.subplots_adjust(bottom=.04, top=0.90, hspace=.05, wspace=.05,
                        left=.03, right=.99)

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        estimator.fit(X)
        plot_results(plt.subplot(gs[0:2, k]), plt.subplot(gs[2, k]), estimator,
                     X, y, r"%s$%.1e$" % (title, concentration),
                     plot_title=k == 0)

plt.show()

```
34 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
```python
if batch_size is None:
            # No memory management
            # (evaluates all given points in a single batch run)

            # Normalize input
            X = (X - self.X_mean) / self.X_std

            # Get pairwise componentwise L1-distances to the input training set
            dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
            # Get regression function and correlation
            f = self.regr(X)
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            # Scaled predictor
            y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

            # Predictor
            y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

            if self.y_ndim_ == 1:
                y = y.ravel()

            # Mean Squared Error
            if eval_MSE:
                C = self.C
                if C is None:
                    # Light storage mode (need to recompute C, F, Ft and G)
                    if self.verbose:
                        print("This GaussianProcess used 'light' storage mode "
                              "at instantiation. Need to recompute "
                              "autocorrelation matrix...")
                    reduced_likelihood_function_value, par = \
                        self.reduced_likelihood_function()
                    self.C = par['C']
                    self.Ft = par['Ft']
                    self.G = par['G']

                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.beta0 is None:
                    # Universal Kriging
                    u = linalg.solve_triangular(self.G.T,
                                                np.dot(self.Ft.T, rt) - f.T,
                                                lower=True)
                else:
                    # Ordinary Kriging
                    u = np.zeros((n_targets, n_eval))

                MSE = np.dot(self.sigma2.reshape(n_targets, 1),
                             (1. - (rt ** 2.).sum(axis=0)
                              + (u ** 2.).sum(axis=0))[np.newaxis, :])
                MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                # Mean Squared Error might be slightly negative depending on
                # machine precision: force to zero!
                MSE[MSE < 0.] = 0.

                if self.y_ndim_ == 1:
                    MSE = MSE.ravel()

                return y, MSE

            else:

                return y

        else:
            # Memory management

            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_MSE:

                y, MSE = np.zeros(n_eval), np.zeros(n_eval)
                for k in range(max(1, int(n_eval / batch_size))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to], MSE[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y, MSE

            else:

                y = np.zeros(n_eval)
                for k in range(max(1, int(n_eval / batch_size))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y

    
```
35 - /tmp/repos/scikit-learn/examples/gaussian_process/plot_gpr_noisy.py
```python
"""
=============================================================
Gaussian process regression (GPR) with noise-level estimation
=============================================================

This example illustrates that GPR with a sum-kernel including a WhiteKernel can
estimate the noise level of data. An illustration of the
log-marginal-likelihood (LML) landscape shows that there exist two local
maxima of LML. The first corresponds to a model with a high noise level and a
large length scale, which explains all variations in the data by noise. The
second one has a smaller noise level and shorter length scale, which explains
most of the variation by the noise-free functional relationship. The second
model has a higher likelihood; however, depending on the initial value for the
hyperparameters, the gradient-based optimization might also converge to the
high-noise solution. It is thus important to repeat the optimization several
times for different initializations.
"""
print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


rng = np.random.RandomState(0)
X = rng.uniform(0, 5, 20)[:, np.newaxis]
y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])

# First run
plt.figure(0)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          % (kernel, gp.kernel_,
             gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()

# Second run
plt.figure(1)
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          % (kernel, gp.kernel_,
             gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()

# Plot LML landscape
plt.figure(2)
theta0 = np.logspace(-2, 3, 49)
theta1 = np.logspace(-2, 0, 50)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
        for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]

```
**36 - /tmp/repos/scikit-learn/sklearn/mixture/base.py**:
```python
"""Base class for mixture models."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np

from .. import cluster
from ..base import BaseEstimator
from ..base import DensityMixin
from ..externals import six
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state
from ..utils.fixes import logsumexp


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components, tol, reg_covar,
                 max_iter, n_init, init_params, random_state, warm_start,
                 verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

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

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        """
        pass

    
```
