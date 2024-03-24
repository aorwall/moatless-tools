0 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
@deprecated("The class RandomizedLasso is deprecated in 0.19"
            " and will be removed in 0.21.")
class RandomizedLasso(BaseRandomizedLinearModel):

    def _make_estimator_and_params(self, X, y):
        alpha = self.alpha
        if isinstance(alpha, six.string_types) and alpha in ('aic', 'bic'):
            model = LassoLarsIC(precompute=self.precompute,
                                criterion=self.alpha,
                                max_iter=self.max_iter,
                                eps=self.eps)
            model.fit(X, y)
            self.alpha_ = alpha = model.alpha_

        precompute = self.precompute
        # A precomputed Gram array is useless, since _randomized_lasso
        # change X a each iteration
        if hasattr(precompute, '__array__'):
            precompute = 'auto'
        assert precompute in (True, False, None, 'auto')
        return _randomized_lasso, dict(alpha=alpha, max_iter=self.max_iter,
                                       eps=self.eps,
                                       precompute=precompute)
```
**1 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py**:
```python
def _lars_path_residues(X_train, y_train, X_test, y_test, Gram=None,
                        copy=True, method='lars', verbose=False,
                        fit_intercept=True, normalize=True, max_iter=500,
                        eps=np.finfo(np.float).eps, positive=False):
    X_train = _check_copy_and_writeable(X_train, copy)
    y_train = _check_copy_and_writeable(y_train, copy)
    X_test = _check_copy_and_writeable(X_test, copy)
    y_test = _check_copy_and_writeable(y_test, copy)

    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    if normalize:
        norms = np.sqrt(np.sum(X_train ** 2, axis=0))
        nonzeros = np.flatnonzero(norms)
        X_train[:, nonzeros] /= norms[nonzeros]

    alphas, active, coefs = lars_path(
        X_train, y_train, Gram=Gram, copy_X=False, copy_Gram=False,
        method=method, verbose=max(0, verbose - 1), max_iter=max_iter, eps=eps,
        positive=positive)
    if normalize:
        coefs[nonzeros] /= norms[nonzeros][:, np.newaxis]
    residues = np.dot(X_test, coefs) - y_test[:, np.newaxis]
    return alphas, active, coefs, residues.T
```
