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
1 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
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
**2 - /tmp/repos/scikit-learn/sklearn/linear_model/least_angle.py**:
```python
class Lars(LinearModel, RegressorMixin):


    def _fit(self, X, y, max_iter, alpha, fit_path, Xy=None):
        """Auxiliary method to fit the model using X, y as training data"""
        n_features = X.shape[1]

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y,
                                                        self.fit_intercept,
                                                        self.normalize,
                                                        self.copy_X)

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_targets = y.shape[1]

        Gram = self._get_gram(self.precompute, X, y)

        self.alphas_ = []
        self.n_iter_ = []
        self.coef_ = np.empty((n_targets, n_features))

        if fit_path:
            self.active_ = []
            self.coef_path_ = []
            for k in xrange(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                alphas, active, coef_path, n_iter_ = lars_path(
                    X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X,
                    copy_Gram=True, alpha_min=alpha, method=self.method,
                    verbose=max(0, self.verbose - 1), max_iter=max_iter,
                    eps=self.eps, return_path=True,
                    return_n_iter=True, positive=self.positive)
                self.alphas_.append(alphas)
                self.active_.append(active)
                self.n_iter_.append(n_iter_)
                self.coef_path_.append(coef_path)
                self.coef_[k] = coef_path[:, -1]

            if n_targets == 1:
                self.alphas_, self.active_, self.coef_path_, self.coef_ = [
                    a[0] for a in (self.alphas_, self.active_, self.coef_path_,
                                   self.coef_)]
                self.n_iter_ = self.n_iter_[0]
        else:
            for k in xrange(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                alphas, _, self.coef_[k], n_iter_ = lars_path(
                    X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X,
                    copy_Gram=True, alpha_min=alpha, method=self.method,
                    verbose=max(0, self.verbose - 1), max_iter=max_iter,
                    eps=self.eps, return_path=False, return_n_iter=True,
                    positive=self.positive)
                self.alphas_.append(alphas)
                self.n_iter_.append(n_iter_)
            if n_targets == 1:
                self.alphas_ = self.alphas_[0]
                self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)
        return self
```
