# scikit-learn__scikit-learn-13280

| **scikit-learn/scikit-learn** | `face9daf045846bb0a39bfb396432c8685570cdd` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 16123 |
| **Any found context length** | 16123 |
| **Avg pos** | 42.0 |
| **Min pos** | 42 |
| **Max pos** | 42 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/naive_bayes.py b/sklearn/naive_bayes.py
--- a/sklearn/naive_bayes.py
+++ b/sklearn/naive_bayes.py
@@ -460,8 +460,14 @@ def _update_class_log_prior(self, class_prior=None):
                                  " classes.")
             self.class_log_prior_ = np.log(class_prior)
         elif self.fit_prior:
+            with warnings.catch_warnings():
+                # silence the warning when count is 0 because class was not yet
+                # observed
+                warnings.simplefilter("ignore", RuntimeWarning)
+                log_class_count = np.log(self.class_count_)
+
             # empirical prior, with sample_weight taken into account
-            self.class_log_prior_ = (np.log(self.class_count_) -
+            self.class_log_prior_ = (log_class_count -
                                      np.log(self.class_count_.sum()))
         else:
             self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/naive_bayes.py | 463 | 463 | 42 | 2 | 16123


## Problem Statement

```
partial_fit does not account for unobserved target values when fitting priors to data
My understanding is that priors should be fitted to the data using observed target frequencies **and a variant of [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) to avoid assigning 0 probability to targets  not yet observed.**


It seems the implementation of `partial_fit` does not account for unobserved targets at the time of the first training batch when computing priors.

\`\`\`python
    import numpy as np
    import sklearn
    from sklearn.naive_bayes import MultinomialNB
    
    print('scikit-learn version:', sklearn.__version__)
    
    # Create toy training data
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    
    # All possible targets
    classes = np.append(y, 7)
    
    clf = MultinomialNB()
    clf.partial_fit(X, y, classes=classes)
\`\`\`
-----------------------------------
    /home/skojoian/.local/lib/python3.6/site-packages/sklearn/naive_bayes.py:465: RuntimeWarning: divide by zero encountered in log
      self.class_log_prior_ = (np.log(self.class_count_) -
    scikit-learn version: 0.20.2

This behavior is not very intuitive to me. It seems `partial_fit` requires `classes` for the right reason, but doesn't actually offset target frequencies to account for unobserved targets.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/multiclass.py | 220 | 274| 513 | 513 | 6439 | 
| 2 | 1 sklearn/multiclass.py | 516 | 566| 424 | 937 | 6439 | 
| 3 | **2 sklearn/naive_bayes.py** | 399 | 425| 246 | 1183 | 14830 | 
| 4 | **2 sklearn/naive_bayes.py** | 309 | 397| 802 | 1985 | 14830 | 
| 5 | **2 sklearn/naive_bayes.py** | 630 | 709| 730 | 2715 | 14830 | 
| 6 | **2 sklearn/naive_bayes.py** | 106 | 165| 542 | 3257 | 14830 | 
| 7 | **2 sklearn/naive_bayes.py** | 840 | 909| 705 | 3962 | 14830 | 
| 8 | **2 sklearn/naive_bayes.py** | 911 | 931| 217 | 4179 | 14830 | 
| 9 | **2 sklearn/naive_bayes.py** | 563 | 611| 430 | 4609 | 14830 | 
| 10 | **2 sklearn/naive_bayes.py** | 483 | 561| 755 | 5364 | 14830 | 
| 11 | **2 sklearn/naive_bayes.py** | 167 | 191| 195 | 5559 | 14830 | 
| 12 | 3 sklearn/neural_network/rbm.py | 220 | 252| 274 | 5833 | 17657 | 
| 13 | 4 sklearn/utils/estimator_checks.py | 1275 | 1298| 185 | 6018 | 39205 | 
| 14 | **4 sklearn/naive_bayes.py** | 613 | 627| 142 | 6160 | 39205 | 
| 15 | 4 sklearn/multiclass.py | 84 | 107| 184 | 6344 | 39205 | 
| 16 | 5 sklearn/multioutput.py | 43 | 59| 123 | 6467 | 44714 | 
| 17 | 6 sklearn/utils/multiclass.py | 328 | 399| 602 | 7069 | 48589 | 
| 18 | **6 sklearn/naive_bayes.py** | 427 | 443| 169 | 7238 | 48589 | 
| 19 | 6 sklearn/utils/multiclass.py | 293 | 325| 297 | 7535 | 48589 | 
| 20 | **6 sklearn/naive_bayes.py** | 267 | 307| 324 | 7859 | 48589 | 
| 21 | 7 sklearn/neighbors/base.py | 876 | 930| 455 | 8314 | 56313 | 
| 22 | **7 sklearn/naive_bayes.py** | 711 | 732| 227 | 8541 | 56313 | 
| 23 | 7 sklearn/multiclass.py | 479 | 514| 283 | 8824 | 56313 | 
| 24 | 7 sklearn/multiclass.py | 184 | 218| 357 | 9181 | 56313 | 
| 25 | 8 sklearn/mixture/bayesian_mixture.py | 395 | 411| 169 | 9350 | 63576 | 
| 26 | **8 sklearn/naive_bayes.py** | 469 | 481| 162 | 9512 | 63576 | 
| 27 | 8 sklearn/multiclass.py | 713 | 760| 382 | 9894 | 63576 | 
| 28 | 8 sklearn/neighbors/base.py | 856 | 873| 166 | 10060 | 63576 | 
| 29 | 9 sklearn/neighbors/lof.py | 216 | 275| 546 | 10606 | 68078 | 
| 30 | **9 sklearn/naive_bayes.py** | 933 | 955| 224 | 10830 | 68078 | 
| 31 | 10 sklearn/ensemble/partial_dependence.py | 134 | 161| 311 | 11141 | 71828 | 
| 32 | 10 sklearn/multiclass.py | 276 | 315| 379 | 11520 | 71828 | 
| 33 | 11 sklearn/linear_model/omp.py | 619 | 676| 534 | 12054 | 79916 | 
| 34 | 12 examples/ensemble/plot_partial_dependence.py | 61 | 115| 496 | 12550 | 80959 | 
| 35 | 12 sklearn/multioutput.py | 62 | 121| 485 | 13035 | 80959 | 
| 36 | 12 sklearn/ensemble/partial_dependence.py | 70 | 132| 717 | 13752 | 80959 | 
| 37 | 13 sklearn/cross_decomposition/pls_.py | 235 | 287| 566 | 14318 | 89178 | 
| 38 | 13 sklearn/multiclass.py | 110 | 130| 137 | 14455 | 89178 | 
| 39 | 14 sklearn/linear_model/base.py | 491 | 547| 615 | 15070 | 93788 | 
| 40 | 15 sklearn/neighbors/regression.py | 125 | 178| 452 | 15522 | 96561 | 
| 41 | 15 sklearn/mixture/bayesian_mixture.py | 413 | 452| 419 | 15941 | 96561 | 
| **-> 42 <-** | **15 sklearn/naive_bayes.py** | 446 | 467| 182 | 16123 | 96561 | 
| 43 | 16 sklearn/impute.py | 661 | 700| 499 | 16622 | 107223 | 
| 44 | 16 examples/ensemble/plot_partial_dependence.py | 1 | 58| 546 | 17168 | 107223 | 
| 45 | 17 examples/mixture/plot_concentration_prior.py | 1 | 40| 331 | 17499 | 108681 | 
| 46 | 17 examples/mixture/plot_concentration_prior.py | 87 | 137| 571 | 18070 | 108681 | 
| 47 | 18 sklearn/discriminant_analysis.py | 640 | 704| 607 | 18677 | 115456 | 
| 48 | 19 sklearn/neural_network/multilayer_perceptron.py | 1009 | 1037| 228 | 18905 | 127013 | 
| 49 | 20 sklearn/linear_model/bayes.py | 357 | 471| 1027 | 19932 | 132617 | 
| 50 | 21 sklearn/preprocessing/data.py | 677 | 741| 568 | 20500 | 157207 | 
| 51 | **21 sklearn/naive_bayes.py** | 1 | 35| 204 | 20704 | 157207 | 
| 52 | 22 sklearn/cluster/birch.py | 538 | 551| 140 | 20844 | 162491 | 
| 53 | **22 sklearn/naive_bayes.py** | 809 | 837| 304 | 21148 | 162491 | 
| 54 | 22 sklearn/neighbors/regression.py | 287 | 334| 353 | 21501 | 162491 | 
| 55 | 22 sklearn/linear_model/omp.py | 857 | 909| 511 | 22012 | 162491 | 
| 56 | **22 sklearn/naive_bayes.py** | 735 | 807| 690 | 22702 | 162491 | 
| 57 | 22 sklearn/utils/estimator_checks.py | 1154 | 1222| 601 | 23303 | 162491 | 
| 58 | 23 sklearn/mixture/base.py | 194 | 276| 684 | 23987 | 166360 | 
| 59 | 24 sklearn/linear_model/stochastic_gradient.py | 7 | 43| 346 | 24333 | 180848 | 


### Hint

```
I can reproduce the bug. Thanks for reporting. I will work on the fix.
```

## Patch

```diff
diff --git a/sklearn/naive_bayes.py b/sklearn/naive_bayes.py
--- a/sklearn/naive_bayes.py
+++ b/sklearn/naive_bayes.py
@@ -460,8 +460,14 @@ def _update_class_log_prior(self, class_prior=None):
                                  " classes.")
             self.class_log_prior_ = np.log(class_prior)
         elif self.fit_prior:
+            with warnings.catch_warnings():
+                # silence the warning when count is 0 because class was not yet
+                # observed
+                warnings.simplefilter("ignore", RuntimeWarning)
+                log_class_count = np.log(self.class_count_)
+
             # empirical prior, with sample_weight taken into account
-            self.class_log_prior_ = (np.log(self.class_count_) -
+            self.class_log_prior_ = (log_class_count -
                                      np.log(self.class_count_.sum()))
         else:
             self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_naive_bayes.py b/sklearn/tests/test_naive_bayes.py
--- a/sklearn/tests/test_naive_bayes.py
+++ b/sklearn/tests/test_naive_bayes.py
@@ -18,6 +18,7 @@
 from sklearn.utils.testing import assert_raise_message
 from sklearn.utils.testing import assert_greater
 from sklearn.utils.testing import assert_warns
+from sklearn.utils.testing import assert_no_warnings
 
 from sklearn.naive_bayes import GaussianNB, BernoulliNB
 from sklearn.naive_bayes import MultinomialNB, ComplementNB
@@ -244,6 +245,33 @@ def check_partial_fit(cls):
     assert_array_equal(clf1.feature_count_, clf3.feature_count_)
 
 
+def test_mnb_prior_unobserved_targets():
+    # test smoothing of prior for yet unobserved targets
+
+    # Create toy training data
+    X = np.array([[0, 1], [1, 0]])
+    y = np.array([0, 1])
+
+    clf = MultinomialNB()
+
+    assert_no_warnings(
+        clf.partial_fit, X, y, classes=[0, 1, 2]
+    )
+
+    assert clf.predict([[0, 1]]) == 0
+    assert clf.predict([[1, 0]]) == 1
+    assert clf.predict([[1, 1]]) == 0
+
+    # add a training example with previously unobserved class
+    assert_no_warnings(
+        clf.partial_fit, [[1, 1]], [2]
+    )
+
+    assert clf.predict([[0, 1]]) == 0
+    assert clf.predict([[1, 0]]) == 1
+    assert clf.predict([[1, 1]]) == 2
+
+
 @pytest.mark.parametrize("cls", [MultinomialNB, BernoulliNB])
 def test_discretenb_partial_fit(cls):
     check_partial_fit(cls)

```


## Code snippets

### 1 - sklearn/multiclass.py:

Start line: 220, End line: 274

```python
class OneVsRestClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin,
                          MultiOutputMixin):

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
            for estimator, column in zip(self.estimators_, columns))

        return self
```
### 2 - sklearn/multiclass.py:

Start line: 516, End line: 566

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
            self.estimators_ = [clone(self.estimator) for _ in
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
                for estimator, (i, j) in zip(self.estimators_,
                                              (combinations)))

        self.pairwise_indices_ = None

        return self
```
### 3 - sklearn/naive_bayes.py:

Start line: 399, End line: 425

```python
class GaussianNB(BaseNB):

    def _partial_fit(self, X, y, classes=None, _refit=False,
                     sample_weight=None):
        # ... other code

        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
                X_i, sw_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self
```
### 4 - sklearn/naive_bayes.py:

Start line: 309, End line: 397

```python
class GaussianNB(BaseNB):

    def _partial_fit(self, X, y, classes=None, _refit=False,
                     sample_weight=None):
        """Actual implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        _refit : bool, optional (default=False)
            If true, act as though this were the first time we called
            _partial_fit (ie, throw away any past fitting and start over).

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()

        if _refit:
            self.classes_ = None

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = np.zeros((n_classes, n_features))
            self.sigma_ = np.zeros((n_classes, n_features))

            self.class_count_ = np.zeros(n_classes, dtype=np.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = np.asarray(self.priors)
                # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of'
                                     ' classes.')
                # Check that the sum is 1
                if not np.isclose(priors.sum(), 1.0):
                    raise ValueError('The sum of the priors should be 1.')
                # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = np.zeros(len(self.classes_),
                                             dtype=np.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.sigma_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = np.unique(y)
        unique_y_in_classes = np.in1d(unique_y, classes)

        if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the "
                             "initial classes %s" %
                             (unique_y[~unique_y_in_classes], classes))
        # ... other code
```
### 5 - sklearn/naive_bayes.py:

Start line: 630, End line: 709

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
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
```
### 6 - sklearn/naive_bayes.py:

Start line: 106, End line: 165

```python
class GaussianNB(BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Can perform online updates to model parameters via `partial_fit` method.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

    Parameters
    ----------
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class

    epsilon_ : float
        absolute additive value to variances

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
```
### 7 - sklearn/naive_bayes.py:

Start line: 840, End line: 909

```python
class BernoulliNB(BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Read more in the :ref:`User Guide <bernoulli_naive_bayes>`.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    binarize : float or None, optional (default=0.0)
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size=[n_classes,], optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_log_prior_ : array, shape = [n_classes]
        Log probability of each class (smoothed).

    feature_log_prob_ : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    class_count_ : array, shape = [n_classes]
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape = [n_classes, n_features]
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]

    References
    ----------

    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """
```
### 8 - sklearn/naive_bayes.py:

Start line: 911, End line: 931

```python
class BernoulliNB(BaseDiscreteNB):

    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))
```
### 9 - sklearn/naive_bayes.py:

Start line: 563, End line: 611

```python
class BaseDiscreteNB(BaseNB):

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self
```
### 10 - sklearn/naive_bayes.py:

Start line: 483, End line: 561

```python
class BaseDiscreteNB(BaseNB):

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        classes : array-like, shape = [n_classes] (default=None)
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like, shape = [n_samples] (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)
        elif n_features != self.coef_.shape[1]:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (n_features, self.coef_.shape[-1]))

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self
```
### 11 - sklearn/naive_bayes.py:

Start line: 167, End line: 191

```python
class GaussianNB(BaseNB):

    def fit(self, X, y, sample_weight=None):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        return self._partial_fit(X, y, np.unique(y), _refit=True,
                                 sample_weight=sample_weight)
```
### 14 - sklearn/naive_bayes.py:

Start line: 613, End line: 627

```python
class BaseDiscreteNB(BaseNB):

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)

    def _more_tags(self):
        return {'poor_score': True}
```
### 18 - sklearn/naive_bayes.py:

Start line: 427, End line: 443

```python
class GaussianNB(BaseNB):

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


_ALPHA_MIN = 1e-10
```
### 20 - sklearn/naive_bayes.py:

Start line: 267, End line: 307

```python
class GaussianNB(BaseNB):

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17

        Returns
        -------
        self : object
        """
        return self._partial_fit(X, y, classes, _refit=False,
                                 sample_weight=sample_weight)
```
### 22 - sklearn/naive_bayes.py:

Start line: 711, End line: 732

```python
class MultinomialNB(BaseDiscreteNB):

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)
```
### 26 - sklearn/naive_bayes.py:

Start line: 469, End line: 481

```python
class BaseDiscreteNB(BaseNB):

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError('Smoothing parameter alpha = %.1e. '
                             'alpha should be > 0.' % np.min(self.alpha))
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.feature_count_.shape[1]:
                raise ValueError("alpha should be a scalar or a numpy array "
                                 "with shape [n_features]")
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn('alpha too small will result in numeric errors, '
                          'setting alpha = %.1e' % _ALPHA_MIN)
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha
```
### 30 - sklearn/naive_bayes.py:

Start line: 933, End line: 955

```python
class BernoulliNB(BaseDiscreteNB):

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll
```
### 42 - sklearn/naive_bayes.py:

Start line: 446, End line: 467

```python
class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_) -
                                     np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
```
### 51 - sklearn/naive_bayes.py:

Start line: 1, End line: 35

```python
# -*- coding: utf-8 -*-

"""
The :mod:`sklearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""
import warnings

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse

from .base import BaseEstimator, ClassifierMixin
from .preprocessing import binarize
from .preprocessing import LabelBinarizer
from .preprocessing import label_binarize
from .utils import check_X_y, check_array, check_consistent_length
from .utils.extmath import safe_sparse_dot
from .utils.fixes import logsumexp
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import check_is_fitted

__all__ = ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB']
```
### 53 - sklearn/naive_bayes.py:

Start line: 809, End line: 837

```python
class ComplementNB(BaseDiscreteNB):

    def _count(self, X, Y):
        """Count feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and compute the weights."""
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        # BaseNB.predict uses argmax, but ComplementNB operates with argmin.
        if self.norm:
            summed = logged.sum(axis=1, keepdims=True)
            feature_log_prob = logged / summed
        else:
            feature_log_prob = -logged
        self.feature_log_prob_ = feature_log_prob

    def _joint_log_likelihood(self, X):
        """Calculate the class scores for the samples in X."""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse="csr")
        jll = safe_sparse_dot(X, self.feature_log_prob_.T)
        if len(self.classes_) == 1:
            jll += self.class_log_prior_
        return jll
```
### 56 - sklearn/naive_bayes.py:

Start line: 735, End line: 807

```python
class ComplementNB(BaseDiscreteNB):
    """The Complement Naive Bayes classifier described in Rennie et al. (2003).

    The Complement Naive Bayes classifier was designed to correct the "severe
    assumptions" made by the standard Multinomial Naive Bayes classifier. It is
    particularly suited for imbalanced data sets.

    Read more in the :ref:`User Guide <complement_naive_bayes>`.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

    fit_prior : boolean, optional (default=True)
        Only used in edge case with a single class in the training set.

    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. Not used.

    norm : boolean, optional (default=False)
        Whether or not a second normalization of the weights is performed. The
        default behavior mirrors the implementations found in Mahout and Weka,
        which do not follow the full algorithm described in Table 9 of the
        paper.

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class. Only used in edge
        case with a single class in the training set.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical weights for class complements.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature) during fitting.
        This value is weighted by the sample weight when provided.

    feature_all_ : array, shape (n_features,)
        Number of samples encountered for each feature during fitting. This
        value is weighted by the sample weight when provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import ComplementNB
    >>> clf = ComplementNB()
    >>> clf.fit(X, y)
    ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
    >>> print(clf.predict(X[2:3]))
    [3]

    References
    ----------
    Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
    Tackling the poor assumptions of naive bayes text classifiers. In ICML
    (Vol. 3, pp. 616-623).
    https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None,
                 norm=False):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm
```
