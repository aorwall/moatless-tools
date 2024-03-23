0 - /tmp/repos/scikit-learn/sklearn/utils/multiclass.py
```python
def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking OvR decision function from OvO.

    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.

    Parameters
    ----------
    predictions : array-like, shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.

    confidences : array-like, shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.

    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    max_confidences = sum_of_confidences.max()
    min_confidences = sum_of_confidences.min()

    if max_confidences == min_confidences:
        return votes

    # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    eps = np.finfo(sum_of_confidences.dtype).eps
    max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
    scale = (0.5 - eps) / max_abs_confidence
    return votes + sum_of_confidences * scale
```
**1 - /tmp/repos/scikit-learn/examples/svm/plot_svm_tie_breaking.py**:
```python
"""
=========================================================
SVM Tie Breaking Example
=========================================================
Tie breaking is costly if ``decision_function_shape='ovr'``, and therefore it
is not enabled by default. This example illustrates the effect of the
``break_ties`` parameter for a multiclass classification problem and
``decision_function_shape='ovr'``.

The two plots differ only in the area in the middle where the classes are
tied. If ``break_ties=False``, all input in that area would be classified as
one class, whereas if ``break_ties=True``, the tie-breaking mechanism will
create a non-convex decision boundary in that area.
"""
print(__doc__)


# Code source: Andreas Mueller, Adrin Jalali
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=27)

fig, sub = plt.subplots(2, 1, figsize=(5, 8))
titles = ("break_ties = False",
          "break_ties = True")

for break_ties, title, ax in zip((False, True), titles, sub.flatten()):

    svm = SVC(kernel="linear", C=1, break_ties=break_ties,
              decision_function_shape='ovr').fit(X, y)

    xlim = [X[:, 0].min(), X[:, 0].max()]
    ylim = [X[:, 1].min(), X[:, 1].max()]

    xs = np.linspace(xlim[0], xlim[1], 1000)
    ys = np.linspace(ylim[0], ylim[1], 1000)
    xx, yy = np.meshgrid(xs, ys)

    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    colors = [plt.cm.Accent(i) for i in [0, 4, 7]]

    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent")
    classes = [(0, 1), (0, 2), (1, 2)]
    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
    ax.imshow(-pred.reshape(xx.shape), cmap="Accent", alpha=.2,
              extent=(xlim[0], xlim[1], ylim[1], ylim[0]))

    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
        line2 = -(line * coef[1] + intercept) / coef[0]
        ax.plot(line2, line, "-", c=colors[col[0]])
        ax.plot(line2, line, "--", c=colors[col[1]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.show()
```
2 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
3 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
4 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
5 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
6 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
7 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
8 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
# ...
```
**9 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def _predict_proba(self, X):
        X = self._validate_for_predict(X)
        if self.probA_.size == 0 or self.probB_.size == 0:
            raise NotFittedError("predict_proba is not available when fitted "
                                 "with probability=False")
        pred_proba = (self._sparse_predict_proba
                      if self._sparse else self._dense_predict_proba)
        return pred_proba(X)
```
10 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python
class OneVsOneClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):


    def decision_function(self, X):
        """Decision function for the OneVsOneClassifier.

        The decision values for the samples are computed by adding the
        normalized sum of pair-wise classification confidence levels to the
        votes in order to disambiguate between the decision values when the
        votes for all the classes are equal leading to a tie.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        Y : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([est.predict(Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.vstack([_predict_binary(est, Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        Y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        if self.n_classes_ == 2:
            return Y[:, 1]
        return Y
```
11 - /tmp/repos/scikit-learn/examples/ensemble/plot_voting_decision_regions.py
```python
"""
==================================================
Plot the decision boundaries of a VotingClassifier
==================================================

Plot the decision boundaries of a `VotingClassifier` for
two features of the Iris dataset.

Plot the class probabilities of the first sample in a toy dataset
predicted by three different classifiers and averaged by the
`VotingClassifier`.

First, three exemplary classifiers are initialized (`DecisionTreeClassifier`,
`KNeighborsClassifier`, and `SVC`) and used to initialize a
soft-voting `VotingClassifier` with weights `[2, 1, 2]`, which means that
the predicted probabilities of the `DecisionTreeClassifier` and `SVC`
count 5 times as much as the weights of the `KNeighborsClassifier` classifier
when the averaged probability is calculated.

"""
print(__doc__)

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
```
**12 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):

    def _check_proba(self):
        if not self.probability:
            raise AttributeError("predict_proba is not available when "
                                 " probability=False")
        if self._impl not in ('c_svc', 'nu_svc'):
            raise AttributeError("predict_proba only implemented for SVC"
                                 " and NuSVC")
```
**13 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def decision_function(self, X):
        """Distance of the samples X to the separating hyperplane.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes)
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec
```
**14 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def _dense_predict_proba(self, X):
        X = self._compute_kernel(X)

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        svm_type = LIBSVM_IMPL.index(self._impl)
        pprob = libsvm.predict_proba(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_,
            svm_type=svm_type, kernel=kernel, degree=self.degree,
            cache_size=self.cache_size, coef0=self.coef0, gamma=self._gamma)

        return pprob
```
**15 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def _sparse_predict_proba(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_predict_proba(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            self.C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)
```
**16 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def _predict_log_proba(self, X):
        return np.log(self.predict_proba(X))
```
17 - /tmp/repos/scikit-learn/sklearn/multioutput.py
```python
class ClassifierChain(_BaseChain, ClassifierMixin, MetaEstimatorMixin):


    @if_delegate_has_method('base_estimator')
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Y_decision : array-like, shape (n_samples, n_classes )
            Returns the decision function of the sample for each model
            in the chain.
        """
        Y_decision_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_decision_chain[:, chain_idx] = estimator.decision_function(X_aug)
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_decision = Y_decision_chain[:, inv_order]

        return Y_decision
```
18 - /tmp/repos/scikit-learn/benchmarks/bench_covertype.py
```python
# ..."""
===========================
Covertype dataset benchmark
===========================

Benchmark stochastic gradient descent (SGD), Liblinear, and Naive Bayes, CART
(decision tree), RandomForest and Extra-Trees on the forest covertype dataset
of Blackard, Jock, and Dean [1]. The dataset comprises 581,012 samples. It is
low dimensional with 54 features and a sparsity of approx. 23%. Here, we
consider the task of predicting class 1 (spruce/fir). The classification
performance of SGD is competitive with Liblinear while being two orders of
magnitude faster to train::

    [..]
    Classification performance:
    ===========================
    Classifier   train-time test-time error-rate
    --------------------------------------------
    liblinear     15.9744s    0.0705s     0.2305
    GaussianNB    3.0666s     0.3884s     0.4841
    SGD           1.0558s     0.1152s     0.2300
    CART          79.4296s    0.0523s     0.0469
    RandomForest  1190.1620s  0.5881s     0.0243
    ExtraTrees    640.3194s   0.6495s     0.0198

The same task has been used in a number of papers including:

 * `"SVM Optimization: Inverse Dependence on Training Set Size"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.139.2112>`_
   S. Shalev-Shwartz, N. Srebro - In Proceedings of ICML '08.

 * `"Pegasos: Primal estimated sub-gradient solver for svm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513>`_
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.

 * `"Training Linear SVMs in Linear Time"
   <www.cs.cornell.edu/People/tj/publications/joachims_06a.pdf>`_
   T. Joachims - In SIGKDD '06

[1] http://archive.ics.uci.edu/ml/datasets/Covertype


```
19 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error,
                               greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)

neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               max_error=max_error_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score),
                     ('jaccard', jaccard_score)]:
    SCORERS[name] = make_scorer(metric, average='binary')
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None,
                                              average=average)

```
20 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator)):


    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X
```
**21 - /tmp/repos/scikit-learn/sklearn/svm/base.py**:
```python
class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):


    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        y = super(BaseSVC, self).predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp))
```
22 - /tmp/repos/scikit-learn/examples/calibration/plot_compare_calibration.py
```python
# ..."""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of the predict_proba method can be directly interpreted as a confidence level.
For instance a well calibrated (binary) classifier should classify the samples
such that among the samples to which it gave a predict_proba value close to
0.8, approx. 80% actually belong to the positive class.

LogisticRegression returns well calibrated predictions as it directly
optimizes log-loss. In contrast, the other methods return biased probabilities,
with different biases per method:

* GaussianNaiveBayes tends to push probabilities to 0 or 1 (note the counts in
  the histograms). This is mainly because it makes the assumption that features
  are conditionally independent given the class, which is not the case in this
  dataset which contains 2 redundant features.

* RandomForestClassifier shows the opposite behavior: the histograms show
  peaks at approx. 0.2 and 0.9 probability, while probabilities close to 0 or 1
  are very rare. An explanation for this is given by Niculescu-Mizil and Caruana
  [1]: "Methods such as bagging and random forests that average predictions from
  a base set of models can have difficulty making predictions near 0 and 1
  because variance in the underlying base models will bias predictions that
  should be near zero or one away from these values. Because predictions are
  restricted to the interval [0,1], errors caused by variance tend to be one-
  sided near zero and one. For example, if a model should predict p = 0 for a
  case, the only way bagging can achieve this is if all bagged trees predict
  zero. If we add noise to the trees that bagging is averaging over, this noise
  will cause some trees to predict values larger than 0 for this case, thus
  moving the average prediction of the bagged ensemble away from 0. We observe
  this effect most strongly with random forests because the base-level trees
  trained with random forests have relatively high variance due to feature
  subsetting." As a result, the calibration curve shows a characteristic
  sigmoid shape, indicating that the classifier could trust its "intuition"
  more and return probabilities closer to 0 or 1 typically.

* Support Vector Classification (SVC) shows an even more sigmoid curve as
  the  RandomForestClassifier, which is typical for maximum-margin methods
  (compare Niculescu-Mizil and Caruana [1]), which focus on hard samples
  that are close to the decision boundary (the support vectors).

.. topic:: References:

    .. [1] Predicting Good Probabilities with Supervised Learning,
          A. Niculescu-Mizil & R. Caruana, ICML 2005

```
**23 - /tmp/repos/scikit-learn/sklearn/svm/classes.py**:
```python
class LinearSVC(BaseEstimator, LinearClassifierMixin,
                SparseCoefMixin):
    """Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to ``coef_``
        vectors that are sparse.

    loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    multi_class : string, 'ovr' or 'crammer_singer' (default='ovr')
        Determines the multi-class strategy if `y` contains more than
        two classes.
        ``"ovr"`` trains n_classes one-vs-rest classifiers, while
        ``"crammer_singer"`` optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from a theoretical perspective
        as it is consistent, it is seldom used in practice as it rarely leads
        to better accuracy and is more expensive to compute.
        If ``"crammer_singer"`` is chosen, the options loss, penalty and dual
        will be ignored.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    intercept_scaling : float, optional (default=1)
        When self.fit_intercept is True, instance vector x becomes
        ``[x, self.intercept_scaling]``,
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to ``class_weight[i]*C`` for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    verbose : int, (default=0)
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data for the dual coordinate descent (if ``dual=True``). When
        ``dual=False`` the underlying implementation of :class:`LinearSVC`
        is not random and ``random_state`` has no effect on the results. If
        int, random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`.

    max_iter : int, (default=1000)
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        ``coef_`` is a readonly property derived from ``raw_coef_`` that
        follows the internal memory layout of liblinear.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = LinearSVC(random_state=0)
    >>> clf.fit(X, y)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
         verbose=0)
    >>> print(clf.coef_)
    [[ 0.08551385  0.39414796  0.49847831  0.37513797]]
    >>> print(clf.intercept_)
    [ 0.28418066]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon
    to have slightly different results for the same input data. If
    that happens, try with a smaller ``tol`` parameter.

    The underlying implementation, liblinear, uses a sparse internal
    representation for the data that will incur a memory copy.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------
    `LIBLINEAR: A Library for Large Linear Classification
    <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

    See also
    --------
    SVC
        Implementation of Support Vector Machine classifier using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

        Furthermore SVC multi-class mode is implemented using one
        vs one scheme while LinearSVC uses one vs the rest. It is
        possible to implement one vs the rest with SVC by using the
        :class:`sklearn.multiclass.OneVsRestClassifier` wrapper.

        Finally SVC can fit dense data without memory copy if the input
        is C-contiguous. Sparse data will still incur memory copy though.

    sklearn.linear_model.SGDClassifier
        SGDClassifier can optimize the same cost function as LinearSVC
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.

    
```
24 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        super(_PredictScorer, self).__call__(estimator, X, y_true,
                                             sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)


class _ProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        super(_ProbaScorer, self).__call__(clf, X, y,
                                           sample_weight=sample_weight)
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            y_pred = y_pred[:, 1]
        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        
```
25 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification produce an array of shape
            [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C",  accept_sparse='csr')
        score = self._decision_function(X)
        if score.shape[1] == 1:
            return score.ravel()
        return score

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape (n_samples, k)
            The decision function of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        for dec in self._staged_decision_function(X):
            # no yield from in Python2.X
            yield dec

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        score = self.decision_function(X)
        decisions = self.loss_._score_to_decision(score)
        return self.classes_.take(decisions, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape (n_samples,)
            The predicted value of the input samples.
        """
        for score in self._staged_decision_function(X):
            decisions = self.loss_._score_to_decision(score)
            yield self.classes_.take(decisions, axis=0)

    
```
**26 - /tmp/repos/scikit-learn/sklearn/svm/classes.py**:
```python
class OneClassSVM(BaseLibSVM, OutlierMixin):


    def decision_function(self, X):
        """Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an outlier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        dec : array-like, shape (n_samples,)
            Returns the decision function of the samples.
        """
        dec = self._decision_function(X).ravel()
        return dec
```
27 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(n_samples=100, random_state=0, n_features=4,
                      centers=centers, cluster_std=1.0, shuffle=True)
    X_test = np.random.randn(20, 2) + 4
    estimator = clone(estimator_orig)

    if (hasattr(estimator, "decision_function") and
            hasattr(estimator, "predict_proba")):

        estimator.fit(X, y)
        a = estimator.predict_proba(X_test)[:, 1]
        b = estimator.decision_function(X_test)
        assert_array_equal(rankdata(a), rankdata(b))
```
28 - /tmp/repos/scikit-learn/benchmarks/bench_saga.py
```python
def _predict_proba(lr, X):
    pred = safe_sparse_dot(X, lr.coef_.T)
    if hasattr(lr, "intercept_"):
        pred += lr.intercept_
    return softmax(pred)
```
29 - /tmp/repos/scikit-learn/examples/classification/plot_classifier_comparison.py
```python
# ...#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets

```
30 - /tmp/repos/scikit-learn/examples/semi_supervised/plot_label_propagation_versus_svm_iris.py
```python
"""
=====================================================================
Decision boundary of label propagation versus SVM on the Iris dataset
=====================================================================

Comparison for decision boundary generated on iris dataset
between Label Propagation and SVM.

This demonstrates Label Propagation learning a good boundary
even with a small amount of labeled data.

"""
print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation

rng = np.random.RandomState(0)

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# step size in the mesh
h = .02

y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
ls30 = (label_propagation.LabelSpreading().fit(X, y_30),
        y_30)
ls50 = (label_propagation.LabelSpreading().fit(X, y_50),
        y_50)
ls100 = (label_propagation.LabelSpreading().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf').fit(X, y), y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Label Spreading 30% data',
          'Label Spreading 50% data',
          'Label Spreading 100% data',
          'SVC with rbf kernel']

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')

    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()
```
31 - /tmp/repos/scikit-learn/examples/svm/plot_svm_scale_c.py
```python
# ..."""
==============================================
Scaling the regularization parameter for SVCs
==============================================

The following example illustrates the effect of scaling the
regularization parameter when using :ref:`svm` for
:ref:`classification <svm_classification>`.
For SVC classification, we are interested in a risk minimization for the
equation:


.. math::

    C \sum_{i=1, n} \mathcal{L} (f(x_i), y_i) + \Omega (w)

where

    - :math:`C` is used to set the amount of regularization
    - :math:`\mathcal{L}` is a `loss` function of our samples
      and our model parameters.
    - :math:`\Omega` is a `penalty` function of our model parameters

If we consider the loss function to be the individual error per
sample, then the data-fit term, or the sum of the error for each sample, will
increase as we add more samples. The penalization term, however, will not
increase.

When using, for example, :ref:`cross validation <cross_validation>`, to
set the amount of regularization with `C`, there will be a
different amount of samples between the main problem and the smaller problems
within the folds of the cross validation.

Since our loss function is dependent on the amount of samples, the latter
will influence the selected value of `C`.
The question that arises is `How do we optimally adjust C to
account for the different amount of training samples?`

The figures below are used to illustrate the effect of scaling our
`C` to compensate for the change in the number of samples, in the
case of using an `l1` penalty, as well as the `l2` penalty.

l1-penalty case
-----------------
In the `l1` case, theory says that prediction consistency
(i.e. that under given hypothesis, the estimator
learned predicts as well as a model knowing the true distribution)
is not possible because of the bias of the `l1`. It does say, however,
that model consistency, in terms of finding the right set of non-zero
parameters as well as their signs, can be achieved by scaling
`C1`.

l2-penalty case
-----------------
The theory says that in order to achieve prediction consistency, the
penalty parameter should be kept constant
as the number of samples grow.

Simulations
------------

The two figures below plot the values of `C` on the `x-axis` and the
corresponding cross-validation scores on the `y-axis`, for several different
fractions of a generated data-set.

In the `l1` penalty case, the cross-validation-error correlates best with
the test-error, when scaling our `C` with the number of samples, `n`,
which can be seen in the first figure.

For the `l2` penalty case, the best result comes from the case where `C`
is not scaled.

.. topic:: Note:

    Two separate datasets are used for the two different plots. The reason
    behind this is the `l1` case works better on sparse data, while `l2`
    is better suited to the 
```
32 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        super(_PredictScorer, self).__call__(estimator, X, y_true,
                                             sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)


class _ProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        super(_ProbaScorer, self).__call__(clf, X, y,
                                           sample_weight=sample_weight)
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            y_pred = y_pred[:, 1]
        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        
```
33 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
: boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
deprecation_msg = ('Scoring method mean_squared_error was renamed to '
                   'neg_mean_squared_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_squared_error_scorer = make_scorer(mean_squared_error,
                                        greater_is_better=False)
mean_squared_error_scorer._deprecation_msg = deprecation_msg
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)
deprecation_msg = ('Scoring method mean_absolute_error was renamed to '
                   'neg_mean_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                         greater_is_better=False)
mean_absolute_error_scorer._deprecation_msg = deprecation_msg
neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)
deprecation_msg = ('Scoring method median_absolute_error was renamed to '
                   'neg_median_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
median_absolute_error_scorer = make_scorer(median_absolute_error,
                                           greater_is_better=False)
median_absolute_error_scorer._deprecation_msg = deprecation_msg


# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
deprecation_msg = ('Scoring method log_loss was renamed to '
                   'neg_log_loss in version 0.18 and will be removed in 0.20.')
log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                              needs_proba=True)
log_loss_scorer._deprecation_msg = deprecation_msg
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               median_absolute_error=median_absolute_error_scorer,
               mean_absolute_error=mean_absolute_error_scorer,
               mean_squared_error=mean_squared_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               log_loss=log_loss_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None,
                                              average=average)

```
34 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
: boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
deprecation_msg = ('Scoring method mean_squared_error was renamed to '
                   'neg_mean_squared_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_squared_error_scorer = make_scorer(mean_squared_error,
                                        greater_is_better=False)
mean_squared_error_scorer._deprecation_msg = deprecation_msg
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)
deprecation_msg = ('Scoring method mean_absolute_error was renamed to '
                   'neg_mean_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                         greater_is_better=False)
mean_absolute_error_scorer._deprecation_msg = deprecation_msg
neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)
deprecation_msg = ('Scoring method median_absolute_error was renamed to '
                   'neg_median_absolute_error in version 0.18 and will '
                   'be removed in 0.20.')
median_absolute_error_scorer = make_scorer(median_absolute_error,
                                           greater_is_better=False)
median_absolute_error_scorer._deprecation_msg = deprecation_msg


# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
deprecation_msg = ('Scoring method log_loss was renamed to '
                   'neg_log_loss in version 0.18 and will be removed in 0.20.')
log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                              needs_proba=True)
log_loss_scorer._deprecation_msg = deprecation_msg
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               median_absolute_error=median_absolute_error_scorer,
               mean_absolute_error=mean_absolute_error_scorer,
               mean_squared_error=mean_squared_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               log_loss=log_loss_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None,
                                              average=average)

```
35 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            `classes_`. Regression and binary classification produce an
            array of shape [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self.loss_._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self.loss_._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    
```
36 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : array, shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            `classes_`. Regression and binary classification produce an
            array of shape [n_samples].
        """
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : generator of array, shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute `classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self.loss_._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : generator of array of shape (n_samples,)
            The predicted value of the input samples.
        """
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self.loss_._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    
```
37 - /tmp/repos/scikit-learn/examples/svm/plot_svm_nonlinear.py
```python
"""
==============
Non-linear SVM
==============

Perform binary classification using non-linear SVC
with RBF kernel. The target to predict is a XOR of the
inputs.

The color map illustrates the decision function learned by the SVC.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# fit the model
clf = svm.NuSVC()
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()
```
38 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = estimator.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)


class _ProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            else:
                raise ValueError('got predict_proba of shape {},'
                                 ' but need classifier with two'
                                 ' classes for {} scoring'.format(
                                     y_pred.shape, self._score_func.__name__))
        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        
```
39 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    
```
40 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    
```
41 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
: boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)

neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None,
                                              average=average)

```
42 - /tmp/repos/scikit-learn/sklearn/metrics/scorer.py
```python
X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = estimator.predict(X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)


class _ProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            y_pred = y_pred[:, 1]
        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        
```
43 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_regressors_no_decision_function(name, regressor_orig):
    # checks whether regressors have decision_function or predict_proba
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 4))
    regressor = clone(regressor_orig)
    y = multioutput_estimator_convert_y_2d(regressor, X[:, 0])

    if hasattr(regressor, "n_components"):
        # FIXME CCA, PLS is not robust to rank 1 effects
        regressor.n_components = 1

    regressor.fit(X, y)
    funcs = ["decision_function", "predict_proba", "predict_log_proba"]
    for func_name in funcs:
        func = getattr(regressor, func_name, None)
        if func is None:
            # doesn't have function
            continue
        # has function. Should raise deprecation warning
        msg = func_name
        assert_warns_message(DeprecationWarning, msg, func, X)
```
44 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)
    if name == 'BernoulliNB':
        X = X > X.mean()
    set_random_state(classifier)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        n_samples, n_features = X.shape
        assert isinstance(decision, np.ndarray)
        if len(classes) == 2:
            dec_pred = (decision.ravel() > 0).astype(np.int)
            dec_exp = classifier.classes_[dec_pred]
            assert_array_equal(dec_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, dec_exp)),
                                ", ".join(map(str, y_pred))))
        elif getattr(classifier, 'decision_function_shape', 'ovr') == 'ovr':
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            assert_array_equal(y_exp, y_pred,
                               err_msg="decision_function does not match "
                               "classifier for %r: expected '%s', got '%s'" %
                               (classifier, ", ".join(map(str, y_exp)),
                                ", ".join(map(str, y_pred))))

    # training set performance
    if name != "ComplementNB":
        # This is a pathological data set for ComplementNB.
        # For some specific cases 'ComplementNB' predicts less classes
        # than expected
        assert_array_equal(np.unique(y), np.unique(y_pred))
    assert_array_equal(classes, classifier.classes_,
                       err_msg="Unexpected classes_ attribute for %r: "
                       "expected '%s', got '%s'" %
                       (classifier, ", ".join(map(str, classes)),
                        ", ".join(map(str, classifier.classes_))))
```
45 - /tmp/repos/scikit-learn/examples/svm/plot_separating_hyperplane_unbalanced.py
```python
"""
=================================================
SVM: Separating hyperplane for unbalanced classes
=================================================

Find the optimal separating hyperplane using an SVC for classes that
are unbalanced.

We first find the separating plane with a plain SVC and then plot
(dashed) the separating hyperplane with automatically correction for
unbalanced classes.

.. currentmodule:: sklearn.linear_model

.. note::

    This example will also work by replacing ``SVC(kernel="linear")``
    with ``SGDClassifier(loss="hinge")``. Setting the ``loss`` parameter
    of the :class:`SGDClassifier` equal to ``hinge`` will yield behaviour
    such as that of a SVC with a linear kernel.

    For example try instead of the ``SVC``::

        clf = SGDClassifier(n_iter=100, alpha=0.01)

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create two clusters of random points
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

# plot separating hyperplanes and samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.legend()

# plot the decision functions for both classifiers
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()
```
46 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
class BinomialDeviance(ClassificationLossFunction):


    def _score_to_decision(self, score):
        proba = self._score_to_proba(score)
        return np.argmax(proba, axis=1)
```
47 - /tmp/repos/scikit-learn/examples/svm/plot_iris_svc.py
```python
"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets



# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
```
48 - /tmp/repos/scikit-learn/examples/svm/plot_rbf_parameters.py
```python
``C_range`` and
``gamma_range`` steps will increase the resolution of the hyper-parameter heat
map.

'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search

iris = load_iris()
X = iris.data
y = iris.target

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy'
```
49 - /tmp/repos/scikit-learn/sklearn/ensemble/_gb_losses.py
```python
class BinomialDeviance(ClassificationLossFunction):


    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)
```
50 - /tmp/repos/scikit-learn/sklearn/multiclass.py
```python
class OneVsOneClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):


    def predict(self, X):
        """Estimate the best class label for each sample in X.

        This is implemented as ``argmax(decision_function(X), axis=1)`` which
        will return the label of the class with most votes by estimators
        predicting the outcome of a decision for each possible class pair.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        Y = self.decision_function(X)
        if self.n_classes_ == 2:
            return self.classes_[(Y > 0).astype(np.int)]
        return self.classes_[Y.argmax(axis=1)]
```
51 - /tmp/repos/scikit-learn/sklearn/ensemble/voting.py
```python
class VotingClassifier(_BaseVoting, ClassifierMixin):


    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg
```
52 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
# ..."""
=========================================================
Imputing missing values with variants of IterativeImputer
=========================================================

The :class:`sklearn.impute.IterativeImputer` class is very flexible - it can be
used with a variety of estimators to do round-robin regression, treating every
variable as an output in turn.

In this example we compare some estimators for the purpose of missing feature
imputation with :class:`sklearn.impute.IterativeImputer`:

* :class:`~sklearn.linear_model.BayesianRidge`: regularized linear regression
* :class:`~sklearn.tree.DecisionTreeRegressor`: non-linear regression
* :class:`~sklearn.ensemble.ExtraTreesRegressor`: similar to missForest in R
* :class:`~sklearn.neighbors.KNeighborsRegressor`: comparable to other KNN
  imputation approaches

Of particular interest is the ability of
:class:`sklearn.impute.IterativeImputer` to mimic the behavior of missForest, a
popular imputation package for R. In this example, we have chosen to use
:class:`sklearn.ensemble.ExtraTreesRegressor` instead of
:class:`sklearn.ensemble.RandomForestRegressor` (as in missForest) due to its
increased speed.

Note that :class:`sklearn.neighbors.KNeighborsRegressor` is different from KNN
imputation, which learns from samples with missing values by using a distance
metric that accounts for missing values, rather than imputing them.

The goal is to compare different estimators to see which one is best for the
:class:`sklearn.impute.IterativeImputer` when using a
:class:`sklearn.linear_model.BayesianRidge` estimator on the California housing
dataset with a single value randomly removed from each row.

For this particular pattern of missing values we see that
:class:`sklearn.ensemble.ExtraTreesRegressor` and
:class:`sklearn.linear_model.BayesianRidge` give the best results.

```
53 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py
```python
classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        X = pairwise_estimator_convert_X(X, classifier)
        y = multioutput_estimator_convert_y_2d(classifier, y)

        set_random_state(classifier)
        # raises error on malformed input for fit
        if not tags["no_validation"]:
            with assert_raises(
                ValueError,
                msg="The classifier {} does not "
                    "raise an error when incorrect/malformed input "
                    "data for fit is passed. The number of training "
                    "examples is not the same as the number of labels. "
                    "Perhaps use check_X_y in fit.".format(name)):
                classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        if not tags['poor_score']:
            assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)")
        msg = ("The classifier {} does not raise an error when the number of "
               "features in {} is different from the number of features in "
               "fit.")

        if not tags["no_validation"]:
            if _is_pairwise(classifier):
                with assert_raises(ValueError,
                                   msg=msg_pairwise.format(name, "predict")):
                    classifier.predict(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError,
                                   msg=msg.format(name, "predict")):
                    classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if not tags["multioutput_only"]:
                        assert_equal(decision.shape, (n_samples,))
                    else:
                        assert_equal(decision.shape, (n_samples, 1))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not tags["no_validation"]:
                    if _is_pairwise(classifier):
                        with assert_raises(ValueError, msg=msg_pairwise.format(
                                name, "decision_function")):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        with assert_raises(ValueError, msg=msg.format(
                                name, "decision_function")):
                            classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        
```
54 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py
```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):


    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm
```
**55 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py**:
```python
class BaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):


    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)
```
