# scikit-learn__scikit-learn-12557

| **scikit-learn/scikit-learn** | `4de404d46d24805ff48ad255ec3169a5155986f0` |
| ---- | ---- |
| **No of patches** | 4 |
| **All found context length** | 8744 |
| **Any found context length** | 2044 |
| **Avg pos** | 101.25 |
| **Min pos** | 5 |
| **Max pos** | 65 |
| **Top file pos** | 3 |
| **Missing snippets** | 12 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/examples/svm/plot_svm_tie_breaking.py b/examples/svm/plot_svm_tie_breaking.py
new file mode 100644
--- /dev/null
+++ b/examples/svm/plot_svm_tie_breaking.py
@@ -0,0 +1,64 @@
+"""
+=========================================================
+SVM Tie Breaking Example
+=========================================================
+Tie breaking is costly if ``decision_function_shape='ovr'``, and therefore it
+is not enabled by default. This example illustrates the effect of the
+``break_ties`` parameter for a multiclass classification problem and
+``decision_function_shape='ovr'``.
+
+The two plots differ only in the area in the middle where the classes are
+tied. If ``break_ties=False``, all input in that area would be classified as
+one class, whereas if ``break_ties=True``, the tie-breaking mechanism will
+create a non-convex decision boundary in that area.
+"""
+print(__doc__)
+
+
+# Code source: Andreas Mueller, Adrin Jalali
+# License: BSD 3 clause
+
+
+import numpy as np
+import matplotlib.pyplot as plt
+from sklearn.svm import SVC
+from sklearn.datasets import make_blobs
+
+X, y = make_blobs(random_state=27)
+
+fig, sub = plt.subplots(2, 1, figsize=(5, 8))
+titles = ("break_ties = False",
+          "break_ties = True")
+
+for break_ties, title, ax in zip((False, True), titles, sub.flatten()):
+
+    svm = SVC(kernel="linear", C=1, break_ties=break_ties,
+              decision_function_shape='ovr').fit(X, y)
+
+    xlim = [X[:, 0].min(), X[:, 0].max()]
+    ylim = [X[:, 1].min(), X[:, 1].max()]
+
+    xs = np.linspace(xlim[0], xlim[1], 1000)
+    ys = np.linspace(ylim[0], ylim[1], 1000)
+    xx, yy = np.meshgrid(xs, ys)
+
+    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
+
+    colors = [plt.cm.Accent(i) for i in [0, 4, 7]]
+
+    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent")
+    classes = [(0, 1), (0, 2), (1, 2)]
+    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
+    ax.imshow(-pred.reshape(xx.shape), cmap="Accent", alpha=.2,
+              extent=(xlim[0], xlim[1], ylim[1], ylim[0]))
+
+    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
+        line2 = -(line * coef[1] + intercept) / coef[0]
+        ax.plot(line2, line, "-", c=colors[col[0]])
+        ax.plot(line2, line, "--", c=colors[col[1]])
+    ax.set_xlim(xlim)
+    ax.set_ylim(ylim)
+    ax.set_title(title)
+    ax.set_aspect("equal")
+
+plt.show()
diff --git a/sklearn/model_selection/_search.py b/sklearn/model_selection/_search.py
--- a/sklearn/model_selection/_search.py
+++ b/sklearn/model_selection/_search.py
@@ -983,11 +983,13 @@ class GridSearchCV(BaseSearchCV):
     >>> clf.fit(iris.data, iris.target)
     ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
     GridSearchCV(cv=5, error_score=...,
-           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
+           estimator=SVC(C=1.0, break_ties=False, cache_size=...,
+                         class_weight=..., coef0=...,
                          decision_function_shape='ovr', degree=..., gamma=...,
-                         kernel='rbf', max_iter=-1, probability=False,
-                         random_state=None, shrinking=True, tol=...,
-                         verbose=False),
+                         kernel='rbf', max_iter=-1,
+                         probability=False,
+                         random_state=None, shrinking=True,
+                         tol=..., verbose=False),
            iid=..., n_jobs=None,
            param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
            scoring=..., verbose=...)
diff --git a/sklearn/svm/base.py b/sklearn/svm/base.py
--- a/sklearn/svm/base.py
+++ b/sklearn/svm/base.py
@@ -501,8 +501,10 @@ class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):
     @abstractmethod
     def __init__(self, kernel, degree, gamma, coef0, tol, C, nu,
                  shrinking, probability, cache_size, class_weight, verbose,
-                 max_iter, decision_function_shape, random_state):
+                 max_iter, decision_function_shape, random_state,
+                 break_ties):
         self.decision_function_shape = decision_function_shape
+        self.break_ties = break_ties
         super().__init__(
             kernel=kernel, degree=degree, gamma=gamma,
             coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0., shrinking=shrinking,
@@ -571,7 +573,17 @@ def predict(self, X):
         y_pred : array, shape (n_samples,)
             Class labels for samples in X.
         """
-        y = super().predict(X)
+        check_is_fitted(self, "classes_")
+        if self.break_ties and self.decision_function_shape == 'ovo':
+            raise ValueError("break_ties must be False when "
+                             "decision_function_shape is 'ovo'")
+
+        if (self.break_ties
+                and self.decision_function_shape == 'ovr'
+                and len(self.classes_) > 2):
+            y = np.argmax(self.decision_function(X), axis=1)
+        else:
+            y = super().predict(X)
         return self.classes_.take(np.asarray(y, dtype=np.intp))
 
     # Hacky way of getting predict_proba to raise an AttributeError when
diff --git a/sklearn/svm/classes.py b/sklearn/svm/classes.py
--- a/sklearn/svm/classes.py
+++ b/sklearn/svm/classes.py
@@ -521,6 +521,15 @@ class SVC(BaseSVC):
         .. versionchanged:: 0.17
            Deprecated *decision_function_shape='ovo' and None*.
 
+    break_ties : bool, optional (default=False)
+        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
+        :term:`predict` will break ties according to the confidence values of
+        :term:`decision_function`; otherwise the first class among the tied
+        classes is returned. Please note that breaking ties comes at a
+        relatively high computational cost compared to a simple predict.
+
+        .. versionadded:: 0.22
+
     random_state : int, RandomState instance or None, optional (default=None)
         The seed of the pseudo random number generator used when shuffling
         the data for probability estimates. If int, random_state is the
@@ -578,10 +587,10 @@ class SVC(BaseSVC):
     >>> from sklearn.svm import SVC
     >>> clf = SVC(gamma='auto')
     >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
-    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
+    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
-        max_iter=-1, probability=False, random_state=None, shrinking=True,
-        tol=0.001, verbose=False)
+        max_iter=-1, probability=False,
+        random_state=None, shrinking=True, tol=0.001, verbose=False)
     >>> print(clf.predict([[-0.8, -1]]))
     [1]
 
@@ -611,6 +620,7 @@ def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape='ovr',
+                 break_ties=False,
                  random_state=None):
 
         super().__init__(
@@ -619,6 +629,7 @@ def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
             probability=probability, cache_size=cache_size,
             class_weight=class_weight, verbose=verbose, max_iter=max_iter,
             decision_function_shape=decision_function_shape,
+            break_ties=break_ties,
             random_state=random_state)
 
 
@@ -707,6 +718,15 @@ class NuSVC(BaseSVC):
         .. versionchanged:: 0.17
            Deprecated *decision_function_shape='ovo' and None*.
 
+    break_ties : bool, optional (default=False)
+        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
+        :term:`predict` will break ties according to the confidence values of
+        :term:`decision_function`; otherwise the first class among the tied
+        classes is returned. Please note that breaking ties comes at a
+        relatively high computational cost compared to a simple predict.
+
+        .. versionadded:: 0.22
+
     random_state : int, RandomState instance or None, optional (default=None)
         The seed of the pseudo random number generator used when shuffling
         the data for probability estimates. If int, random_state is the seed
@@ -750,10 +770,10 @@ class NuSVC(BaseSVC):
     >>> from sklearn.svm import NuSVC
     >>> clf = NuSVC(gamma='scale')
     >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
-    NuSVC(cache_size=200, class_weight=None, coef0=0.0,
+    NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
           decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
-          max_iter=-1, nu=0.5, probability=False, random_state=None,
-          shrinking=True, tol=0.001, verbose=False)
+          max_iter=-1, nu=0.5, probability=False,
+          random_state=None, shrinking=True, tol=0.001, verbose=False)
     >>> print(clf.predict([[-0.8, -1]]))
     [1]
 
@@ -778,7 +798,8 @@ class NuSVC(BaseSVC):
     def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated',
                  coef0=0.0, shrinking=True, probability=False, tol=1e-3,
                  cache_size=200, class_weight=None, verbose=False, max_iter=-1,
-                 decision_function_shape='ovr', random_state=None):
+                 decision_function_shape='ovr', break_ties=False,
+                 random_state=None):
 
         super().__init__(
             kernel=kernel, degree=degree, gamma=gamma,
@@ -786,6 +807,7 @@ def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated',
             probability=probability, cache_size=cache_size,
             class_weight=class_weight, verbose=verbose, max_iter=max_iter,
             decision_function_shape=decision_function_shape,
+            break_ties=break_ties,
             random_state=random_state)
 
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| examples/svm/plot_svm_tie_breaking.py | 0 | 0 | - | - | -
| sklearn/model_selection/_search.py | 986 | 990 | - | 23 | -
| sklearn/svm/base.py | 504 | 504 | 65 | 3 | 31280
| sklearn/svm/base.py | 574 | 574 | 18 | 3 | 8890
| sklearn/svm/classes.py | 524 | 524 | 59 | 5 | 28417
| sklearn/svm/classes.py | 581 | 584 | 59 | 5 | 28417
| sklearn/svm/classes.py | 614 | 614 | 59 | 5 | 28417
| sklearn/svm/classes.py | 622 | 622 | 59 | 5 | 28417
| sklearn/svm/classes.py | 710 | 710 | 17 | 5 | 8744
| sklearn/svm/classes.py | 753 | 756 | 17 | 5 | 8744
| sklearn/svm/classes.py | 781 | 781 | 7 | 5 | 2867
| sklearn/svm/classes.py | 789 | 789 | 7 | 5 | 2867


## Problem Statement

```
SVC.decision_function disagrees with predict
In ``SVC`` with ``decision_function_shape="ovr"`` argmax of the decision function is not the same as ``predict``. This is related to the tie-breaking mentioned in #8276.

The ``decision_function`` now includes tie-breaking, which the ``predict`` doesn't.
I'm not sure the tie-breaking is good, but we should be consistent either way.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 benchmarks/bench_20newsgroups.py | 1 | 97| 778 | 778 | 778 | 
| 2 | 2 sklearn/utils/multiclass.py | 402 | 444| 442 | 1220 | 4654 | 
| 3 | **3 sklearn/svm/base.py** | 527 | 556| 303 | 1523 | 12870 | 
| 4 | 4 sklearn/multiclass.py | 591 | 632| 344 | 1867 | 19312 | 
| **-> 5 <-** | **5 sklearn/svm/classes.py** | 608 | 622| 177 | 2044 | 30888 | 
| 6 | 6 examples/ensemble/plot_voting_decision_regions.py | 1 | 74| 644 | 2688 | 31532 | 
| **-> 7 <-** | **6 sklearn/svm/classes.py** | 776 | 789| 179 | 2867 | 31532 | 
| 8 | **6 sklearn/svm/base.py** | 577 | 587| 133 | 3000 | 31532 | 
| 9 | **6 sklearn/svm/base.py** | 382 | 410| 241 | 3241 | 31532 | 
| 10 | 7 sklearn/multioutput.py | 619 | 652| 296 | 3537 | 37133 | 
| 11 | **7 sklearn/svm/base.py** | 658 | 676| 170 | 3707 | 37133 | 
| 12 | 8 benchmarks/bench_saga.py | 107 | 189| 637 | 4344 | 39599 | 
| 13 | **8 sklearn/svm/base.py** | 678 | 713| 317 | 4661 | 39599 | 
| **-> 14 <-** | **8 sklearn/svm/classes.py** | 430 | 606| 1839 | 6500 | 39599 | 
| 15 | 9 sklearn/tree/tree.py | 388 | 404| 180 | 6680 | 53138 | 
| 16 | 10 examples/classification/plot_classifier_comparison.py | 1 | 78| 633 | 7313 | 54490 | 
| **-> 17 <-** | **10 sklearn/svm/classes.py** | 625 | 774| 1431 | 8744 | 54490 | 
| **-> 18 <-** | **10 sklearn/svm/base.py** | 558 | 575| 146 | 8890 | 54490 | 
| 19 | 11 examples/svm/plot_svm_nonlinear.py | 1 | 43| 318 | 9208 | 54808 | 
| 20 | 12 examples/semi_supervised/plot_label_propagation_versus_svm_iris.py | 1 | 80| 697 | 9905 | 55525 | 
| 21 | 13 sklearn/utils/estimator_checks.py | 1433 | 1532| 940 | 10845 | 77501 | 
| 22 | 14 sklearn/metrics/scorer.py | 466 | 530| 702 | 11547 | 81942 | 
| 23 | 14 sklearn/utils/estimator_checks.py | 1728 | 1769| 417 | 11964 | 81942 | 
| 24 | **14 sklearn/svm/classes.py** | 1192 | 1221| 190 | 12154 | 81942 | 
| 25 | 14 sklearn/utils/estimator_checks.py | 1883 | 1904| 214 | 12368 | 81942 | 
| 26 | **14 sklearn/svm/base.py** | 412 | 426| 150 | 12518 | 81942 | 
| 27 | 14 sklearn/metrics/scorer.py | 146 | 210| 478 | 12996 | 81942 | 
| 28 | 15 examples/svm/plot_separating_hyperplane_unbalanced.py | 1 | 81| 648 | 13644 | 82590 | 
| 29 | 15 sklearn/multiclass.py | 569 | 589| 182 | 13826 | 82590 | 
| 30 | 16 examples/plot_kernel_ridge_regression.py | 1 | 79| 759 | 14585 | 84310 | 
| 31 | 17 sklearn/linear_model/stochastic_gradient.py | 997 | 1038| 307 | 14892 | 98238 | 
| 32 | 17 sklearn/utils/estimator_checks.py | 1555 | 1627| 655 | 15547 | 98238 | 
| 33 | 18 examples/svm/plot_iris_svc.py | 1 | 40| 300 | 15847 | 99217 | 
| 34 | 18 sklearn/utils/estimator_checks.py | 2358 | 2387| 294 | 16141 | 99217 | 
| 35 | 19 examples/svm/plot_rbf_parameters.py | 74 | 160| 732 | 16873 | 101199 | 
| 36 | **19 sklearn/svm/base.py** | 513 | 525| 138 | 17011 | 101199 | 
| 37 | 20 sklearn/ensemble/weight_boosting.py | 699 | 750| 441 | 17452 | 110315 | 
| 38 | 21 examples/ensemble/plot_bias_variance.py | 1 | 64| 761 | 18213 | 112129 | 
| 39 | 21 sklearn/tree/tree.py | 278 | 363| 800 | 19013 | 112129 | 
| 40 | 21 sklearn/ensemble/weight_boosting.py | 659 | 697| 370 | 19383 | 112129 | 
| 41 | **21 sklearn/svm/classes.py** | 15 | 171| 1643 | 21026 | 112129 | 
| 42 | 21 sklearn/utils/estimator_checks.py | 1907 | 1951| 472 | 21498 | 112129 | 
| 43 | 22 examples/svm/plot_svm_anova.py | 1 | 57| 458 | 21956 | 112587 | 
| 44 | 22 examples/svm/plot_rbf_parameters.py | 161 | 202| 453 | 22409 | 112587 | 
| 45 | **22 sklearn/svm/base.py** | 450 | 475| 275 | 22684 | 112587 | 
| 46 | 22 examples/plot_kernel_ridge_regression.py | 80 | 153| 738 | 23422 | 112587 | 
| 47 | **23 sklearn/model_selection/_search.py** | 493 | 508| 130 | 23552 | 125954 | 
| 48 | **23 sklearn/svm/classes.py** | 910 | 920| 161 | 23713 | 125954 | 
| 49 | 24 sklearn/metrics/classification.py | 2263 | 2301| 383 | 24096 | 148316 | 
| 50 | 25 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 24889 | 150162 | 
| 51 | 26 examples/decomposition/plot_pca_vs_fa_model_selection.py | 88 | 126| 437 | 25326 | 151283 | 
| 52 | 26 sklearn/multiclass.py | 110 | 130| 137 | 25463 | 151283 | 
| 53 | 26 sklearn/tree/tree.py | 203 | 276| 816 | 26279 | 151283 | 
| 54 | **26 sklearn/svm/classes.py** | 1223 | 1242| 138 | 26417 | 151283 | 
| 55 | 27 examples/svm/plot_svm_scale_c.py | 1 | 102| 763 | 27180 | 152629 | 
| 56 | 28 sklearn/model_selection/_validation.py | 884 | 938| 538 | 27718 | 166182 | 
| 57 | 29 sklearn/ensemble/bagging.py | 170 | 181| 104 | 27822 | 174166 | 
| 58 | 29 examples/svm/plot_iris_svc.py | 63 | 77| 114 | 27936 | 174166 | 
| **-> 59 <-** | **29 sklearn/svm/classes.py** | 376 | 622| 481 | 28417 | 174166 | 
| 60 | 29 sklearn/utils/estimator_checks.py | 2390 | 2434| 427 | 28844 | 174166 | 
| 61 | 30 benchmarks/bench_isolation_forest.py | 54 | 161| 1032 | 29876 | 175634 | 
| 62 | **30 sklearn/svm/base.py** | 428 | 448| 205 | 30081 | 175634 | 
| 63 | 30 sklearn/model_selection/_validation.py | 815 | 881| 518 | 30599 | 175634 | 
| 64 | 31 examples/linear_model/plot_lasso_model_selection.py | 95 | 159| 513 | 31112 | 177004 | 
| **-> 65 <-** | **31 sklearn/svm/base.py** | 499 | 511| 168 | 31280 | 177004 | 
| 66 | **31 sklearn/svm/base.py** | 1 | 21| 187 | 31467 | 177004 | 
| 67 | 32 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 820 | 839| 156 | 31623 | 185102 | 
| 68 | 33 examples/ensemble/plot_voting_regressor.py | 1 | 54| 363 | 31986 | 185465 | 
| 69 | **33 sklearn/svm/base.py** | 307 | 324| 152 | 32138 | 185465 | 
| 70 | 34 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 32519 | 186780 | 


## Missing Patch Files

 * 1: examples/svm/plot_svm_tie_breaking.py
 * 2: sklearn/model_selection/_search.py
 * 3: sklearn/svm/base.py
 * 4: sklearn/svm/classes.py

### Hint

```
The relevant issue on `libsvm` (i.e. issue https://github.com/cjlin1/libsvm/issues/85) seems to be stalled and I'm not sure if it's had a conclusion. At least on the `libsvm` side it seems they prefer not to include the confidences for computational cost of it.

Now the question is, do we want to change the `svm.cpp` to fix the issue and take the confidences into account? Or do we want the `SVC.predict` to use `SVC.decision_function` instead of calling `libsvm`'s `predict`? Or to fix the documentation and make it clear that `decision_function` and `predict` may not agree in case of a tie?
Does this related to #9174 also? Does #10440 happen to fix it??​

@jnothman not really, that's exactly the issue actually. the `predict` function which calls the `libsvm`'s `predict` function, does not use confidences to break the ties, but the `decision_function` does (as a result of the issue and the PR you mentioned).
Ah, I see.​ Hmm...

Another alternative is to have a `break_ties_in_predict` (or a much better parameter name that you can think of), have the default `False`, and if `True`, return the argmax of the decision_function. It would be very little code and backward compatible.
@adrinjalali and then change the default value? Or keep it inconsistent by default? (neither of these seems like great options to me ;)
@amueller I know it's not ideal, but my intuition says:

1- argmax of decision function is much slower than predict, which is why I'd see why in most usercases people would prefer to just ignore the tie breaking issue.
2- in practice, ties happen very rarely, therefore the inconsistency is actually not that big of an issue (we'd be seeing more reported issues if that was happening more often and people were noticing, I guess).

Therefore, I'd say the default value `False` is not an unreasonable case at all. On top of that, we'd document this properly both in the docstring and in the user manuals. And if the user really would like them to be consistent, they can make it so.

One could make the same argument and say the change is not necessary at all then. In response to which, I'd say the change is trivial and not much code at all, and easy to maintain, therefore there's not much damage the change would do. I kinda see this proposal as a compromise between the two cases of leaving it as is, and fixing it for everybody and breaking backward compatibility.

Also, independent of this change, if after the introduction of the change,we see some demand for the default to be `True` (which I'd doubt), we can do it in a rather long deprecation cycle to give people enough time to change/fix their code.

(I'm just explaining my proposed solution, absolutely no attachments to the proposal :P )
Adding a parameter would certainly make more users aware of this issue, and
it is somewhat like other parameters giving efficiency tradeoffs.

>

```

## Patch

```diff
diff --git a/examples/svm/plot_svm_tie_breaking.py b/examples/svm/plot_svm_tie_breaking.py
new file mode 100644
--- /dev/null
+++ b/examples/svm/plot_svm_tie_breaking.py
@@ -0,0 +1,64 @@
+"""
+=========================================================
+SVM Tie Breaking Example
+=========================================================
+Tie breaking is costly if ``decision_function_shape='ovr'``, and therefore it
+is not enabled by default. This example illustrates the effect of the
+``break_ties`` parameter for a multiclass classification problem and
+``decision_function_shape='ovr'``.
+
+The two plots differ only in the area in the middle where the classes are
+tied. If ``break_ties=False``, all input in that area would be classified as
+one class, whereas if ``break_ties=True``, the tie-breaking mechanism will
+create a non-convex decision boundary in that area.
+"""
+print(__doc__)
+
+
+# Code source: Andreas Mueller, Adrin Jalali
+# License: BSD 3 clause
+
+
+import numpy as np
+import matplotlib.pyplot as plt
+from sklearn.svm import SVC
+from sklearn.datasets import make_blobs
+
+X, y = make_blobs(random_state=27)
+
+fig, sub = plt.subplots(2, 1, figsize=(5, 8))
+titles = ("break_ties = False",
+          "break_ties = True")
+
+for break_ties, title, ax in zip((False, True), titles, sub.flatten()):
+
+    svm = SVC(kernel="linear", C=1, break_ties=break_ties,
+              decision_function_shape='ovr').fit(X, y)
+
+    xlim = [X[:, 0].min(), X[:, 0].max()]
+    ylim = [X[:, 1].min(), X[:, 1].max()]
+
+    xs = np.linspace(xlim[0], xlim[1], 1000)
+    ys = np.linspace(ylim[0], ylim[1], 1000)
+    xx, yy = np.meshgrid(xs, ys)
+
+    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
+
+    colors = [plt.cm.Accent(i) for i in [0, 4, 7]]
+
+    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent")
+    classes = [(0, 1), (0, 2), (1, 2)]
+    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
+    ax.imshow(-pred.reshape(xx.shape), cmap="Accent", alpha=.2,
+              extent=(xlim[0], xlim[1], ylim[1], ylim[0]))
+
+    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
+        line2 = -(line * coef[1] + intercept) / coef[0]
+        ax.plot(line2, line, "-", c=colors[col[0]])
+        ax.plot(line2, line, "--", c=colors[col[1]])
+    ax.set_xlim(xlim)
+    ax.set_ylim(ylim)
+    ax.set_title(title)
+    ax.set_aspect("equal")
+
+plt.show()
diff --git a/sklearn/model_selection/_search.py b/sklearn/model_selection/_search.py
--- a/sklearn/model_selection/_search.py
+++ b/sklearn/model_selection/_search.py
@@ -983,11 +983,13 @@ class GridSearchCV(BaseSearchCV):
     >>> clf.fit(iris.data, iris.target)
     ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
     GridSearchCV(cv=5, error_score=...,
-           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
+           estimator=SVC(C=1.0, break_ties=False, cache_size=...,
+                         class_weight=..., coef0=...,
                          decision_function_shape='ovr', degree=..., gamma=...,
-                         kernel='rbf', max_iter=-1, probability=False,
-                         random_state=None, shrinking=True, tol=...,
-                         verbose=False),
+                         kernel='rbf', max_iter=-1,
+                         probability=False,
+                         random_state=None, shrinking=True,
+                         tol=..., verbose=False),
            iid=..., n_jobs=None,
            param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
            scoring=..., verbose=...)
diff --git a/sklearn/svm/base.py b/sklearn/svm/base.py
--- a/sklearn/svm/base.py
+++ b/sklearn/svm/base.py
@@ -501,8 +501,10 @@ class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):
     @abstractmethod
     def __init__(self, kernel, degree, gamma, coef0, tol, C, nu,
                  shrinking, probability, cache_size, class_weight, verbose,
-                 max_iter, decision_function_shape, random_state):
+                 max_iter, decision_function_shape, random_state,
+                 break_ties):
         self.decision_function_shape = decision_function_shape
+        self.break_ties = break_ties
         super().__init__(
             kernel=kernel, degree=degree, gamma=gamma,
             coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0., shrinking=shrinking,
@@ -571,7 +573,17 @@ def predict(self, X):
         y_pred : array, shape (n_samples,)
             Class labels for samples in X.
         """
-        y = super().predict(X)
+        check_is_fitted(self, "classes_")
+        if self.break_ties and self.decision_function_shape == 'ovo':
+            raise ValueError("break_ties must be False when "
+                             "decision_function_shape is 'ovo'")
+
+        if (self.break_ties
+                and self.decision_function_shape == 'ovr'
+                and len(self.classes_) > 2):
+            y = np.argmax(self.decision_function(X), axis=1)
+        else:
+            y = super().predict(X)
         return self.classes_.take(np.asarray(y, dtype=np.intp))
 
     # Hacky way of getting predict_proba to raise an AttributeError when
diff --git a/sklearn/svm/classes.py b/sklearn/svm/classes.py
--- a/sklearn/svm/classes.py
+++ b/sklearn/svm/classes.py
@@ -521,6 +521,15 @@ class SVC(BaseSVC):
         .. versionchanged:: 0.17
            Deprecated *decision_function_shape='ovo' and None*.
 
+    break_ties : bool, optional (default=False)
+        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
+        :term:`predict` will break ties according to the confidence values of
+        :term:`decision_function`; otherwise the first class among the tied
+        classes is returned. Please note that breaking ties comes at a
+        relatively high computational cost compared to a simple predict.
+
+        .. versionadded:: 0.22
+
     random_state : int, RandomState instance or None, optional (default=None)
         The seed of the pseudo random number generator used when shuffling
         the data for probability estimates. If int, random_state is the
@@ -578,10 +587,10 @@ class SVC(BaseSVC):
     >>> from sklearn.svm import SVC
     >>> clf = SVC(gamma='auto')
     >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
-    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
+    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
-        max_iter=-1, probability=False, random_state=None, shrinking=True,
-        tol=0.001, verbose=False)
+        max_iter=-1, probability=False,
+        random_state=None, shrinking=True, tol=0.001, verbose=False)
     >>> print(clf.predict([[-0.8, -1]]))
     [1]
 
@@ -611,6 +620,7 @@ def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-3, cache_size=200, class_weight=None,
                  verbose=False, max_iter=-1, decision_function_shape='ovr',
+                 break_ties=False,
                  random_state=None):
 
         super().__init__(
@@ -619,6 +629,7 @@ def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
             probability=probability, cache_size=cache_size,
             class_weight=class_weight, verbose=verbose, max_iter=max_iter,
             decision_function_shape=decision_function_shape,
+            break_ties=break_ties,
             random_state=random_state)
 
 
@@ -707,6 +718,15 @@ class NuSVC(BaseSVC):
         .. versionchanged:: 0.17
            Deprecated *decision_function_shape='ovo' and None*.
 
+    break_ties : bool, optional (default=False)
+        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
+        :term:`predict` will break ties according to the confidence values of
+        :term:`decision_function`; otherwise the first class among the tied
+        classes is returned. Please note that breaking ties comes at a
+        relatively high computational cost compared to a simple predict.
+
+        .. versionadded:: 0.22
+
     random_state : int, RandomState instance or None, optional (default=None)
         The seed of the pseudo random number generator used when shuffling
         the data for probability estimates. If int, random_state is the seed
@@ -750,10 +770,10 @@ class NuSVC(BaseSVC):
     >>> from sklearn.svm import NuSVC
     >>> clf = NuSVC(gamma='scale')
     >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
-    NuSVC(cache_size=200, class_weight=None, coef0=0.0,
+    NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
           decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
-          max_iter=-1, nu=0.5, probability=False, random_state=None,
-          shrinking=True, tol=0.001, verbose=False)
+          max_iter=-1, nu=0.5, probability=False,
+          random_state=None, shrinking=True, tol=0.001, verbose=False)
     >>> print(clf.predict([[-0.8, -1]]))
     [1]
 
@@ -778,7 +798,8 @@ class NuSVC(BaseSVC):
     def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated',
                  coef0=0.0, shrinking=True, probability=False, tol=1e-3,
                  cache_size=200, class_weight=None, verbose=False, max_iter=-1,
-                 decision_function_shape='ovr', random_state=None):
+                 decision_function_shape='ovr', break_ties=False,
+                 random_state=None):
 
         super().__init__(
             kernel=kernel, degree=degree, gamma=gamma,
@@ -786,6 +807,7 @@ def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated',
             probability=probability, cache_size=cache_size,
             class_weight=class_weight, verbose=verbose, max_iter=max_iter,
             decision_function_shape=decision_function_shape,
+            break_ties=break_ties,
             random_state=random_state)
 
 

```

## Test Patch

```diff
diff --git a/sklearn/svm/tests/test_svm.py b/sklearn/svm/tests/test_svm.py
--- a/sklearn/svm/tests/test_svm.py
+++ b/sklearn/svm/tests/test_svm.py
@@ -985,6 +985,41 @@ def test_ovr_decision_function():
     assert np.all(pred_class_deci_val[:, 0] < pred_class_deci_val[:, 1])
 
 
+@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
+def test_svc_invalid_break_ties_param(SVCClass):
+    X, y = make_blobs(random_state=42)
+
+    svm = SVCClass(kernel="linear", decision_function_shape='ovo',
+                   break_ties=True, random_state=42).fit(X, y)
+
+    with pytest.raises(ValueError, match="break_ties must be False"):
+        svm.predict(y)
+
+
+@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
+def test_svc_ovr_tie_breaking(SVCClass):
+    """Test if predict breaks ties in OVR mode.
+    Related issue: https://github.com/scikit-learn/scikit-learn/issues/8277
+    """
+    X, y = make_blobs(random_state=27)
+
+    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 1000)
+    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 1000)
+    xx, yy = np.meshgrid(xs, ys)
+
+    svm = SVCClass(kernel="linear", decision_function_shape='ovr',
+                   break_ties=False, random_state=42).fit(X, y)
+    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
+    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
+    assert not np.all(pred == np.argmax(dv, axis=1))
+
+    svm = SVCClass(kernel="linear", decision_function_shape='ovr',
+                   break_ties=True, random_state=42).fit(X, y)
+    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
+    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
+    assert np.all(pred == np.argmax(dv, axis=1))
+
+
 def test_gamma_auto():
     X, y = [[0.0, 1.2], [1.0, 1.3]], [0, 1]
 

```


## Code snippets

### 1 - benchmarks/bench_20newsgroups.py:

Start line: 1, End line: 97

```python
from time import time
import argparse
import numpy as np

from sklearn.dummy import DummyClassifier

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ESTIMATORS = {
    "dummy": DummyClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100,
                                            max_features="sqrt",
                                            min_samples_split=10),
    "extra_trees": ExtraTreesClassifier(n_estimators=100,
                                        max_features="sqrt",
                                        min_samples_split=10),
    "logistic_regression": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "adaboost": AdaBoostClassifier(n_estimators=10),
}


###############################################################################
# Data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=ESTIMATORS)
    args = vars(parser.parse_args())

    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = check_array(data_train.data, dtype=np.float32,
                          accept_sparse="csc")
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")
    y_train = data_train.target
    y_test = data_test.target

    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train.shape))
    print("X_train.format = {0}".format(X_train.format))
    print("X_train.dtype = {0}".format(X_train.dtype))
    print("X_train density = {0}"
          "".format(X_train.nnz / np.product(X_train.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test.shape))
    print("X_test.format = {0}".format(X_test.format))
    print("X_test.dtype = {0}".format(X_test.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}
    for name in sorted(args["estimators"]):
        clf = ESTIMATORS[name]
        try:
            clf.set_params(random_state=0)
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time",
                           "Accuracy"))
    print("-" * 44)
    for name in sorted(accuracy, key=accuracy.get):
        print("%s %s %s %s" % (name.ljust(16),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % accuracy[name]).center(10)))

    print()
```
### 2 - sklearn/utils/multiclass.py:

Start line: 402, End line: 444

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

    # Monotonically transform the sum_of_confidences to (-1/3, 1/3)
    # and add it with votes. The monotonic transformation  is
    # f: x -> x / (3 * (|x| + 1)), it uses 1/3 instead of 1/2
    # to ensure that we won't reach the limits and change vote order.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    transformed_confidences = (sum_of_confidences /
                               (3 * (np.abs(sum_of_confidences) + 1)))
    return votes + transformed_confidences
```
### 3 - sklearn/svm/base.py:

Start line: 527, End line: 556

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes).

        Notes
        ------
        If decision_function_shape='ovo', the function values are proportional
        to the distance of the samples X to the separating hyperplane. If the
        exact distances are required, divide the function values by the norm of
        the weight vector (``coef_``). See also `this question
        <https://stats.stackexchange.com/questions/14876/
        interpreting-distance-from-hyperplane-in-svm>`_ for further details.
        If decision_function_shape='ovr', the decision function is a monotonic
        transformation of ovo decision function.
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec
```
### 4 - sklearn/multiclass.py:

Start line: 591, End line: 632

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

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)
```
### 5 - sklearn/svm/classes.py:

Start line: 608, End line: 622

```python
class SVC(BaseSVC):

    _impl = 'c_svc'

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)
```
### 6 - examples/ensemble/plot_voting_decision_regions.py:

Start line: 1, End line: 74

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
clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
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
### 7 - sklearn/svm/classes.py:

Start line: 776, End line: 789

```python
class NuSVC(BaseSVC):

    _impl = 'nu_svc'

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False, tol=1e-3,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=0., nu=nu, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)
```
### 8 - sklearn/svm/base.py:

Start line: 577, End line: 587

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        if not self.probability:
            raise AttributeError("predict_proba is not available when "
                                 " probability=False")
        if self._impl not in ('c_svc', 'nu_svc'):
            raise AttributeError("predict_proba only implemented for SVC"
                                 " and NuSVC")
```
### 9 - sklearn/svm/base.py:

Start line: 382, End line: 410

```python
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):

    def _decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        # NOTE: _validate_for_predict contains check for is_fitted
        # hence must be placed before any other attributes are used.
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)

        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function.
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func
```
### 10 - sklearn/multioutput.py:

Start line: 619, End line: 652

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

    def _more_tags(self):
        return {'_skip_test': True,
                'multioutput_only': True}
```
### 11 - sklearn/svm/base.py:

Start line: 658, End line: 676

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

    def _predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

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
### 13 - sklearn/svm/base.py:

Start line: 678, End line: 713

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

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

    def _get_coef(self):
        if self.dual_coef_.shape[0] == 1:
            # binary classifier
            coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
        else:
            # 1vs1 classifier
            coef = _one_vs_one_coef(self.dual_coef_, self.n_support_,
                                    self.support_vectors_)
            if sp.issparse(coef[0]):
                coef = sp.vstack(coef).tocsr()
            else:
                coef = np.vstack(coef)

        return coef
```
### 14 - sklearn/svm/classes.py:

Start line: 430, End line: 606

```python
class SVC(BaseSVC):
    """C-Support Vector Classification.

    The implementation is based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to datasets with more than a couple of 10000 samples. For large
    datasets consider using :class:`sklearn.linear_model.LinearSVC` or
    :class:`sklearn.linear_model.SGDClassifier` instead, possibly after a
    :class:`sklearn.kernel_approximation.Nystroem` transformer.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each
    other, see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        Current default is 'auto' which uses 1 / n_features,
        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, 'auto', will change
        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
        'auto' is used as a default indicating that no explicit value of gamma
        was passed.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : 'ovo', 'ovr', default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy.

        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.

        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.

        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator used when shuffling
        the data for probability estimates. If int, random_state is the
        seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random
        number generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the section about multi-class classification in the
        SVM section of the User Guide for details.

    coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    probA_ : array, shape = [n_class * (n_class-1) / 2]
    probB_ : array, shape = [n_class * (n_class-1) / 2]
        If probability=True, the parameters learned in Platt scaling to
        produce probability estimates from decision values. If
        probability=False, an empty array. Platt scaling uses the logistic
        function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = SVC(gamma='auto')
    >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SVR
        Support Vector Machine for Regression implemented using libsvm.

    LinearSVC
        Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See also section of
        LinearSVC for more comparison element.

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic outputs for support vector
        machines and comparison to regularizedlikelihood methods."
        <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
    """
```
### 17 - sklearn/svm/classes.py:

Start line: 625, End line: 774

```python
class NuSVC(BaseSVC):
    """Nu-Support Vector Classification.

    Similar to SVC but uses a parameter to control the number of support
    vectors.

    The implementation is based on libsvm.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    nu : float, optional (default=0.5)
        An upper bound on the fraction of training errors and a lower
        bound of the fraction of support vectors. Should be in the
        interval (0, 1].

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        Current default is 'auto' which uses 1 / n_features,
        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, 'auto', will change
        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
        'auto' is used as a default indicating that no explicit value of gamma
        was passed.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The "balanced" mode uses the values of y to automatically
        adjust weights inversely proportional to class frequencies as
        ``n_samples / (n_classes * np.bincount(y))``

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : 'ovo', 'ovr', default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2).

        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.

        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.

        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator used when shuffling
        the data for probability estimates. If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random
        number generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the section about multi-class classification in
        the SVM section of the User Guide for details.

    coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import NuSVC
    >>> clf = NuSVC(gamma='scale')
    >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
    NuSVC(cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
          max_iter=-1, nu=0.5, probability=False, random_state=None,
          shrinking=True, tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SVC
        Support Vector Machine for classification using libsvm.

    LinearSVC
        Scalable linear Support Vector Machine for classification using
        liblinear.

    Notes
    -----
    **References:**
    `LIBSVM: A Library for Support Vector Machines
    <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`__
    """
```
### 18 - sklearn/svm/base.py:

Start line: 558, End line: 575

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

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
        y = super().predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp))
```
### 24 - sklearn/svm/classes.py:

Start line: 1192, End line: 1221

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

    def score_samples(self, X):
        """Raw scoring function of the samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        score_samples : array-like, shape (n_samples,)
            Returns the (unshifted) scoring function of the samples.
        """
        return self.decision_function(X) + self.offset_
```
### 26 - sklearn/svm/base.py:

Start line: 412, End line: 426

```python
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order="C",
                        accept_large_sparse=False)

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        return libsvm.decision_function(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel, degree=self.degree, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma)
```
### 36 - sklearn/svm/base.py:

Start line: 513, End line: 525

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')
```
### 41 - sklearn/svm/classes.py:

Start line: 15, End line: 171

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
    >>> clf = LinearSVC(random_state=0, tol=1e-5)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    >>> print(clf.coef_)
    [[0.085... 0.394... 0.498... 0.375...]]
    >>> print(clf.intercept_)
    [0.284...]
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
    <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

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

    """
```
### 45 - sklearn/svm/base.py:

Start line: 450, End line: 475

```python
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):

    def _validate_for_predict(self, X):
        check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C",
                        accept_large_sparse=False)
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)
        n_samples, n_features = X.shape

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
        elif n_features != self.shape_fit_[1]:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time" %
                             (n_features, self.shape_fit_[1]))
        return X
```
### 47 - sklearn/model_selection/_search.py:

Start line: 493, End line: 508

```python
class BaseSearchCV(BaseEstimator, MetaEstimatorMixin, metaclass=ABCMeta):

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
### 48 - sklearn/svm/classes.py:

Start line: 910, End line: 920

```python
class SVR(BaseLibSVM, RegressorMixin):

    _impl = 'epsilon_svr'

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., epsilon=epsilon, verbose=verbose,
            shrinking=shrinking, probability=False, cache_size=cache_size,
            class_weight=None, max_iter=max_iter, random_state=None)
```
### 54 - sklearn/svm/classes.py:

Start line: 1223, End line: 1242

```python
class OneClassSVM(BaseLibSVM, OutlierMixin):

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

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
        y = super().predict(X)
        return np.asarray(y, dtype=np.intp)
```
### 59 - sklearn/svm/classes.py:

Start line: 376, End line: 622

```python
class LinearSVR(LinearModel, RegressorMixin):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        # FIXME Remove l1/l2 support in 1.0 -----------------------------------
        msg = ("loss='%s' has been deprecated in favor of "
               "loss='%s' as of 0.16. Backward compatibility"
               " for the loss='%s' will be removed in %s")

        if self.loss in ('l1', 'l2'):
            old_loss = self.loss
            self.loss = {'l1': 'epsilon_insensitive',
                         'l2': 'squared_epsilon_insensitive'
                         }.get(self.loss)
            warnings.warn(msg % (old_loss, self.loss, old_loss, '1.0'),
                          DeprecationWarning)
        # ---------------------------------------------------------------------

        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)

        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64, order="C",
                         accept_large_sparse=False)
        penalty = 'l2'  # SVR only accepts l2 penalty
        self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
            X, y, self.C, self.fit_intercept, self.intercept_scaling,
            None, penalty, self.dual, self.verbose,
            self.max_iter, self.tol, self.random_state, loss=self.loss,
            epsilon=self.epsilon, sample_weight=sample_weight)
        self.coef_ = self.coef_.ravel()

        return self


class SVC(BaseSVC):
```
### 62 - sklearn/svm/base.py:

Start line: 428, End line: 448

```python
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')

        kernel = self.kernel
        if hasattr(kernel, '__call__'):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_decision_function(
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
### 65 - sklearn/svm/base.py:

Start line: 499, End line: 511

```python
class BaseSVC(BaseLibSVM, ClassifierMixin, metaclass=ABCMeta):
    """ABC for LibSVM-based classifiers."""
    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu,
                 shrinking, probability, cache_size, class_weight, verbose,
                 max_iter, decision_function_shape, random_state):
        self.decision_function_shape = decision_function_shape
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            random_state=random_state)
```
### 66 - sklearn/svm/base.py:

Start line: 1, End line: 21

```python
import numpy as np
import scipy.sparse as sp
import warnings
from abc import ABCMeta, abstractmethod

from . import libsvm, liblinear
from . import libsvm_sparse
from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelEncoder
from ..utils.multiclass import _ovr_decision_function
from ..utils import check_array, check_consistent_length, check_random_state
from ..utils import column_or_1d, check_X_y
from ..utils import compute_class_weight
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted, _check_large_sparse
from ..utils.multiclass import check_classification_targets
from ..exceptions import ConvergenceWarning
from ..exceptions import NotFittedError


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
```
### 69 - sklearn/svm/base.py:

Start line: 307, End line: 324

```python
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)
```
