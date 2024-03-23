# scikit-learn__scikit-learn-12557

* repo: scikit-learn/scikit-learn
* base_commit: `4de404d46d24805ff48ad255ec3169a5155986f0`

## Problem statement

SVC.decision_function disagrees with predict
In ``SVC`` with ``decision_function_shape="ovr"`` argmax of the decision function is not the same as ``predict``. This is related to the tie-breaking mentioned in #8276.

The ``decision_function`` now includes tie-breaking, which the ``predict`` doesn't.
I'm not sure the tie-breaking is good, but we should be consistent either way.


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
