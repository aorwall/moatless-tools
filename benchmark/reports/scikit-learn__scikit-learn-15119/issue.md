# scikit-learn__scikit-learn-15119

* repo: scikit-learn/scikit-learn
* base_commit: `4ca6ee4a5068f60fde2a70ed6e9f15bdfc2ce396`

## Problem statement

Inconsistent fit + transform and fit_transform for FeatureUnion
Is there a reason why the `FeatureUnion` method signature `fit_transform` accepts `fit_args` but neither `fit` nor `transform` do? It seems to go against the pattern that `fit_transform()` is the same as calling `fit().transform()`?

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L895

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L871

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L944

I see there's been discussion on supporting  `fit_args` but it's not clear if this is resolved. My case is I'm trying to migrage code I wrote a while back where I used a Pipeline and each of my transformers adds columns to a dataframe, to a FeatureUnion where each transform only returns the new columns. One of my transforms takes a third data set in addition to X and y which is used as the transform. I guess as a workaround I'll make it a param of the transform rather than a fit_arg.


## Patch

```diff
diff --git a/sklearn/pipeline.py b/sklearn/pipeline.py
--- a/sklearn/pipeline.py
+++ b/sklearn/pipeline.py
@@ -876,7 +876,7 @@ def get_feature_names(self):
                                   trans.get_feature_names()])
         return feature_names
 
-    def fit(self, X, y=None):
+    def fit(self, X, y=None, **fit_params):
         """Fit all transformers using X.
 
         Parameters
@@ -892,7 +892,7 @@ def fit(self, X, y=None):
         self : FeatureUnion
             This estimator
         """
-        transformers = self._parallel_func(X, y, {}, _fit_one)
+        transformers = self._parallel_func(X, y, fit_params, _fit_one)
         if not transformers:
             # All transformers are None
             return self

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_pipeline.py b/sklearn/tests/test_pipeline.py
--- a/sklearn/tests/test_pipeline.py
+++ b/sklearn/tests/test_pipeline.py
@@ -21,7 +21,7 @@
 from sklearn.utils.testing import assert_array_almost_equal
 from sklearn.utils.testing import assert_no_warnings
 
-from sklearn.base import clone, BaseEstimator
+from sklearn.base import clone, BaseEstimator, TransformerMixin
 from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
 from sklearn.svm import SVC
 from sklearn.neighbors import LocalOutlierFactor
@@ -35,6 +35,7 @@
 from sklearn.preprocessing import StandardScaler
 from sklearn.feature_extraction.text import CountVectorizer
 
+iris = load_iris()
 
 JUNK_FOOD_DOCS = (
     "the pizza pizza beer copyright",
@@ -240,7 +241,6 @@ def test_pipeline_init_tuple():
 
 def test_pipeline_methods_anova():
     # Test the various methods of the pipeline (anova).
-    iris = load_iris()
     X = iris.data
     y = iris.target
     # Test with Anova + LogisticRegression
@@ -319,7 +319,6 @@ def test_pipeline_raise_set_params_error():
 
 def test_pipeline_methods_pca_svm():
     # Test the various methods of the pipeline (pca + svm).
-    iris = load_iris()
     X = iris.data
     y = iris.target
     # Test with PCA + SVC
@@ -334,7 +333,6 @@ def test_pipeline_methods_pca_svm():
 
 
 def test_pipeline_score_samples_pca_lof():
-    iris = load_iris()
     X = iris.data
     # Test that the score_samples method is implemented on a pipeline.
     # Test that the score_samples method on pipeline yields same results as
@@ -365,7 +363,6 @@ def test_score_samples_on_pipeline_without_score_samples():
 
 def test_pipeline_methods_preprocessing_svm():
     # Test the various methods of the pipeline (preprocessing + svm).
-    iris = load_iris()
     X = iris.data
     y = iris.target
     n_samples = X.shape[0]
@@ -398,7 +395,6 @@ def test_fit_predict_on_pipeline():
     # test that the fit_predict method is implemented on a pipeline
     # test that the fit_predict on pipeline yields same results as applying
     # transform and clustering steps separately
-    iris = load_iris()
     scaler = StandardScaler()
     km = KMeans(random_state=0)
     # As pipeline doesn't clone estimators on construction,
@@ -456,7 +452,6 @@ def test_predict_with_predict_params():
 
 def test_feature_union():
     # basic sanity check for feature union
-    iris = load_iris()
     X = iris.data
     X -= X.mean(axis=0)
     y = iris.target
@@ -530,7 +525,6 @@ def test_make_union_kwargs():
 def test_pipeline_transform():
     # Test whether pipeline works with a transformer at the end.
     # Also test pipeline.transform and pipeline.inverse_transform
-    iris = load_iris()
     X = iris.data
     pca = PCA(n_components=2, svd_solver='full')
     pipeline = Pipeline([('pca', pca)])
@@ -549,7 +543,6 @@ def test_pipeline_transform():
 
 def test_pipeline_fit_transform():
     # Test whether pipeline works with a transformer missing fit_transform
-    iris = load_iris()
     X = iris.data
     y = iris.target
     transf = Transf()
@@ -771,7 +764,6 @@ def test_make_pipeline():
 
 def test_feature_union_weights():
     # test feature union with transformer weights
-    iris = load_iris()
     X = iris.data
     y = iris.target
     pca = PCA(n_components=2, svd_solver='randomized', random_state=0)
@@ -865,7 +857,6 @@ def test_feature_union_feature_names():
 
 
 def test_classes_property():
-    iris = load_iris()
     X = iris.data
     y = iris.target
 
@@ -987,7 +978,6 @@ def test_set_params_nested_pipeline():
 def test_pipeline_wrong_memory():
     # Test that an error is raised when memory is not a string or a Memory
     # instance
-    iris = load_iris()
     X = iris.data
     y = iris.target
     # Define memory as an integer
@@ -1022,7 +1012,6 @@ def test_pipeline_with_cache_attribute():
 
 
 def test_pipeline_memory():
-    iris = load_iris()
     X = iris.data
     y = iris.target
     cachedir = mkdtemp()
@@ -1161,3 +1150,26 @@ def test_verbose(est, method, pattern, capsys):
     est.set_params(verbose=True)
     func(X, y)
     assert re.match(pattern, capsys.readouterr().out)
+
+
+def test_feature_union_fit_params():
+    # Regression test for issue: #15117
+    class Dummy(TransformerMixin, BaseEstimator):
+        def fit(self, X, y=None, **fit_params):
+            if fit_params != {'a': 0}:
+                raise ValueError
+            return self
+
+        def transform(self, X, y=None):
+            return X
+
+    X, y = iris.data, iris.target
+    t = FeatureUnion([('dummy0', Dummy()), ('dummy1', Dummy())])
+    with pytest.raises(ValueError):
+        t.fit(X, y)
+
+    with pytest.raises(ValueError):
+        t.fit_transform(X, y)
+
+    t.fit(X, y, a=0)
+    t.fit_transform(X, y, a=0)

```
