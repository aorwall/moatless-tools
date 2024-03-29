# scikit-learn__scikit-learn-15119

| **scikit-learn/scikit-learn** | `4ca6ee4a5068f60fde2a70ed6e9f15bdfc2ce396` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 383 |
| **Any found context length** | 383 |
| **Avg pos** | 4.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/pipeline.py | 879 | 879 | 2 | 1 | 383
| sklearn/pipeline.py | 895 | 895 | 2 | 1 | 383


## Problem Statement

```
Inconsistent fit + transform and fit_transform for FeatureUnion
Is there a reason why the `FeatureUnion` method signature `fit_transform` accepts `fit_args` but neither `fit` nor `transform` do? It seems to go against the pattern that `fit_transform()` is the same as calling `fit().transform()`?

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L895

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L871

https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/pipeline.py#L944

I see there's been discussion on supporting  `fit_args` but it's not clear if this is resolved. My case is I'm trying to migrage code I wrote a while back where I used a Pipeline and each of my transformers adds columns to a dataframe, to a FeatureUnion where each transform only returns the new columns. One of my transforms takes a third data set in addition to X and y which is used as the transform. I guess as a workaround I'll make it a param of the transform rather than a fit_arg.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/pipeline.py** | 903 | 932| 245 | 245 | 7824 | 
| **-> 2 <-** | **1 sklearn/pipeline.py** | 879 | 901| 138 | 383 | 7824 | 
| 3 | **1 sklearn/pipeline.py** | 835 | 859| 212 | 595 | 7824 | 
| 4 | **1 sklearn/pipeline.py** | 934 | 950| 178 | 773 | 7824 | 
| 5 | **1 sklearn/pipeline.py** | 985 | 1032| 380 | 1153 | 7824 | 
| 6 | **1 sklearn/pipeline.py** | 952 | 982| 271 | 1424 | 7824 | 
| 7 | **1 sklearn/pipeline.py** | 702 | 744| 285 | 1709 | 7824 | 
| 8 | **1 sklearn/pipeline.py** | 861 | 877| 137 | 1846 | 7824 | 
| 9 | 2 sklearn/compose/_column_transformer.py | 427 | 457| 284 | 2130 | 14124 | 
| 10 | **2 sklearn/pipeline.py** | 353 | 389| 304 | 2434 | 14124 | 
| 11 | 2 sklearn/compose/_column_transformer.py | 482 | 535| 453 | 2887 | 14124 | 
| 12 | 2 sklearn/compose/_column_transformer.py | 399 | 425| 257 | 3144 | 14124 | 
| 13 | **2 sklearn/pipeline.py** | 258 | 318| 543 | 3687 | 14124 | 
| 14 | 2 sklearn/compose/_column_transformer.py | 363 | 384| 184 | 3871 | 14124 | 
| 15 | **2 sklearn/pipeline.py** | 320 | 351| 241 | 4112 | 14124 | 
| 16 | 3 examples/compose/plot_feature_union.py | 1 | 60| 403 | 4515 | 14551 | 
| 17 | 3 sklearn/compose/_column_transformer.py | 459 | 480| 151 | 4666 | 14551 | 
| 18 | 4 examples/neighbors/approximate_nearest_neighbors.py | 156 | 190| 313 | 4979 | 17061 | 
| 19 | 4 sklearn/compose/_column_transformer.py | 36 | 166| 1354 | 6333 | 17061 | 
| 20 | **4 sklearn/pipeline.py** | 747 | 833| 677 | 7010 | 17061 | 
| 21 | 5 examples/compose/plot_column_transformer.py | 87 | 136| 379 | 7389 | 18057 | 
| 22 | **5 sklearn/pipeline.py** | 29 | 123| 993 | 8382 | 18057 | 
| 23 | 6 sklearn/impute/_base.py | 571 | 620| 362 | 8744 | 23546 | 
| 24 | 7 sklearn/neighbors/graph.py | 311 | 331| 158 | 8902 | 27695 | 
| 25 | 7 sklearn/compose/_column_transformer.py | 264 | 293| 220 | 9122 | 27695 | 
| 26 | 8 sklearn/preprocessing/data.py | 2747 | 2774| 294 | 9416 | 53248 | 
| 27 | 9 sklearn/utils/estimator_checks.py | 1157 | 1230| 650 | 10066 | 77781 | 
| 28 | **9 sklearn/pipeline.py** | 418 | 449| 270 | 10336 | 77781 | 
| 29 | **9 sklearn/pipeline.py** | 527 | 554| 190 | 10526 | 77781 | 
| 30 | 10 sklearn/base.py | 526 | 556| 218 | 10744 | 82760 | 
| 31 | 11 sklearn/feature_selection/from_model.py | 164 | 197| 285 | 11029 | 84561 | 
| 32 | 11 sklearn/utils/estimator_checks.py | 1129 | 1154| 287 | 11316 | 84561 | 
| 33 | 11 sklearn/neighbors/graph.py | 449 | 470| 158 | 11474 | 84561 | 
| 34 | **11 sklearn/pipeline.py** | 470 | 487| 131 | 11605 | 84561 | 
| 35 | 11 sklearn/compose/_column_transformer.py | 738 | 762| 239 | 11844 | 84561 | 
| 36 | 11 sklearn/compose/_column_transformer.py | 325 | 361| 273 | 12117 | 84561 | 
| 37 | 12 sklearn/preprocessing/_function_transformer.py | 99 | 117| 120 | 12237 | 85705 | 
| 38 | **12 sklearn/pipeline.py** | 391 | 416| 207 | 12444 | 85705 | 
| 39 | 13 sklearn/compose/_target.py | 107 | 149| 441 | 12885 | 87626 | 
| 40 | 13 sklearn/compose/_column_transformer.py | 227 | 262| 274 | 13159 | 87626 | 
| 41 | 13 sklearn/preprocessing/_function_transformer.py | 89 | 97| 129 | 13288 | 87626 | 
| 42 | 13 examples/compose/plot_column_transformer.py | 1 | 54| 371 | 13659 | 87626 | 
| 43 | 14 sklearn/cluster/_feature_agglomeration.py | 1 | 55| 369 | 14028 | 88172 | 
| 44 | **14 sklearn/pipeline.py** | 587 | 624| 278 | 14306 | 88172 | 
| 45 | **14 sklearn/pipeline.py** | 162 | 188| 248 | 14554 | 88172 | 
| 46 | 15 examples/ensemble/plot_feature_transformation.py | 1 | 87| 761 | 15315 | 89320 | 
| 47 | 16 sklearn/feature_extraction/text.py | 666 | 721| 420 | 15735 | 103964 | 
| 48 | 16 sklearn/impute/_base.py | 299 | 348| 442 | 16177 | 103964 | 
| 49 | 16 sklearn/compose/_column_transformer.py | 1 | 33| 213 | 16390 | 103964 | 
| 50 | **16 sklearn/pipeline.py** | 508 | 525| 134 | 16524 | 103964 | 
| 51 | 16 sklearn/preprocessing/data.py | 2776 | 2802| 215 | 16739 | 103964 | 
| 52 | 16 sklearn/impute/_base.py | 262 | 297| 317 | 17056 | 103964 | 
| 53 | 16 sklearn/compose/_target.py | 5 | 106| 813 | 17869 | 103964 | 
| 54 | 17 examples/compose/plot_column_transformer_mixed_types.py | 1 | 103| 797 | 18666 | 104783 | 
| 55 | 18 sklearn/impute/_iterative.py | 292 | 332| 502 | 19168 | 110802 | 
| 56 | **18 sklearn/pipeline.py** | 489 | 506| 125 | 19293 | 110802 | 
| 57 | 18 sklearn/preprocessing/data.py | 2638 | 2722| 726 | 20019 | 110802 | 
| 58 | **18 sklearn/pipeline.py** | 451 | 468| 131 | 20150 | 110802 | 
| 59 | 19 sklearn/feature_extraction/dict_vectorizer.py | 211 | 229| 145 | 20295 | 113489 | 
| 60 | **19 sklearn/pipeline.py** | 556 | 585| 225 | 20520 | 113489 | 
| 61 | 19 sklearn/preprocessing/data.py | 2724 | 2745| 143 | 20663 | 113489 | 
| 62 | 20 sklearn/ensemble/voting.py | 59 | 95| 323 | 20986 | 117417 | 
| 63 | 20 sklearn/feature_selection/from_model.py | 199 | 230| 239 | 21225 | 117417 | 
| 64 | 20 sklearn/compose/_column_transformer.py | 386 | 397| 142 | 21367 | 117417 | 
| 65 | 20 sklearn/preprocessing/data.py | 2322 | 2389| 677 | 22044 | 117417 | 
| 66 | 20 sklearn/preprocessing/data.py | 2941 | 2984| 389 | 22433 | 117417 | 
| 67 | 20 sklearn/feature_selection/from_model.py | 142 | 162| 208 | 22641 | 117417 | 
| 68 | 20 sklearn/preprocessing/_function_transformer.py | 119 | 161| 257 | 22898 | 117417 | 
| 69 | 20 sklearn/utils/estimator_checks.py | 1115 | 1126| 141 | 23039 | 117417 | 
| 70 | 20 sklearn/compose/_column_transformer.py | 167 | 225| 376 | 23415 | 117417 | 
| 71 | 21 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 24230 | 120525 | 
| 72 | 21 sklearn/impute/_iterative.py | 586 | 628| 401 | 24631 | 120525 | 
| 73 | 21 sklearn/compose/_column_transformer.py | 537 | 590| 477 | 25108 | 120525 | 
| 74 | 21 examples/neighbors/approximate_nearest_neighbors.py | 193 | 215| 200 | 25308 | 120525 | 
| 75 | 21 sklearn/feature_extraction/dict_vectorizer.py | 135 | 209| 604 | 25912 | 120525 | 
| 76 | 21 sklearn/preprocessing/data.py | 2391 | 2420| 323 | 26235 | 120525 | 
| 77 | 22 examples/compose/plot_transformed_target.py | 99 | 179| 749 | 26984 | 122357 | 
| 78 | 23 sklearn/ensemble/forest.py | 2167 | 2199| 299 | 27283 | 142122 | 
| 79 | 23 sklearn/preprocessing/data.py | 2455 | 2474| 166 | 27449 | 142122 | 
| 80 | 23 sklearn/compose/_target.py | 151 | 203| 425 | 27874 | 142122 | 
| 81 | 23 sklearn/impute/_iterative.py | 684 | 705| 128 | 28002 | 142122 | 
| 82 | 23 sklearn/impute/_iterative.py | 507 | 585| 647 | 28649 | 142122 | 
| 83 | 24 sklearn/feature_extraction/hashing.py | 95 | 126| 259 | 28908 | 143602 | 
| 84 | 25 examples/preprocessing/plot_function_transformer.py | 1 | 73| 450 | 29358 | 144052 | 
| 85 | 25 sklearn/cluster/_feature_agglomeration.py | 57 | 78| 166 | 29524 | 144052 | 
| 86 | 25 sklearn/compose/_column_transformer.py | 295 | 323| 279 | 29803 | 144052 | 
| 87 | 26 sklearn/kernel_approximation.py | 183 | 209| 209 | 30012 | 148861 | 
| 88 | 26 sklearn/feature_selection/from_model.py | 81 | 140| 591 | 30603 | 148861 | 
| 89 | 26 sklearn/kernel_approximation.py | 77 | 103| 191 | 30794 | 148861 | 
| 90 | 26 sklearn/utils/estimator_checks.py | 192 | 207| 187 | 30981 | 148861 | 
| 91 | 26 sklearn/utils/estimator_checks.py | 2549 | 2584| 398 | 31379 | 148861 | 
| 92 | 27 sklearn/manifold/isomap.py | 220 | 236| 119 | 31498 | 151041 | 
| 93 | 27 sklearn/utils/estimator_checks.py | 1262 | 1291| 318 | 31816 | 151041 | 
| 94 | 27 sklearn/kernel_approximation.py | 574 | 598| 165 | 31981 | 151041 | 
| 95 | 27 examples/compose/plot_transformed_target.py | 1 | 98| 743 | 32724 | 151041 | 
| 96 | 27 sklearn/feature_extraction/text.py | 473 | 779| 307 | 33031 | 151041 | 
| 97 | 28 benchmarks/bench_random_projections.py | 44 | 63| 122 | 33153 | 152776 | 
| 98 | 29 sklearn/feature_selection/base.py | 85 | 122| 347 | 33500 | 153682 | 
| 99 | 29 sklearn/manifold/isomap.py | 142 | 168| 221 | 33721 | 153682 | 
| 100 | 29 sklearn/preprocessing/data.py | 683 | 747| 572 | 34293 | 153682 | 
| 101 | **29 sklearn/pipeline.py** | 1 | 26| 130 | 34423 | 153682 | 
| 102 | 30 sklearn/decomposition/online_lda.py | 598 | 625| 225 | 34648 | 160283 | 
| 103 | 30 sklearn/preprocessing/data.py | 382 | 417| 228 | 34876 | 160283 | 
| 104 | 30 sklearn/compose/_column_transformer.py | 654 | 737| 826 | 35702 | 160283 | 
| 105 | 30 sklearn/preprocessing/data.py | 2987 | 3080| 822 | 36524 | 160283 | 
| 106 | 30 sklearn/preprocessing/data.py | 1515 | 1603| 783 | 37307 | 160283 | 
| 107 | 31 sklearn/ensemble/_stacking.py | 149 | 247| 833 | 38140 | 165948 | 
| 108 | 31 sklearn/compose/_column_transformer.py | 592 | 618| 222 | 38362 | 165948 | 
| 109 | 32 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 94| 785 | 39147 | 166932 | 
| 110 | 32 examples/compose/plot_transformed_target.py | 180 | 208| 312 | 39459 | 166932 | 
| 111 | 33 sklearn/isotonic.py | 222 | 248| 272 | 39731 | 170341 | 
| 112 | 34 sklearn/neighbors/nca.py | 375 | 452| 550 | 40281 | 174564 | 
| 113 | 34 sklearn/preprocessing/data.py | 420 | 493| 636 | 40917 | 174564 | 
| 114 | 34 sklearn/neighbors/nca.py | 266 | 335| 573 | 41490 | 174564 | 
| 115 | 35 examples/feature_selection/plot_feature_selection_pipeline.py | 1 | 41| 301 | 41791 | 174865 | 
| 116 | 35 sklearn/isotonic.py | 346 | 384| 286 | 42077 | 174865 | 
| 117 | 35 examples/neighbors/approximate_nearest_neighbors.py | 131 | 154| 219 | 42296 | 174865 | 
| 118 | 36 sklearn/neighbors/base.py | 1113 | 1167| 455 | 42751 | 184369 | 
| 119 | 37 sklearn/preprocessing/_encoders.py | 326 | 359| 207 | 42958 | 189787 | 
| 120 | 38 sklearn/cluster/hierarchical.py | 1012 | 1044| 242 | 43200 | 198786 | 


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


## Code snippets

### 1 - sklearn/pipeline.py:

Start line: 903, End line: 932

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs
```
### 2 - sklearn/pipeline.py:

Start line: 879, End line: 901

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : FeatureUnion
            This estimator
        """
        transformers = self._parallel_func(X, y, {}, _fit_one)
        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self
```
### 3 - sklearn/pipeline.py:

Start line: 835, End line: 859

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None or t == 'drop':
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform. '%s' (type %s) doesn't" %
                                (t, type(t)))

    def _iter(self):
        """
        Generate (name, trans, weight) tuples excluding None and
        'drop' transformers.
        """
        get_weight = (self.transformer_weights or {}).get
        return ((name, trans, get_weight(name))
                for name, trans in self.transformer_list
                if trans is not None and trans != 'drop')
```
### 4 - sklearn/pipeline.py:

Start line: 934, End line: 950

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(delayed(func)(
            transformer, X, y, weight,
            message_clsname='FeatureUnion',
            message=self._log_message(name, idx, len(transformers)),
            **fit_params) for idx, (name, transformer,
                                    weight) in enumerate(transformers, 1))
```
### 5 - sklearn/pipeline.py:

Start line: 985, End line: 1032

```python
def make_union(*transformers, **kwargs):
    """Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    ----------
    *transformers : list of estimators

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : boolean, optional(default=False)
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion

    See also
    --------
    sklearn.pipeline.FeatureUnion : Class for concatenating the results
        of multiple transformer objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())
     FeatureUnion(transformer_list=[('pca', PCA()),
                                   ('truncatedsvd', TruncatedSVD())])
    """
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return FeatureUnion(
        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
```
### 6 - sklearn/pipeline.py:

Start line: 952, End line: 982

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old is None or old == 'drop'
                                     else next(transformers))
                                    for name, old in self.transformer_list]
```
### 7 - sklearn/pipeline.py:

Start line: 702, End line: 744

```python
def _transform_one(transformer, X, y, weight, **fit_params):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X, y, **fit_params)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)

    if weight is None:
        return res, transformer
    return res * weight, transformer


def _fit_one(transformer,
             X,
             y,
             weight,
             message_clsname='',
             message=None,
             **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``.
    """
    with _print_elapsed_time(message_clsname, message):
        return transformer.fit(X, y, **fit_params)
```
### 8 - sklearn/pipeline.py:

Start line: 861, End line: 877

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def get_feature_names(self):
        """Get feature names from all transformers.

        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        feature_names = []
        for name, trans, weight in self._iter():
            if not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            feature_names.extend([name + "__" + f for f in
                                  trans.get_feature_names()])
        return feature_names
```
### 9 - sklearn/compose/_column_transformer.py:

Start line: 427, End line: 457

```python
class ColumnTransformer(TransformerMixin, _BaseComposition):

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(%d of %d) Processing %s' % (idx, total, name)

    def _fit_transform(self, X, y, func, fitted=False):
        """
        Private function to fit and/or transform on demand.

        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.
        """
        transformers = list(
            self._iter(fitted=fitted, replace_strings=True))
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(
                    transformer=clone(trans) if not fitted else trans,
                    X=safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname='ColumnTransformer',
                    message=self._log_message(name, idx, len(transformers)))
                for idx, (name, trans, column, weight) in enumerate(
                        self._iter(fitted=fitted, replace_strings=True), 1))
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN)
            else:
                raise
```
### 10 - sklearn/pipeline.py:

Start line: 353, End line: 389

```python
class Pipeline(_BaseComposition):

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == 'passthrough':
                return Xt
            if hasattr(last_step, 'fit_transform'):
                return last_step.fit_transform(Xt, y, **fit_params)
            else:
                return last_step.fit(Xt, y, **fit_params).transform(Xt)
```
### 13 - sklearn/pipeline.py:

Start line: 258, End line: 318

```python
class Pipeline(_BaseComposition):

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transfomer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, X, y, None,
                message_clsname='Pipeline',
                message=self._log_message(step_idx),
                **fit_params_steps[name])
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return X, {}
        return X, fit_params_steps[self.steps[-1][0]]
```
### 15 - sklearn/pipeline.py:

Start line: 320, End line: 351

```python
class Pipeline(_BaseComposition):

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                self._final_estimator.fit(Xt, y, **fit_params)
        return self
```
### 20 - sklearn/pipeline.py:

Start line: 747, End line: 833

```python
class FeatureUnion(TransformerMixin, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop' or ``None``.

    Read more in the :ref:`User Guide <feature_union>`.

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    verbose : boolean, optional(default=False)
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    See also
    --------
    sklearn.pipeline.make_union : convenience function for simplified
        feature union construction.

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[ 1.5       ,  3.0...,  0.8...],
           [-1.5       ,  5.7..., -0.4...]])
    """
    _required_parameters = ["transformer_list"]

    def __init__(self, transformer_list, n_jobs=None,
                 transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self._validate_transformers()

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('transformer_list', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('transformer_list', **kwargs)
        return self
```
### 22 - sklearn/pipeline.py:

Start line: 29, End line: 123

```python
class Pipeline(_BaseComposition):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Read more in the :ref:`User Guide <pipeline>`.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    See also
    --------
    sklearn.pipeline.make_pipeline : convenience function for simplified
        pipeline construction.

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.datasets import samples_generator
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.feature_selection import f_regression
    >>> from sklearn.pipeline import Pipeline
    >>> # generate some data to play with
    >>> X, y = samples_generator.make_classification(
    ...     n_informative=5, n_redundant=0, random_state=42)
    >>> # ANOVA SVM-C
    >>> anova_filter = SelectKBest(f_regression, k=5)
    >>> clf = svm.SVC(kernel='linear')
    >>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
    >>> # You can set the parameters using the names issued
    >>> # For instance, fit using a k of 10 in the SelectKBest
    >>> # and a parameter 'C' of the svm
    >>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
    Pipeline(steps=[('anova', SelectKBest(...)), ('svc', SVC(...))])
    >>> prediction = anova_svm.predict(X)
    >>> anova_svm.score(X, y)
    0.83
    >>> # getting the selected features chosen by anova_filter
    >>> anova_svm['anova'].get_support()
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Another way to get selected features chosen by anova_filter
    >>> anova_svm.named_steps.anova.get_support()
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Indexing can also be used to extract a sub-pipeline.
    >>> sub_pipeline = anova_svm[:1]
    >>> sub_pipeline
    Pipeline(steps=[('anova', SelectKBest(...))])
    >>> coef = anova_svm[-1].coef_
    >>> anova_svm['svc'] is anova_svm[-1]
    True
    >>> coef.shape
    (1, 10)
    >>> sub_pipeline.inverse_transform(coef).shape
    (1, 20)
    """
```
### 28 - sklearn/pipeline.py:

Start line: 418, End line: 449

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        Xt, fit_params = self._fit(X, y, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][-1].fit_predict(Xt, y, **fit_params)
        return y_pred
```
### 29 - sklearn/pipeline.py:

Start line: 527, End line: 554

```python
class Pipeline(_BaseComposition):

    @property
    def transform(self):
        """Apply transforms, and transform with the final estimator

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        # XXX: Handling the None case means we can't use if_delegate_has_method
        if self._final_estimator != 'passthrough':
            self._final_estimator.transform
        return self._transform

    def _transform(self, X):
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.transform(Xt)
        return Xt
```
### 34 - sklearn/pipeline.py:

Start line: 470, End line: 487

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)
```
### 38 - sklearn/pipeline.py:

Start line: 391, End line: 416

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

        Returns
        -------
        y_pred : array-like
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)
```
### 44 - sklearn/pipeline.py:

Start line: 587, End line: 624

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)

    @property
    def classes_(self):
        return self.steps[-1][-1].classes_

    @property
    def _pairwise(self):
        # check if first estimator expects pairwise input
        return getattr(self.steps[0][1], '_pairwise', False)
```
### 45 - sklearn/pipeline.py:

Start line: 162, End line: 188

```python
class Pipeline(_BaseComposition):

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == 'passthrough':
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform "
                                "or be the string 'passthrough' "
                                "'%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if (estimator is not None and estimator != 'passthrough'
                and not hasattr(estimator, "fit")):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator)))
```
### 50 - sklearn/pipeline.py:

Start line: 508, End line: 525

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)
```
### 56 - sklearn/pipeline.py:

Start line: 489, End line: 506

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def score_samples(self, X):
        """Apply transforms, and score_samples of the final estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray, shape (n_samples,)
        """
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        return self.steps[-1][-1].score_samples(Xt)
```
### 58 - sklearn/pipeline.py:

Start line: 451, End line: 468

```python
class Pipeline(_BaseComposition):

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)
```
### 60 - sklearn/pipeline.py:

Start line: 556, End line: 585

```python
class Pipeline(_BaseComposition):

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order

        All estimators in the pipeline must support ``inverse_transform``.

        Parameters
        ----------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_features]
        """
        # raise AttributeError if necessary for hasattr behaviour
        # XXX: Handling the None case means we can't use if_delegate_has_method
        for _, _, transform in self._iter():
            transform.inverse_transform
        return self._inverse_transform

    def _inverse_transform(self, X):
        Xt = X
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            Xt = transform.inverse_transform(Xt)
        return Xt
```
### 101 - sklearn/pipeline.py:

Start line: 1, End line: 26

```python
"""
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"""

from collections import defaultdict
from itertools import islice

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed

from .base import clone, TransformerMixin
from .utils.metaestimators import if_delegate_has_method
from .utils import Bunch, _print_elapsed_time
from .utils.validation import check_memory

from .utils.metaestimators import _BaseComposition

__all__ = ['Pipeline', 'FeatureUnion', 'make_pipeline', 'make_union']
```
