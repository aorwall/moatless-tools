# scikit-learn__scikit-learn-10443

| **scikit-learn/scikit-learn** | `48f3303bfc0be26136b98e9aa95dc3b3f916daff` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 27618 |
| **Any found context length** | 3393 |
| **Avg pos** | 209.0 |
| **Min pos** | 5 |
| **Max pos** | 62 |
| **Top file pos** | 2 |
| **Missing snippets** | 11 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/feature_extraction/text.py b/sklearn/feature_extraction/text.py
--- a/sklearn/feature_extraction/text.py
+++ b/sklearn/feature_extraction/text.py
@@ -11,7 +11,7 @@
 The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
 build feature vectors from text documents.
 """
-from __future__ import unicode_literals
+from __future__ import unicode_literals, division
 
 import array
 from collections import Mapping, defaultdict
@@ -19,6 +19,7 @@
 from operator import itemgetter
 import re
 import unicodedata
+import warnings
 
 import numpy as np
 import scipy.sparse as sp
@@ -29,7 +30,7 @@
 from ..preprocessing import normalize
 from .hashing import FeatureHasher
 from .stop_words import ENGLISH_STOP_WORDS
-from ..utils.validation import check_is_fitted
+from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
 from ..utils.fixes import sp_version
 
 __all__ = ['CountVectorizer',
@@ -573,7 +574,7 @@ def _document_frequency(X):
     if sp.isspmatrix_csr(X):
         return np.bincount(X.indices, minlength=X.shape[1])
     else:
-        return np.diff(sp.csc_matrix(X, copy=False).indptr)
+        return np.diff(X.indptr)
 
 
 class CountVectorizer(BaseEstimator, VectorizerMixin):
@@ -1117,11 +1118,14 @@ def fit(self, X, y=None):
         X : sparse matrix, [n_samples, n_features]
             a matrix of term/token counts
         """
+        X = check_array(X, accept_sparse=('csr', 'csc'))
         if not sp.issparse(X):
-            X = sp.csc_matrix(X)
+            X = sp.csr_matrix(X)
+        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
+
         if self.use_idf:
             n_samples, n_features = X.shape
-            df = _document_frequency(X)
+            df = _document_frequency(X).astype(dtype)
 
             # perform idf smoothing if required
             df += int(self.smooth_idf)
@@ -1129,9 +1133,11 @@ def fit(self, X, y=None):
 
             # log+1 instead of log makes sure terms with zero idf don't get
             # suppressed entirely.
-            idf = np.log(float(n_samples) / df) + 1.0
-            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
-                                        n=n_features, format='csr')
+            idf = np.log(n_samples / df) + 1
+            self._idf_diag = sp.diags(idf, offsets=0,
+                                      shape=(n_features, n_features),
+                                      format='csr',
+                                      dtype=dtype)
 
         return self
 
@@ -1151,12 +1157,9 @@ def transform(self, X, copy=True):
         -------
         vectors : sparse matrix, [n_samples, n_features]
         """
-        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
-            # preserve float family dtype
-            X = sp.csr_matrix(X, copy=copy)
-        else:
-            # convert counts or binary occurrences to floats
-            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
+        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
+        if not sp.issparse(X):
+            X = sp.csr_matrix(X, dtype=np.float64)
 
         n_samples, n_features = X.shape
 
@@ -1367,7 +1370,7 @@ def __init__(self, input='content', encoding='utf-8',
                  stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                  ngram_range=(1, 1), max_df=1.0, min_df=1,
                  max_features=None, vocabulary=None, binary=False,
-                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
+                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                  sublinear_tf=False):
 
         super(TfidfVectorizer, self).__init__(
@@ -1432,6 +1435,13 @@ def idf_(self, value):
                                  (len(value), len(self.vocabulary)))
         self._tfidf.idf_ = value
 
+    def _check_params(self):
+        if self.dtype not in FLOAT_DTYPES:
+            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
+                          "be converted to np.float64."
+                          .format(FLOAT_DTYPES, self.dtype),
+                          UserWarning)
+
     def fit(self, raw_documents, y=None):
         """Learn vocabulary and idf from training set.
 
@@ -1444,6 +1454,7 @@ def fit(self, raw_documents, y=None):
         -------
         self : TfidfVectorizer
         """
+        self._check_params()
         X = super(TfidfVectorizer, self).fit_transform(raw_documents)
         self._tfidf.fit(X)
         return self
@@ -1464,6 +1475,7 @@ def fit_transform(self, raw_documents, y=None):
         X : sparse matrix, [n_samples, n_features]
             Tf-idf-weighted document-term matrix.
         """
+        self._check_params()
         X = super(TfidfVectorizer, self).fit_transform(raw_documents)
         self._tfidf.fit(X)
         # X is already a transformed view of raw_documents so

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/feature_extraction/text.py | 14 | 14 | 31 | 2 | 15396
| sklearn/feature_extraction/text.py | 22 | 22 | 31 | 2 | 15396
| sklearn/feature_extraction/text.py | 32 | 32 | 31 | 2 | 15396
| sklearn/feature_extraction/text.py | 576 | 576 | 62 | 2 | 27618
| sklearn/feature_extraction/text.py | 1120 | 1123 | 7 | 2 | 3813
| sklearn/feature_extraction/text.py | 1132 | 1134 | 7 | 2 | 3813
| sklearn/feature_extraction/text.py | 1154 | 1159 | 5 | 2 | 3393
| sklearn/feature_extraction/text.py | 1370 | 1370 | 8 | 2 | 4486
| sklearn/feature_extraction/text.py | 1435 | 1435 | 8 | 2 | 4486
| sklearn/feature_extraction/text.py | 1447 | 1447 | 8 | 2 | 4486
| sklearn/feature_extraction/text.py | 1467 | 1467 | 11 | 2 | 5352


## Problem Statement

```
TfidfVectorizer dtype argument ignored
#### Description
TfidfVectorizer's fit/fit_transform output is always np.float64 instead of the specified dtype

#### Steps/Code to Reproduce
\`\`\`py
from sklearn.feature_extraction.text import TfidfVectorizer
test = TfidfVectorizer(dtype=np.float32)
print(test.fit_transform(["Help I have a bug"]).dtype)
\`\`\`

#### Expected Results
\`\`\`py
dtype('float32')
\`\`\`

#### Actual Results
\`\`\`py
dtype('float64')
\`\`\`

#### Versions
\`\`\`
Darwin-17.2.0-x86_64-i386-64bit
Python 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09) 
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
NumPy 1.13.3
SciPy 1.0.0
Scikit-Learn 0.19.0
\`\`\`
  

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/feature_extraction/dict_vectorizer.py | 137 | 211| 606 | 606 | 2719 | 
| 2 | **2 sklearn/feature_extraction/text.py** | 1183 | 1194| 132 | 738 | 14770 | 
| 3 | **2 sklearn/feature_extraction/text.py** | 1030 | 1103| 836 | 1574 | 14770 | 
| 4 | **2 sklearn/feature_extraction/text.py** | 1197 | 1362| 1486 | 3060 | 14770 | 
| **-> 5 <-** | **2 sklearn/feature_extraction/text.py** | 1138 | 1181| 333 | 3393 | 14770 | 
| 6 | 2 sklearn/feature_extraction/dict_vectorizer.py | 213 | 231| 144 | 3537 | 14770 | 
| **-> 7 <-** | **2 sklearn/feature_extraction/text.py** | 1105 | 1136| 276 | 3813 | 14770 | 
| **-> 8 <-** | **2 sklearn/feature_extraction/text.py** | 1364 | 1449| 673 | 4486 | 14770 | 
| 9 | **2 sklearn/feature_extraction/text.py** | 1473 | 1497| 174 | 4660 | 14770 | 
| 10 | 3 benchmarks/bench_text_vectorizers.py | 1 | 77| 532 | 5192 | 15302 | 
| **-> 11 <-** | **3 sklearn/feature_extraction/text.py** | 1451 | 1471| 160 | 5352 | 15302 | 
| 12 | 3 sklearn/feature_extraction/dict_vectorizer.py | 274 | 318| 314 | 5666 | 15302 | 
| 13 | 3 sklearn/feature_extraction/dict_vectorizer.py | 5 | 101| 798 | 6464 | 15302 | 
| 14 | **3 sklearn/feature_extraction/text.py** | 273 | 299| 230 | 6694 | 15302 | 
| 15 | 4 sklearn/metrics/pairwise.py | 33 | 55| 157 | 6851 | 26871 | 
| 16 | 5 examples/text/plot_hashing_vs_dict_vectorizer.py | 1 | 112| 802 | 7653 | 27690 | 
| 17 | **5 sklearn/feature_extraction/text.py** | 467 | 512| 377 | 8030 | 27690 | 
| 18 | 6 sklearn/preprocessing/imputation.py | 173 | 251| 675 | 8705 | 30632 | 
| 19 | 7 sklearn/utils/estimator_checks.py | 520 | 555| 328 | 9033 | 50224 | 
| 20 | 8 sklearn/impute.py | 164 | 235| 575 | 9608 | 57383 | 
| 21 | **8 sklearn/feature_extraction/text.py** | 729 | 760| 319 | 9927 | 57383 | 
| 22 | 8 sklearn/feature_extraction/dict_vectorizer.py | 233 | 272| 320 | 10247 | 57383 | 
| 23 | 9 sklearn/datasets/twenty_newsgroups.py | 279 | 362| 770 | 11017 | 60652 | 
| 24 | **9 sklearn/feature_extraction/text.py** | 301 | 316| 146 | 11163 | 60652 | 
| 25 | 9 sklearn/utils/estimator_checks.py | 852 | 926| 702 | 11865 | 60652 | 
| 26 | 9 sklearn/preprocessing/imputation.py | 253 | 296| 438 | 12303 | 60652 | 
| 27 | 10 sklearn/externals/_pilutil.py | 69 | 140| 678 | 12981 | 65257 | 
| 28 | 11 sklearn/utils/validation.py | 65 | 111| 465 | 13446 | 72640 | 
| 29 | **11 sklearn/feature_extraction/text.py** | 579 | 727| 1362 | 14808 | 72640 | 
| 30 | 11 sklearn/impute.py | 237 | 281| 415 | 15223 | 72640 | 
| **-> 31 <-** | **11 sklearn/feature_extraction/text.py** | 1 | 41| 173 | 15396 | 72640 | 
| 32 | 12 sklearn/ensemble/forest.py | 1902 | 1935| 306 | 15702 | 89632 | 
| 33 | **12 sklearn/feature_extraction/text.py** | 319 | 466| 1341 | 17043 | 89632 | 
| 34 | 13 sklearn/utils/sparsefuncs.py | 6 | 26| 182 | 17225 | 92964 | 
| 35 | 14 sklearn/utils/fixes.py | 1 | 71| 443 | 17668 | 95680 | 
| 36 | 15 sklearn/datasets/svmlight_format.py | 167 | 189| 273 | 17941 | 100191 | 
| 37 | 15 sklearn/feature_extraction/dict_vectorizer.py | 103 | 135| 213 | 18154 | 100191 | 
| 38 | 16 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 18787 | 100917 | 
| 39 | 17 sklearn/isotonic.py | 364 | 389| 211 | 18998 | 104295 | 
| 40 | 18 sklearn/decomposition/fastica_.py | 471 | 508| 316 | 19314 | 109059 | 
| 41 | 18 sklearn/utils/estimator_checks.py | 827 | 849| 248 | 19562 | 109059 | 
| 42 | 18 sklearn/utils/estimator_checks.py | 994 | 1017| 269 | 19831 | 109059 | 
| 43 | 18 sklearn/utils/estimator_checks.py | 1043 | 1111| 601 | 20432 | 109059 | 
| 44 | 19 sklearn/feature_extraction/hashing.py | 88 | 100| 139 | 20571 | 110596 | 
| 45 | 19 sklearn/preprocessing/imputation.py | 129 | 171| 382 | 20953 | 110596 | 
| 46 | 20 sklearn/preprocessing/data.py | 2401 | 2420| 215 | 21168 | 136626 | 
| 47 | 20 sklearn/preprocessing/data.py | 2465 | 2484| 161 | 21329 | 136626 | 
| 48 | 21 benchmarks/bench_covertype.py | 71 | 97| 265 | 21594 | 138529 | 
| 49 | 21 sklearn/externals/_pilutil.py | 43 | 66| 176 | 21770 | 138529 | 
| 50 | 22 doc/tutorial/text_analytics/skeletons/exercise_02_sentiment.py | 14 | 64| 441 | 22211 | 139063 | 
| 51 | 22 sklearn/impute.py | 718 | 798| 731 | 22942 | 139063 | 
| 52 | 22 sklearn/utils/fixes.py | 74 | 150| 754 | 23696 | 139063 | 
| 53 | 22 sklearn/utils/validation.py | 549 | 565| 230 | 23926 | 139063 | 
| 54 | 23 sklearn/utils/testing.py | 249 | 266| 164 | 24090 | 145982 | 
| 55 | 23 sklearn/utils/estimator_checks.py | 758 | 789| 297 | 24387 | 145982 | 
| 56 | 24 sklearn/manifold/t_sne.py | 713 | 806| 896 | 25283 | 154469 | 
| 57 | 24 sklearn/preprocessing/data.py | 2978 | 3045| 541 | 25824 | 154469 | 
| 58 | 25 benchmarks/bench_random_projections.py | 48 | 67| 122 | 25946 | 156229 | 
| 59 | **25 sklearn/feature_extraction/text.py** | 952 | 982| 199 | 26145 | 156229 | 
| 60 | 26 sklearn/decomposition/pca.py | 724 | 765| 399 | 26544 | 163428 | 
| 61 | 26 sklearn/feature_extraction/hashing.py | 4 | 86| 802 | 27346 | 163428 | 
| **-> 62 <-** | **26 sklearn/feature_extraction/text.py** | 544 | 576| 272 | 27618 | 163428 | 
| 63 | 26 sklearn/impute.py | 128 | 162| 263 | 27881 | 163428 | 
| 64 | 27 sklearn/decomposition/online_lda.py | 617 | 635| 146 | 28027 | 170086 | 
| 65 | 27 sklearn/decomposition/online_lda.py | 586 | 615| 249 | 28276 | 170086 | 
| 66 | 27 sklearn/preprocessing/data.py | 2422 | 2430| 119 | 28395 | 170086 | 
| 67 | 28 examples/preprocessing/plot_power_transformer.py | 1 | 84| 619 | 29014 | 170939 | 
| 68 | 29 examples/preprocessing/plot_all_scaling.py | 322 | 356| 287 | 29301 | 174041 | 
| 69 | 30 sklearn/preprocessing/_function_transformer.py | 167 | 185| 170 | 29471 | 175429 | 
| 70 | 31 sklearn/pipeline.py | 575 | 596| 164 | 29635 | 181722 | 
| 71 | 31 sklearn/utils/estimator_checks.py | 792 | 807| 139 | 29774 | 181722 | 
| 72 | 32 examples/bicluster/plot_bicluster_newsgroups.py | 1 | 89| 759 | 30533 | 183105 | 
| 73 | 33 benchmarks/bench_plot_nmf.py | 1 | 47| 298 | 30831 | 187018 | 
| 74 | 34 sklearn/decomposition/nmf.py | 481 | 506| 247 | 31078 | 198712 | 


## Patch

```diff
diff --git a/sklearn/feature_extraction/text.py b/sklearn/feature_extraction/text.py
--- a/sklearn/feature_extraction/text.py
+++ b/sklearn/feature_extraction/text.py
@@ -11,7 +11,7 @@
 The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
 build feature vectors from text documents.
 """
-from __future__ import unicode_literals
+from __future__ import unicode_literals, division
 
 import array
 from collections import Mapping, defaultdict
@@ -19,6 +19,7 @@
 from operator import itemgetter
 import re
 import unicodedata
+import warnings
 
 import numpy as np
 import scipy.sparse as sp
@@ -29,7 +30,7 @@
 from ..preprocessing import normalize
 from .hashing import FeatureHasher
 from .stop_words import ENGLISH_STOP_WORDS
-from ..utils.validation import check_is_fitted
+from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
 from ..utils.fixes import sp_version
 
 __all__ = ['CountVectorizer',
@@ -573,7 +574,7 @@ def _document_frequency(X):
     if sp.isspmatrix_csr(X):
         return np.bincount(X.indices, minlength=X.shape[1])
     else:
-        return np.diff(sp.csc_matrix(X, copy=False).indptr)
+        return np.diff(X.indptr)
 
 
 class CountVectorizer(BaseEstimator, VectorizerMixin):
@@ -1117,11 +1118,14 @@ def fit(self, X, y=None):
         X : sparse matrix, [n_samples, n_features]
             a matrix of term/token counts
         """
+        X = check_array(X, accept_sparse=('csr', 'csc'))
         if not sp.issparse(X):
-            X = sp.csc_matrix(X)
+            X = sp.csr_matrix(X)
+        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
+
         if self.use_idf:
             n_samples, n_features = X.shape
-            df = _document_frequency(X)
+            df = _document_frequency(X).astype(dtype)
 
             # perform idf smoothing if required
             df += int(self.smooth_idf)
@@ -1129,9 +1133,11 @@ def fit(self, X, y=None):
 
             # log+1 instead of log makes sure terms with zero idf don't get
             # suppressed entirely.
-            idf = np.log(float(n_samples) / df) + 1.0
-            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
-                                        n=n_features, format='csr')
+            idf = np.log(n_samples / df) + 1
+            self._idf_diag = sp.diags(idf, offsets=0,
+                                      shape=(n_features, n_features),
+                                      format='csr',
+                                      dtype=dtype)
 
         return self
 
@@ -1151,12 +1157,9 @@ def transform(self, X, copy=True):
         -------
         vectors : sparse matrix, [n_samples, n_features]
         """
-        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
-            # preserve float family dtype
-            X = sp.csr_matrix(X, copy=copy)
-        else:
-            # convert counts or binary occurrences to floats
-            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
+        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
+        if not sp.issparse(X):
+            X = sp.csr_matrix(X, dtype=np.float64)
 
         n_samples, n_features = X.shape
 
@@ -1367,7 +1370,7 @@ def __init__(self, input='content', encoding='utf-8',
                  stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                  ngram_range=(1, 1), max_df=1.0, min_df=1,
                  max_features=None, vocabulary=None, binary=False,
-                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
+                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                  sublinear_tf=False):
 
         super(TfidfVectorizer, self).__init__(
@@ -1432,6 +1435,13 @@ def idf_(self, value):
                                  (len(value), len(self.vocabulary)))
         self._tfidf.idf_ = value
 
+    def _check_params(self):
+        if self.dtype not in FLOAT_DTYPES:
+            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
+                          "be converted to np.float64."
+                          .format(FLOAT_DTYPES, self.dtype),
+                          UserWarning)
+
     def fit(self, raw_documents, y=None):
         """Learn vocabulary and idf from training set.
 
@@ -1444,6 +1454,7 @@ def fit(self, raw_documents, y=None):
         -------
         self : TfidfVectorizer
         """
+        self._check_params()
         X = super(TfidfVectorizer, self).fit_transform(raw_documents)
         self._tfidf.fit(X)
         return self
@@ -1464,6 +1475,7 @@ def fit_transform(self, raw_documents, y=None):
         X : sparse matrix, [n_samples, n_features]
             Tf-idf-weighted document-term matrix.
         """
+        self._check_params()
         X = super(TfidfVectorizer, self).fit_transform(raw_documents)
         self._tfidf.fit(X)
         # X is already a transformed view of raw_documents so

```

## Test Patch

```diff
diff --git a/sklearn/feature_extraction/tests/test_text.py b/sklearn/feature_extraction/tests/test_text.py
--- a/sklearn/feature_extraction/tests/test_text.py
+++ b/sklearn/feature_extraction/tests/test_text.py
@@ -1,6 +1,9 @@
 from __future__ import unicode_literals
 import warnings
 
+import pytest
+from scipy import sparse
+
 from sklearn.feature_extraction.text import strip_tags
 from sklearn.feature_extraction.text import strip_accents_unicode
 from sklearn.feature_extraction.text import strip_accents_ascii
@@ -28,15 +31,14 @@
                                    assert_in, assert_less, assert_greater,
                                    assert_warns_message, assert_raise_message,
                                    clean_warning_registry, ignore_warnings,
-                                   SkipTest, assert_raises)
+                                   SkipTest, assert_raises,
+                                   assert_allclose_dense_sparse)
 
 from collections import defaultdict, Mapping
 from functools import partial
 import pickle
 from io import StringIO
 
-import pytest
-
 JUNK_FOOD_DOCS = (
     "the pizza pizza beer copyright",
     "the pizza burger beer copyright",
@@ -1042,6 +1044,42 @@ def test_vectorizer_string_object_as_input():
             ValueError, message, vec.transform, "hello world!")
 
 
+@pytest.mark.parametrize("X_dtype", [np.float32, np.float64])
+def test_tfidf_transformer_type(X_dtype):
+    X = sparse.rand(10, 20000, dtype=X_dtype, random_state=42)
+    X_trans = TfidfTransformer().fit_transform(X)
+    assert X_trans.dtype == X.dtype
+
+
+def test_tfidf_transformer_sparse():
+    X = sparse.rand(10, 20000, dtype=np.float64, random_state=42)
+    X_csc = sparse.csc_matrix(X)
+    X_csr = sparse.csr_matrix(X)
+
+    X_trans_csc = TfidfTransformer().fit_transform(X_csc)
+    X_trans_csr = TfidfTransformer().fit_transform(X_csr)
+    assert_allclose_dense_sparse(X_trans_csc, X_trans_csr)
+    assert X_trans_csc.format == X_trans_csr.format
+
+
+@pytest.mark.parametrize(
+    "vectorizer_dtype, output_dtype, expected_warning, msg_warning",
+    [(np.int32, np.float64, UserWarning, "'dtype' should be used."),
+     (np.int64, np.float64, UserWarning, "'dtype' should be used."),
+     (np.float32, np.float32, None, None),
+     (np.float64, np.float64, None, None)]
+)
+def test_tfidf_vectorizer_type(vectorizer_dtype, output_dtype,
+                               expected_warning, msg_warning):
+    X = np.array(["numpy", "scipy", "sklearn"])
+    vectorizer = TfidfVectorizer(dtype=vectorizer_dtype)
+    with pytest.warns(expected_warning, match=msg_warning) as record:
+            X_idf = vectorizer.fit_transform(X)
+    if expected_warning is None:
+        assert len(record) == 0
+    assert X_idf.dtype == output_dtype
+
+
 @pytest.mark.parametrize("vec", [
         HashingVectorizer(ngram_range=(2, 1)),
         CountVectorizer(ngram_range=(2, 1)),

```


## Code snippets

### 1 - sklearn/feature_extraction/dict_vectorizer.py:

Start line: 137, End line: 211

```python
class DictVectorizer(BaseEstimator, TransformerMixin):

    def _transform(self, X, fitting):
        # Sanity check: Python's array has no way of explicitly requesting the
        # signed 32-bit integers that scipy.sparse needs, so we use the next
        # best thing: typecode "i" (int). However, if that gives larger or
        # smaller integers than 32-bit ones, np.frombuffer screws up.
        assert array("i").itemsize == 4, (
            "sizeof(int) != 4 on your platform; please report this at"
            " https://github.com/scikit-learn/scikit-learn/issues and"
            " include the output from platform.platform() in your bug report")

        dtype = self.dtype
        if fitting:
            feature_names = []
            vocab = {}
        else:
            feature_names = self.feature_names_
            vocab = self.vocabulary_

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                if f in vocab:
                    indices.append(vocab[f])
                    values.append(dtype(v))
                else:
                    if fitting:
                        feature_names.append(f)
                        vocab[f] = len(vocab)
                        indices.append(vocab[f])
                        values.append(dtype(v))

            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        # Sort everything if asked
        if fitting and self.sort:
            feature_names.sort()
            map_index = np.empty(len(feature_names), dtype=np.int32)
            for new_val, f in enumerate(feature_names):
                map_index[new_val] = vocab[f]
                vocab[f] = new_val
            result_matrix = result_matrix[:, map_index]

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        if fitting:
            self.feature_names_ = feature_names
            self.vocabulary_ = vocab

        return result_matrix
```
### 2 - sklearn/feature_extraction/text.py:

Start line: 1183, End line: 1194

```python
class TfidfTransformer(BaseEstimator, TransformerMixin):

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')
```
### 3 - sklearn/feature_extraction/text.py:

Start line: 1030, End line: 1103

```python
class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-idf representation

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The formula that is used to compute the tf-idf of term t is
    tf-idf(d, t) = tf(t) * idf(d, t), and the idf is computed as
    idf(d, t) = log [ n / df(d, t) ] + 1 (if ``smooth_idf=False``),
    where n is the total number of documents and df(d, t) is the
    document frequency; the document frequency is the number of documents d
    that contain term t. The effect of adding "1" to the idf in the equation
    above is that terms with zero idf, i.e., terms  that occur in all documents
    in a training set, will not be entirely ignored.
    (Note that the idf formula above differs from the standard
    textbook notation that defines the idf as
    idf(d, t) = log [ n / (df(d, t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend
    on parameter settings that correspond to the SMART notation used in IR
    as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
    when ``norm=None``.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.

    References
    ----------

    .. [Yates2011] `R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.`

    .. [MRS2008] `C.D. Manning, P. Raghavan and H. Schütze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.`
    """
```
### 4 - sklearn/feature_extraction/text.py:

Start line: 1197, End line: 1362

```python
class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to CountVectorizer followed by TfidfTransformer.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean, default=False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    idf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    TfidfTransformer
        Apply Term Frequency Inverse Document Frequency normalization to a
        sparse matrix of occurrence counts.

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """
```
### 5 - sklearn/feature_extraction/text.py:

Start line: 1138, End line: 1181

```python
class TfidfTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X
```
### 6 - sklearn/feature_extraction/dict_vectorizer.py:

Start line: 213, End line: 231

```python
class DictVectorizer(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self._transform(X, fitting=True)
```
### 7 - sklearn/feature_extraction/text.py:

Start line: 1105, End line: 1136

```python
class TfidfTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self
```
### 8 - sklearn/feature_extraction/text.py:

Start line: 1364, End line: 1449

```python
class TfidfVectorizer(CountVectorizer):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        """
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self
```
### 9 - sklearn/feature_extraction/text.py:

Start line: 1473, End line: 1497

```python
class TfidfVectorizer(CountVectorizer):

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(TfidfVectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
```
### 10 - benchmarks/bench_text_vectorizers.py:

Start line: 1, End line: 77

```python
"""

To run this benchmark, you will need,

 * scikit-learn
 * pandas
 * memory_profiler
 * psutil (optional, but recommended)

"""

from __future__ import print_function

import timeit
import itertools

import numpy as np
import pandas as pd
from memory_profiler import memory_usage

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             HashingVectorizer)

n_repeat = 3


def run_vectorizer(Vectorizer, X, **params):
    def f():
        vect = Vectorizer(**params)
        vect.fit_transform(X)
    return f


text = fetch_20newsgroups(subset='train').data

print("="*80 + '\n#' + "    Text vectorizers benchmark" + '\n' + '='*80 + '\n')
print("Using a subset of the 20 newsrgoups dataset ({} documents)."
      .format(len(text)))
print("This benchmarks runs in ~20 min ...")

res = []

for Vectorizer, (analyzer, ngram_range) in itertools.product(
            [CountVectorizer, TfidfVectorizer, HashingVectorizer],
            [('word', (1, 1)),
             ('word', (1, 2)),
             ('word', (1, 4)),
             ('char', (4, 4)),
             ('char_wb', (4, 4))
             ]):

    bench = {'vectorizer': Vectorizer.__name__}
    params = {'analyzer': analyzer, 'ngram_range': ngram_range}
    bench.update(params)
    dt = timeit.repeat(run_vectorizer(Vectorizer, text, **params),
                       number=1,
                       repeat=n_repeat)
    bench['time'] = "{:.2f} (+-{:.2f})".format(np.mean(dt), np.std(dt))

    mem_usage = memory_usage(run_vectorizer(Vectorizer, text, **params))

    bench['memory'] = "{:.1f}".format(np.max(mem_usage))

    res.append(bench)


df = pd.DataFrame(res).set_index(['analyzer', 'ngram_range', 'vectorizer'])

print('\n========== Run time performance (sec) ===========\n')
print('Computing the mean and the standard deviation '
      'of the run time over {} runs...\n'.format(n_repeat))
print(df['time'].unstack(level=-1))

print('\n=============== Memory usage (MB) ===============\n')
print(df['memory'].unstack(level=-1))
```
### 11 - sklearn/feature_extraction/text.py:

Start line: 1451, End line: 1471

```python
class TfidfVectorizer(CountVectorizer):

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)
```
### 14 - sklearn/feature_extraction/text.py:

Start line: 273, End line: 299

```python
class VectorizerMixin(object):

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in xrange(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False
```
### 17 - sklearn/feature_extraction/text.py:

Start line: 467, End line: 512

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 non_negative=False, dtype=np.float64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.non_negative = non_negative
        self.dtype = dtype

    def partial_fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        """
        return self

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless."""
        # triggers a parameter validation
        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()

        self._get_hasher().fit(X, y=y)
        return self
```
### 21 - sklearn/feature_extraction/text.py:

Start line: 729, End line: 760

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
```
### 24 - sklearn/feature_extraction/text.py:

Start line: 301, End line: 316

```python
class VectorizerMixin(object):

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fit-ed)"""
        msg = "%(name)s - Vocabulary wasn't fitted."
        check_is_fitted(self, 'vocabulary_', msg=msg),

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    def _validate_params(self):
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError(
                "Invalid value for ngram_range=%s "
                "lower boundary larger than the upper boundary."
                % str(self.ngram_range))
```
### 29 - sklearn/feature_extraction/text.py:

Start line: 579, End line: 727

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    See also
    --------
    HashingVectorizer, TfidfVectorizer

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """
```
### 31 - sklearn/feature_extraction/text.py:

Start line: 1, End line: 41

```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import array
from collections import Mapping, defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six.moves import xrange
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils.validation import check_is_fitted
from ..utils.fixes import sp_version

__all__ = ['CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags']
```
### 33 - sklearn/feature_extraction/text.py:

Start line: 319, End line: 466

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    """Convert a collection of text documents to a matrix of token occurrences

    It turns a collection of text documents into a scipy.sparse matrix holding
    token occurrence counts (or binary occurrence information), possibly
    normalized as token frequencies if norm='l1' or projected on the euclidean
    unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:

    - it is very low memory scalable to large datasets as there is no need to
      store a vocabulary dictionary in memory

    - it is fast to pickle and un-pickle as it holds no state besides the
      constructor parameters

    - it can be used in a streaming (partial fit) or parallel pipeline as there
      is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):

    - there is no way to compute the inverse transform (from feature indices to
      string feature names) which can be a problem when trying to introspect
      which features are most important to a model.

    - there can be collisions: distinct tokens can be mapped to the same
      feature index. However in practice this is rarely an issue if n_features
      is large enough (e.g. 2 ** 18 for text classification problems).

    - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------

    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

    lowercase : boolean, default=True
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    n_features : integer, default=(2 ** 20)
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    binary : boolean, default=False.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    alternate_sign : boolean, optional, default True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.

        .. versionadded:: 0.19

    non_negative : boolean, optional, default False
        When True, an absolute value is applied to the features matrix prior to
        returning it. When used in conjunction with alternate_sign=True, this
        significantly reduces the inner product preservation property.

        .. deprecated:: 0.19
            This option will be removed in 0.21.

    See also
    --------
    CountVectorizer, TfidfVectorizer

    """
```
### 59 - sklearn/feature_extraction/text.py:

Start line: 952, End line: 982

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X
```
### 62 - sklearn/feature_extraction/text.py:

Start line: 544, End line: 576

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):

    def fit_transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Document-term matrix.
        """
        return self.fit(X, y).transform(X)

    def _get_hasher(self):
        return FeatureHasher(n_features=self.n_features,
                             input_type='string', dtype=self.dtype,
                             alternate_sign=self.alternate_sign,
                             non_negative=self.non_negative)


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
```
