# scikit-learn__scikit-learn-10443

* repo: scikit-learn/scikit-learn
* base_commit: `48f3303bfc0be26136b98e9aa95dc3b3f916daff`

## Problem statement

TfidfVectorizer dtype argument ignored
#### Description
TfidfVectorizer's fit/fit_transform output is always np.float64 instead of the specified dtype

#### Steps/Code to Reproduce
```py
from sklearn.feature_extraction.text import TfidfVectorizer
test = TfidfVectorizer(dtype=np.float32)
print(test.fit_transform(["Help I have a bug"]).dtype)
```

#### Expected Results
```py
dtype('float32')
```

#### Actual Results
```py
dtype('float64')
```

#### Versions
```
Darwin-17.2.0-x86_64-i386-64bit
Python 3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09) 
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
NumPy 1.13.3
SciPy 1.0.0
Scikit-Learn 0.19.0
```
  


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
