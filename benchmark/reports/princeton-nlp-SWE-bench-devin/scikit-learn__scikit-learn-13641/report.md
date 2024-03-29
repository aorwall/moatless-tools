# scikit-learn__scikit-learn-13641

| **scikit-learn/scikit-learn** | `badaa153e67ffa56fb1a413b3b7b5b8507024291` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 16655 |
| **Any found context length** | 2209 |
| **Avg pos** | 125.0 |
| **Min pos** | 2 |
| **Max pos** | 37 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/feature_extraction/text.py b/sklearn/feature_extraction/text.py
--- a/sklearn/feature_extraction/text.py
+++ b/sklearn/feature_extraction/text.py
@@ -31,6 +31,7 @@
 from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
 from ..utils import _IS_32BIT
 from ..utils.fixes import _astype_copy_false
+from ..exceptions import ChangedBehaviorWarning
 
 
 __all__ = ['HashingVectorizer',
@@ -304,10 +305,34 @@ def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
             self._stop_words_id = id(self.stop_words)
             return 'error'
 
+    def _validate_custom_analyzer(self):
+        # This is to check if the given custom analyzer expects file or a
+        # filename instead of data.
+        # Behavior changed in v0.21, function could be removed in v0.23
+        import tempfile
+        with tempfile.NamedTemporaryFile() as f:
+            fname = f.name
+        # now we're sure fname doesn't exist
+
+        msg = ("Since v0.21, vectorizers pass the data to the custom analyzer "
+               "and not the file names or the file objects. This warning "
+               "will be removed in v0.23.")
+        try:
+            self.analyzer(fname)
+        except FileNotFoundError:
+            warnings.warn(msg, ChangedBehaviorWarning)
+        except AttributeError as e:
+            if str(e) == "'str' object has no attribute 'read'":
+                warnings.warn(msg, ChangedBehaviorWarning)
+        except Exception:
+            pass
+
     def build_analyzer(self):
         """Return a callable that handles preprocessing and tokenization"""
         if callable(self.analyzer):
-            return self.analyzer
+            if self.input in ['file', 'filename']:
+                self._validate_custom_analyzer()
+            return lambda doc: self.analyzer(self.decode(doc))
 
         preprocess = self.build_preprocessor()
 
@@ -490,6 +515,11 @@ class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     n_features : integer, default=(2 ** 20)
         The number of features (columns) in the output matrices. Small numbers
         of features are likely to cause hash collisions, but large numbers
@@ -745,6 +775,11 @@ class CountVectorizer(BaseEstimator, VectorizerMixin):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     max_df : float in range [0.0, 1.0] or int, default=1.0
         When building the vocabulary ignore terms that have a document
         frequency strictly higher than the given threshold (corpus-specific
@@ -1369,6 +1404,11 @@ class TfidfVectorizer(CountVectorizer):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     stop_words : string {'english'}, list, or None (default=None)
         If a string, it is passed to _check_stop_list and the appropriate stop
         list is returned. 'english' is currently the only supported string

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/feature_extraction/text.py | 34 | 34 | 36 | 1 | 16655
| sklearn/feature_extraction/text.py | 307 | 307 | 6 | 1 | 4643
| sklearn/feature_extraction/text.py | 493 | 493 | 15 | 1 | 9812
| sklearn/feature_extraction/text.py | 748 | 748 | 37 | 1 | 16940
| sklearn/feature_extraction/text.py | 1372 | 1372 | 25 | 1 | 13253


## Problem Statement

```
CountVectorizer with custom analyzer ignores input argument
Example:

\`\`\` py
cv = CountVectorizer(analyzer=lambda x: x.split(), input='filename')
cv.fit(['hello world']).vocabulary_
\`\`\`

Same for `input="file"`. Not sure if this should be fixed or just documented; I don't like changing the behavior of the vectorizers yet again...


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/feature_extraction/text.py** | 829 | 874| 429 | 429 | 13238 | 
| **-> 2 <-** | **1 sklearn/feature_extraction/text.py** | 1311 | 1498| 1780 | 2209 | 13238 | 
| 3 | **1 sklearn/feature_extraction/text.py** | 333 | 359| 225 | 2434 | 13238 | 
| **-> 4 <-** | **1 sklearn/feature_extraction/text.py** | 659 | 827| 1610 | 4044 | 13238 | 
| 5 | **1 sklearn/feature_extraction/text.py** | 535 | 589| 411 | 4455 | 13238 | 
| **-> 6 <-** | **1 sklearn/feature_extraction/text.py** | 307 | 331| 188 | 4643 | 13238 | 
| 7 | **1 sklearn/feature_extraction/text.py** | 361 | 376| 145 | 4788 | 13238 | 
| 8 | 2 sklearn/feature_extraction/dict_vectorizer.py | 135 | 209| 603 | 5391 | 15959 | 
| 9 | **2 sklearn/feature_extraction/text.py** | 116 | 145| 169 | 5560 | 15959 | 
| 10 | **2 sklearn/feature_extraction/text.py** | 229 | 268| 337 | 5897 | 15959 | 
| 11 | **2 sklearn/feature_extraction/text.py** | 1500 | 1593| 729 | 6626 | 15959 | 
| 12 | 3 examples/text/plot_hashing_vs_dict_vectorizer.py | 1 | 110| 794 | 7420 | 16770 | 
| 13 | 4 sklearn/model_selection/_split.py | 2008 | 2063| 473 | 7893 | 35838 | 
| 14 | 5 benchmarks/bench_text_vectorizers.py | 1 | 73| 517 | 8410 | 36355 | 
| **-> 15 <-** | **5 sklearn/feature_extraction/text.py** | 379 | 534| 1402 | 9812 | 36355 | 
| 16 | **5 sklearn/feature_extraction/text.py** | 917 | 975| 446 | 10258 | 36355 | 
| 17 | **5 sklearn/feature_extraction/text.py** | 876 | 915| 374 | 10632 | 36355 | 
| 18 | 6 sklearn/feature_extraction/hashing.py | 96 | 127| 262 | 10894 | 37841 | 
| 19 | **6 sklearn/feature_extraction/text.py** | 270 | 305| 303 | 11197 | 37841 | 
| 20 | **6 sklearn/feature_extraction/text.py** | 201 | 227| 233 | 11430 | 37841 | 
| 21 | 7 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 12063 | 38567 | 
| 22 | **7 sklearn/feature_extraction/text.py** | 1050 | 1080| 197 | 12260 | 38567 | 
| 23 | 7 sklearn/feature_extraction/dict_vectorizer.py | 211 | 229| 144 | 12404 | 38567 | 
| 24 | 8 sklearn/decomposition/dict_learning.py | 110 | 178| 692 | 13096 | 50124 | 
| **-> 25 <-** | **8 sklearn/feature_extraction/text.py** | 1293 | 1644| 157 | 13253 | 50124 | 
| 26 | **8 sklearn/feature_extraction/text.py** | 1618 | 1645| 191 | 13444 | 50124 | 
| 27 | 9 sklearn/preprocessing/_encoders.py | 532 | 544| 147 | 13591 | 58654 | 
| 28 | 9 sklearn/model_selection/_split.py | 1956 | 1980| 142 | 13733 | 58654 | 
| 29 | 9 sklearn/feature_extraction/dict_vectorizer.py | 318 | 368| 378 | 14111 | 58654 | 
| 30 | 10 doc/tutorial/text_analytics/skeletons/exercise_02_sentiment.py | 14 | 64| 441 | 14552 | 59188 | 
| 31 | 10 sklearn/feature_extraction/dict_vectorizer.py | 5 | 99| 783 | 15335 | 59188 | 
| 32 | 11 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 15716 | 60503 | 
| 33 | 12 sklearn/preprocessing/data.py | 2319 | 2338| 215 | 15931 | 85629 | 
| 34 | 12 sklearn/feature_extraction/dict_vectorizer.py | 101 | 133| 208 | 16139 | 85629 | 
| 35 | **12 sklearn/feature_extraction/text.py** | 1082 | 1128| 335 | 16474 | 85629 | 
| **-> 36 <-** | **12 sklearn/feature_extraction/text.py** | 1 | 43| 181 | 16655 | 85629 | 
| **-> 37 <-** | **12 sklearn/feature_extraction/text.py** | 621 | 1123| 285 | 16940 | 85629 | 
| 38 | 12 sklearn/feature_extraction/dict_vectorizer.py | 272 | 316| 311 | 17251 | 85629 | 
| 39 | **12 sklearn/feature_extraction/text.py** | 977 | 1048| 448 | 17699 | 85629 | 
| 40 | 13 examples/text/plot_document_classification_20newsgroups.py | 121 | 200| 689 | 18388 | 88168 | 
| 41 | 13 sklearn/preprocessing/data.py | 2870 | 2913| 388 | 18776 | 88168 | 
| 42 | **13 sklearn/feature_extraction/text.py** | 147 | 176| 237 | 19013 | 88168 | 
| 43 | **13 sklearn/feature_extraction/text.py** | 1595 | 1616| 161 | 19174 | 88168 | 
| 44 | 14 sklearn/model_selection/__init__.py | 1 | 60| 405 | 19579 | 88573 | 
| 45 | **14 sklearn/feature_extraction/text.py** | 178 | 199| 191 | 19770 | 88573 | 
| 46 | 15 sklearn/impute.py | 1229 | 1252| 254 | 20024 | 99711 | 
| 47 | 16 examples/compose/plot_column_transformer.py | 1 | 54| 370 | 20394 | 100684 | 
| 48 | 17 sklearn/utils/estimator_checks.py | 1897 | 1941| 472 | 20866 | 122559 | 
| 49 | 18 examples/applications/plot_out_of_core_classification.py | 187 | 214| 241 | 21107 | 125864 | 
| 50 | 19 examples/text/plot_document_clustering.py | 89 | 186| 713 | 21820 | 127742 | 
| 51 | 19 sklearn/preprocessing/data.py | 2340 | 2348| 119 | 21939 | 127742 | 
| 52 | 19 examples/text/plot_document_clustering.py | 1 | 88| 750 | 22689 | 127742 | 
| 53 | 20 sklearn/compose/_column_transformer.py | 591 | 641| 388 | 23077 | 134394 | 
| 54 | 21 sklearn/utils/validation.py | 600 | 707| 1053 | 24130 | 142611 | 
| 55 | 21 sklearn/utils/validation.py | 332 | 425| 924 | 25054 | 142611 | 
| 56 | 21 sklearn/utils/estimator_checks.py | 950 | 972| 248 | 25302 | 142611 | 
| 57 | 22 sklearn/datasets/twenty_newsgroups.py | 316 | 398| 742 | 26044 | 146299 | 
| 58 | 22 sklearn/preprocessing/_encoders.py | 676 | 710| 326 | 26370 | 146299 | 
| 59 | 22 sklearn/utils/estimator_checks.py | 2092 | 2106| 211 | 26581 | 146299 | 
| 60 | 22 sklearn/utils/estimator_checks.py | 1423 | 1522| 940 | 27521 | 146299 | 
| 61 | 22 sklearn/compose/_column_transformer.py | 719 | 735| 121 | 27642 | 146299 | 
| 62 | 22 sklearn/utils/estimator_checks.py | 1372 | 1403| 258 | 27900 | 146299 | 
| 63 | 22 examples/text/plot_document_classification_20newsgroups.py | 1 | 119| 783 | 28683 | 146299 | 
| 64 | 23 sklearn/linear_model/coordinate_descent.py | 2273 | 2288| 189 | 28872 | 167394 | 
| 65 | 23 sklearn/utils/estimator_checks.py | 1828 | 1870| 469 | 29341 | 167394 | 
| 66 | 23 sklearn/decomposition/dict_learning.py | 275 | 331| 495 | 29836 | 167394 | 
| 67 | 23 examples/compose/plot_column_transformer.py | 87 | 136| 358 | 30194 | 167394 | 
| 68 | 23 sklearn/preprocessing/_encoders.py | 632 | 674| 446 | 30640 | 167394 | 
| 69 | 24 sklearn/linear_model/stochastic_gradient.py | 110 | 144| 422 | 31062 | 181322 | 
| 70 | 25 doc/tutorial/text_analytics/solutions/exercise_01_language_train_model.py | 1 | 71| 485 | 31547 | 181831 | 
| 71 | 25 sklearn/preprocessing/_encoders.py | 72 | 105| 291 | 31838 | 181831 | 
| 72 | 25 sklearn/preprocessing/_encoders.py | 352 | 467| 1112 | 32950 | 181831 | 
| 73 | 25 sklearn/utils/estimator_checks.py | 1077 | 1102| 282 | 33232 | 181831 | 
| 74 | 25 sklearn/preprocessing/_encoders.py | 546 | 603| 606 | 33838 | 181831 | 
| 75 | 25 sklearn/preprocessing/_encoders.py | 107 | 852| 334 | 34172 | 181831 | 
| 76 | 25 sklearn/datasets/twenty_newsgroups.py | 400 | 449| 425 | 34597 | 181831 | 
| 77 | 25 sklearn/model_selection/_split.py | 1982 | 2005| 131 | 34728 | 181831 | 
| 78 | 25 sklearn/preprocessing/data.py | 1785 | 1854| 580 | 35308 | 181831 | 
| 79 | 25 sklearn/compose/_column_transformer.py | 262 | 291| 219 | 35527 | 181831 | 
| 80 | 26 benchmarks/bench_plot_randomized_svd.py | 111 | 128| 156 | 35683 | 186211 | 


### Hint

```
To be sure, the current docstring says:

\`\`\`
If a callable is passed it is used to extract the sequence of features
out of the raw, unprocessed input.
\`\`\`

"Unprocessed" seems to mean that even `input=` is ignored, but this is not obvious.

I'll readily agree that's the wrong behaviour even with that docstring.

On 20 October 2015 at 22:59, Lars notifications@github.com wrote:

> To be sure, the current docstring says:
> 
> \`\`\`
> If a callable is passed it is used to extract the sequence of features
> out of the raw, unprocessed input.
> \`\`\`
> 
> "Unprocessed" seems to mean that even input= is ignored, but this is not
> obvious.
> 
> —
> Reply to this email directly or view it on GitHub
> https://github.com/scikit-learn/scikit-learn/issues/5482#issuecomment-149541462
> .

I'm a new contributor, i'm interested to work on this issue. To be sure, what is expected is improving the docstring on that behavior ?

I'm not at all sure. The behavior is probably a bug, but it has stood for so long that it's very hard to fix without breaking someone's code.

@TTRh , did you have had some more thoughts on that? Otherwise, I will give it a shot and clarify how the input parameter is ought to be used vs. providing input in the fit method.

Please go ahead, i didn't produce anything on it !

@jnothman @larsmans commit is pending on the docstring side of things.. after looking at the code, I think one would need to introduce a parameter like preprocessing="none" to not break old code. If supplying a custom analyzer and using inbuilt preprocessing is no boundary case, this should become a feature request?

I'd be tempted to say that any user using `input='file'` or `input='filename'` who then passed text to `fit` or `transform` was doing something obviously wrong. That is, I think this is a bug that can be fixed without notice. However, the correct behaviour still requires some definition. If we load from file for the user, do we decode? Probably not. Which means behaviour will differ between Py 2/3. But that's the user's problem.


```

## Patch

```diff
diff --git a/sklearn/feature_extraction/text.py b/sklearn/feature_extraction/text.py
--- a/sklearn/feature_extraction/text.py
+++ b/sklearn/feature_extraction/text.py
@@ -31,6 +31,7 @@
 from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
 from ..utils import _IS_32BIT
 from ..utils.fixes import _astype_copy_false
+from ..exceptions import ChangedBehaviorWarning
 
 
 __all__ = ['HashingVectorizer',
@@ -304,10 +305,34 @@ def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
             self._stop_words_id = id(self.stop_words)
             return 'error'
 
+    def _validate_custom_analyzer(self):
+        # This is to check if the given custom analyzer expects file or a
+        # filename instead of data.
+        # Behavior changed in v0.21, function could be removed in v0.23
+        import tempfile
+        with tempfile.NamedTemporaryFile() as f:
+            fname = f.name
+        # now we're sure fname doesn't exist
+
+        msg = ("Since v0.21, vectorizers pass the data to the custom analyzer "
+               "and not the file names or the file objects. This warning "
+               "will be removed in v0.23.")
+        try:
+            self.analyzer(fname)
+        except FileNotFoundError:
+            warnings.warn(msg, ChangedBehaviorWarning)
+        except AttributeError as e:
+            if str(e) == "'str' object has no attribute 'read'":
+                warnings.warn(msg, ChangedBehaviorWarning)
+        except Exception:
+            pass
+
     def build_analyzer(self):
         """Return a callable that handles preprocessing and tokenization"""
         if callable(self.analyzer):
-            return self.analyzer
+            if self.input in ['file', 'filename']:
+                self._validate_custom_analyzer()
+            return lambda doc: self.analyzer(self.decode(doc))
 
         preprocess = self.build_preprocessor()
 
@@ -490,6 +515,11 @@ class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     n_features : integer, default=(2 ** 20)
         The number of features (columns) in the output matrices. Small numbers
         of features are likely to cause hash collisions, but large numbers
@@ -745,6 +775,11 @@ class CountVectorizer(BaseEstimator, VectorizerMixin):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     max_df : float in range [0.0, 1.0] or int, default=1.0
         When building the vocabulary ignore terms that have a document
         frequency strictly higher than the given threshold (corpus-specific
@@ -1369,6 +1404,11 @@ class TfidfVectorizer(CountVectorizer):
         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.
 
+        .. versionchanged:: 0.21
+        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
+        first read from the file and then passed to the given callable
+        analyzer.
+
     stop_words : string {'english'}, list, or None (default=None)
         If a string, it is passed to _check_stop_list and the appropriate stop
         list is returned. 'english' is currently the only supported string

```

## Test Patch

```diff
diff --git a/sklearn/feature_extraction/tests/test_text.py b/sklearn/feature_extraction/tests/test_text.py
--- a/sklearn/feature_extraction/tests/test_text.py
+++ b/sklearn/feature_extraction/tests/test_text.py
@@ -29,6 +29,7 @@
 from numpy.testing import assert_array_almost_equal
 from numpy.testing import assert_array_equal
 from sklearn.utils import IS_PYPY
+from sklearn.exceptions import ChangedBehaviorWarning
 from sklearn.utils.testing import (assert_equal, assert_not_equal,
                                    assert_almost_equal, assert_in,
                                    assert_less, assert_greater,
@@ -1196,3 +1197,47 @@ def build_preprocessor(self):
                                             .findall(doc),
                     stop_words=['and'])
     assert _check_stop_words_consistency(vec) is True
+
+
+@pytest.mark.parametrize('Estimator',
+                         [CountVectorizer, TfidfVectorizer, HashingVectorizer])
+@pytest.mark.parametrize(
+    'input_type, err_type, err_msg',
+    [('filename', FileNotFoundError, ''),
+     ('file', AttributeError, "'str' object has no attribute 'read'")]
+)
+def test_callable_analyzer_error(Estimator, input_type, err_type, err_msg):
+    data = ['this is text, not file or filename']
+    with pytest.raises(err_type, match=err_msg):
+        Estimator(analyzer=lambda x: x.split(),
+                  input=input_type).fit_transform(data)
+
+
+@pytest.mark.parametrize('Estimator',
+                         [CountVectorizer, TfidfVectorizer, HashingVectorizer])
+@pytest.mark.parametrize(
+    'analyzer', [lambda doc: open(doc, 'r'), lambda doc: doc.read()]
+)
+@pytest.mark.parametrize('input_type', ['file', 'filename'])
+def test_callable_analyzer_change_behavior(Estimator, analyzer, input_type):
+    data = ['this is text, not file or filename']
+    warn_msg = 'Since v0.21, vectorizer'
+    with pytest.raises((FileNotFoundError, AttributeError)):
+        with pytest.warns(ChangedBehaviorWarning, match=warn_msg) as records:
+            Estimator(analyzer=analyzer, input=input_type).fit_transform(data)
+    assert len(records) == 1
+    assert warn_msg in str(records[0])
+
+
+@pytest.mark.parametrize('Estimator',
+                         [CountVectorizer, TfidfVectorizer, HashingVectorizer])
+def test_callable_analyzer_reraise_error(tmpdir, Estimator):
+    # check if a custom exception from the analyzer is shown to the user
+    def analyzer(doc):
+        raise Exception("testing")
+
+    f = tmpdir.join("file.txt")
+    f.write("sample content\n")
+
+    with pytest.raises(Exception, match="testing"):
+        Estimator(analyzer=analyzer, input='file').fit_transform([f])

```


## Code snippets

### 1 - sklearn/feature_extraction/text.py:

Start line: 829, End line: 874

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

    def _sort_features(self, X, vocabulary):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X
```
### 2 - sklearn/feature_extraction/text.py:

Start line: 1311, End line: 1498

```python
class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TfidfTransformer`.

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

    decode_error : {'strict', 'ignore', 'replace'} (default='strict')
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default=None)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    stop_words : string {'english'}, list, or None (default=None)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=None)
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean (default=False)
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional (default=float64)
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`

    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean (default=True)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean (default=False)
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

    Examples
    --------
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.shape)
    (4, 9)

    See also
    --------
    CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

    TfidfTransformer : Performs the TF-IDF transformation from a provided
        matrix of counts.

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """
```
### 3 - sklearn/feature_extraction/text.py:

Start line: 333, End line: 359

```python
class VectorizerMixin:

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
                indices = set(vocabulary.values())
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
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
### 4 - sklearn/feature_extraction/text.py:

Start line: 659, End line: 827

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

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

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

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.toarray())  # doctest: +NORMALIZE_WHITESPACE
    [[0 1 1 1 0 0 1 0 1]
     [0 2 0 1 0 1 1 0 1]
     [1 0 0 1 1 0 1 1 1]
     [0 1 1 1 0 0 1 0 1]]

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
### 5 - sklearn/feature_extraction/text.py:

Start line: 535, End line: 589

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 dtype=np.float64):
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
        self.dtype = dtype

    def partial_fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        return self

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        # triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()

        self._get_hasher().fit(X, y=y)
        return self
```
### 6 - sklearn/feature_extraction/text.py:

Start line: 307, End line: 331

```python
class VectorizerMixin:

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self._check_stop_words_consistency(stop_words, preprocess,
                                               tokenize)
            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)
```
### 7 - sklearn/feature_extraction/text.py:

Start line: 361, End line: 376

```python
class VectorizerMixin:

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
### 8 - sklearn/feature_extraction/dict_vectorizer.py:

Start line: 135, End line: 209

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
            for f, v in x.items():
                if isinstance(v, str):
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
### 9 - sklearn/feature_extraction/text.py:

Start line: 116, End line: 145

```python
class VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        """Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.

        Parameters
        ----------
        doc : string
            The string to decode
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc
```
### 10 - sklearn/feature_extraction/text.py:

Start line: 229, End line: 268

```python
class VectorizerMixin:

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor

        # unfortunately python functools package does not have an efficient
        # `compose` function that would have allowed us to chain a dynamic
        # number of functions. However the cost of a lambda call is a few
        # hundreds of nanoseconds which is negligible when compared to the
        # cost of tokenizing a string of 1000 chars for instance.
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        if self.lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def get_stop_words(self):
        """Build or fetch the effective stop words list"""
        return _check_stop_list(self.stop_words)
```
### 11 - sklearn/feature_extraction/text.py:

Start line: 1500, End line: 1593

```python
class TfidfVectorizer(CountVectorizer):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super().__init__(
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

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

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
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self
```
### 15 - sklearn/feature_extraction/text.py:

Start line: 379, End line: 534

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

    lowercase : boolean, default=True
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    n_features : integer, default=(2 ** 20)
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.

    binary : boolean, default=False.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    alternate_sign : boolean, optional, default True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.

        .. versionadded:: 0.19

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    Examples
    --------
    >>> from sklearn.feature_extraction.text import HashingVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = HashingVectorizer(n_features=2**4)
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(X.shape)
    (4, 16)

    See also
    --------
    CountVectorizer, TfidfVectorizer

    """
```
### 16 - sklearn/feature_extraction/text.py:

Start line: 917, End line: 975

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(('sparse CSR array has {} non-zero '
                                  'elements and requires 64 bit indexing, '
                                  'which is unsupported with 32 bit Python.')
                                 .format(indptr[-1]))
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X
```
### 17 - sklearn/feature_extraction/text.py:

Start line: 876, End line: 915

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def _limit_features(self, X, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms
```
### 19 - sklearn/feature_extraction/text.py:

Start line: 270, End line: 305

```python
class VectorizerMixin:

    def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        """Check if stop words are consistent

        Returns
        -------
        is_consistent : True if stop words are consistent with the preprocessor
                        and tokenizer, False if they are not, None if the check
                        was previously performed, "error" if it could not be
                        performed (e.g. because of the use of a custom
                        preprocessor / tokenizer)
        """
        if id(self.stop_words) == getattr(self, '_stop_words_id', None):
            # Stop words are were previously validated
            return None

        # NB: stop_words is validated, unlike self.stop_words
        try:
            inconsistent = set()
            for w in stop_words or ():
                tokens = list(tokenize(preprocess(w)))
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            self._stop_words_id = id(self.stop_words)

            if inconsistent:
                warnings.warn('Your stop_words may be inconsistent with '
                              'your preprocessing. Tokenizing the stop '
                              'words generated tokens %r not in '
                              'stop_words.' % sorted(inconsistent))
            return not inconsistent
        except Exception:
            # Failed to check stop words consistency (e.g. because a custom
            # preprocessor or tokenizer was used)
            self._stop_words_id = id(self.stop_words)
            return 'error'
```
### 20 - sklearn/feature_extraction/text.py:

Start line: 201, End line: 227

```python
class VectorizerMixin:

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams
```
### 22 - sklearn/feature_extraction/text.py:

Start line: 1050, End line: 1080

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
        if isinstance(raw_documents, str):
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
### 25 - sklearn/feature_extraction/text.py:

Start line: 1293, End line: 1644

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

    def _more_tags(self):
        return {'X_types': 'sparse'}


class TfidfVectorizer(CountVectorizer):
```
### 26 - sklearn/feature_extraction/text.py:

Start line: 1618, End line: 1645

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

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {'X_types': ['string'], '_skip_test': True}
```
### 35 - sklearn/feature_extraction/text.py:

Start line: 1082, End line: 1128

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        self._check_vocabulary()

        if sp.issparse(X):
            # We need CSR format for fast row manipulations.
            X = X.tocsr()
        else:
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        return [t for t, i in sorted(self.vocabulary_.items(),
                                     key=itemgetter(1))]

    def _more_tags(self):
        return {'X_types': ['string']}


def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))
```
### 36 - sklearn/feature_extraction/text.py:

Start line: 1, End line: 43

```python
# -*- coding: utf-8 -*-

import array
from collections import defaultdict
from collections.abc import Mapping
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from ..utils import _IS_32BIT
from ..utils.fixes import _astype_copy_false


__all__ = ['HashingVectorizer',
           'CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags']
```
### 37 - sklearn/feature_extraction/text.py:

Start line: 621, End line: 1123

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
                             alternate_sign=self.alternate_sign)

    def _more_tags(self):
        return {'X_types': ['string']}


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


class CountVectorizer(BaseEstimator, VectorizerMixin):
```
### 39 - sklearn/feature_extraction/text.py:

Start line: 977, End line: 1048

```python
class CountVectorizer(BaseEstimator, VectorizerMixin):

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_params()
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X
```
### 42 - sklearn/feature_extraction/text.py:

Start line: 147, End line: 176

```python
class VectorizerMixin:

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens
```
### 43 - sklearn/feature_extraction/text.py:

Start line: 1595, End line: 1616

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
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)
```
### 45 - sklearn/feature_extraction/text.py:

Start line: 178, End line: 199

```python
class VectorizerMixin:

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams
```
