# scikit-learn__scikit-learn-10495

| **scikit-learn/scikit-learn** | `d6aa098dadc5eddca5287e823cacef474ac0d23f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1293 |
| **Any found context length** | 1293 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -516,6 +516,15 @@ def check_array(array, accept_sparse=False, dtype="numeric", order=None,
             # To ensure that array flags are maintained
             array = np.array(array, dtype=dtype, order=order, copy=copy)
 
+        # in the future np.flexible dtypes will be handled like object dtypes
+        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
+            warnings.warn(
+                "Beginning in version 0.22, arrays of strings will be "
+                "interpreted as decimal numbers if parameter 'dtype' is "
+                "'numeric'. It is recommended that you convert the array to "
+                "type np.float64 before passing it to check_array.",
+                FutureWarning)
+
         # make sure we actually converted to numeric:
         if dtype_numeric and array.dtype.kind == "O":
             array = array.astype(np.float64)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/validation.py | 519 | 519 | 2 | 1 | 1293


## Problem Statement

```
check_array(X, dtype='numeric') should fail if X has strings
Currently, dtype='numeric' is defined as "dtype is preserved unless array.dtype is object". This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that `check_array(['a', 'b', 'c'], dtype='numeric')` works without error and produces an array of strings! This behaviour is not tested and it's hard to believe that it is useful and intended. Perhaps we need a deprecation cycle, but I think dtype='numeric' should raise an error, or attempt to coerce, if the data does not actually have a numeric, real-valued dtype. 
check_array(X, dtype='numeric') should fail if X has strings
Currently, dtype='numeric' is defined as "dtype is preserved unless array.dtype is object". This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that `check_array(['a', 'b', 'c'], dtype='numeric')` works without error and produces an array of strings! This behaviour is not tested and it's hard to believe that it is useful and intended. Perhaps we need a deprecation cycle, but I think dtype='numeric' should raise an error, or attempt to coerce, if the data does not actually have a numeric, real-valued dtype. 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/utils/validation.py** | 431 | 475| 454 | 454 | 7263 | 
| **-> 2 <-** | **1 sklearn/utils/validation.py** | 477 | 550| 839 | 1293 | 7263 | 
| 3 | **1 sklearn/utils/validation.py** | 654 | 668| 205 | 1498 | 7263 | 
| 4 | **1 sklearn/utils/validation.py** | 345 | 430| 849 | 2347 | 7263 | 
| 5 | 2 sklearn/utils/estimator_checks.py | 508 | 543| 328 | 2675 | 26593 | 
| 6 | **2 sklearn/utils/validation.py** | 65 | 111| 465 | 3140 | 26593 | 
| 7 | 3 sklearn/utils/sparsefuncs.py | 6 | 26| 182 | 3322 | 29927 | 
| 8 | **3 sklearn/utils/validation.py** | 34 | 62| 267 | 3589 | 29927 | 
| 9 | **3 sklearn/utils/validation.py** | 553 | 653| 979 | 4568 | 29927 | 
| 10 | 4 sklearn/metrics/pairwise.py | 58 | 125| 664 | 5232 | 41459 | 
| 11 | 4 sklearn/utils/estimator_checks.py | 1890 | 1904| 211 | 5443 | 41459 | 
| 12 | 4 sklearn/utils/estimator_checks.py | 366 | 422| 300 | 5743 | 41459 | 
| 13 | 4 sklearn/utils/estimator_checks.py | 1907 | 1927| 200 | 5943 | 41459 | 
| 14 | **4 sklearn/utils/validation.py** | 114 | 139| 222 | 6165 | 41459 | 
| 15 | 4 sklearn/metrics/pairwise.py | 128 | 159| 302 | 6467 | 41459 | 
| 16 | **4 sklearn/utils/validation.py** | 748 | 800| 469 | 6936 | 41459 | 
| 17 | 5 sklearn/feature_extraction/dict_vectorizer.py | 5 | 101| 807 | 7743 | 44187 | 
| 18 | **5 sklearn/utils/validation.py** | 206 | 221| 127 | 7870 | 44187 | 
| 19 | 6 sklearn/mixture/base.py | 41 | 64| 208 | 8078 | 47752 | 
| 20 | 6 sklearn/utils/estimator_checks.py | 811 | 833| 248 | 8326 | 47752 | 
| 21 | 6 sklearn/metrics/pairwise.py | 33 | 55| 157 | 8483 | 47752 | 
| 22 | 7 sklearn/preprocessing/data.py | 1568 | 1609| 386 | 8869 | 73650 | 
| 23 | **7 sklearn/utils/validation.py** | 250 | 342| 776 | 9645 | 73650 | 
| 24 | **7 sklearn/utils/validation.py** | 671 | 697| 181 | 9826 | 73650 | 
| 25 | 8 sklearn/utils/__init__.py | 346 | 371| 167 | 9993 | 77485 | 
| 26 | 8 sklearn/utils/estimator_checks.py | 425 | 466| 427 | 10420 | 77485 | 
| 27 | 8 sklearn/preprocessing/data.py | 1450 | 1480| 278 | 10698 | 77485 | 
| 28 | 9 sklearn/discriminant_analysis.py | 734 | 769| 250 | 10948 | 84116 | 
| 29 | 9 sklearn/utils/estimator_checks.py | 1027 | 1095| 601 | 11549 | 84116 | 
| 30 | 10 sklearn/utils/testing.py | 248 | 265| 164 | 11713 | 90947 | 
| 31 | 11 sklearn/gaussian_process/kernels.py | 36 | 44| 115 | 11828 | 106199 | 
| 32 | 11 sklearn/utils/estimator_checks.py | 978 | 1001| 269 | 12097 | 106199 | 
| 33 | 11 sklearn/utils/estimator_checks.py | 1494 | 1523| 324 | 12421 | 106199 | 
| 34 | 12 sklearn/utils/mocking.py | 1 | 40| 246 | 12667 | 106827 | 
| 35 | 12 sklearn/utils/__init__.py | 77 | 99| 120 | 12787 | 106827 | 
| 36 | 12 sklearn/utils/estimator_checks.py | 836 | 910| 702 | 13489 | 106827 | 
| 37 | 12 sklearn/utils/testing.py | 231 | 245| 131 | 13620 | 106827 | 
| 38 | 13 sklearn/tree/tree.py | 373 | 389| 180 | 13800 | 119785 | 
| 39 | 13 sklearn/preprocessing/data.py | 2869 | 2956| 936 | 14736 | 119785 | 
| 40 | 13 sklearn/utils/estimator_checks.py | 143 | 163| 203 | 14939 | 119785 | 
| 41 | 14 sklearn/decomposition/dict_learning.py | 255 | 309| 479 | 15418 | 130469 | 
| 42 | 14 sklearn/preprocessing/data.py | 2386 | 2403| 192 | 15610 | 130469 | 
| 43 | 14 sklearn/preprocessing/data.py | 141 | 194| 588 | 16198 | 130469 | 
| 44 | 14 sklearn/preprocessing/data.py | 1544 | 1565| 229 | 16427 | 130469 | 
| 45 | 15 sklearn/cluster/birch.py | 22 | 37| 142 | 16569 | 135760 | 
| 46 | 16 sklearn/preprocessing/imputation.py | 4 | 33| 160 | 16729 | 138786 | 
| 47 | 16 sklearn/preprocessing/data.py | 1669 | 1690| 210 | 16939 | 138786 | 
| 48 | **16 sklearn/utils/validation.py** | 224 | 247| 163 | 17102 | 138786 | 
| 49 | 16 sklearn/decomposition/dict_learning.py | 107 | 167| 632 | 17734 | 138786 | 
| 50 | 16 sklearn/preprocessing/data.py | 2061 | 2115| 502 | 18236 | 138786 | 
| 51 | 17 sklearn/utils/fixes.py | 74 | 150| 753 | 18989 | 141182 | 
| 52 | 18 sklearn/decomposition/online_lda.py | 454 | 507| 355 | 19344 | 147841 | 
| 53 | 19 sklearn/preprocessing/_function_transformer.py | 100 | 119| 126 | 19470 | 149229 | 
| 54 | 19 sklearn/utils/estimator_checks.py | 780 | 795| 139 | 19609 | 149229 | 
| 55 | 19 sklearn/utils/estimator_checks.py | 77 | 111| 269 | 19878 | 149229 | 
| 56 | 19 sklearn/utils/estimator_checks.py | 1867 | 1887| 233 | 20111 | 149229 | 
| 57 | 19 sklearn/utils/estimator_checks.py | 1221 | 1235| 146 | 20257 | 149229 | 
| 58 | 19 sklearn/utils/estimator_checks.py | 469 | 489| 253 | 20510 | 149229 | 
| 59 | 19 sklearn/feature_extraction/dict_vectorizer.py | 233 | 272| 320 | 20830 | 149229 | 
| 60 | 19 sklearn/utils/estimator_checks.py | 746 | 777| 297 | 21127 | 149229 | 
| 61 | 20 sklearn/metrics/cluster/supervised.py | 34 | 50| 171 | 21298 | 157139 | 
| 62 | 20 sklearn/preprocessing/imputation.py | 36 | 61| 240 | 21538 | 157139 | 
| 63 | 21 sklearn/isotonic.py | 6 | 75| 534 | 22072 | 160517 | 
| 64 | 22 sklearn/mixture/bayesian_mixture.py | 412 | 451| 419 | 22491 | 167776 | 
| 65 | 22 sklearn/preprocessing/data.py | 1385 | 1448| 562 | 23053 | 167776 | 
| 66 | 22 sklearn/utils/estimator_checks.py | 1004 | 1024| 257 | 23310 | 167776 | 
| 67 | 22 sklearn/feature_extraction/dict_vectorizer.py | 137 | 211| 606 | 23916 | 167776 | 
| 68 | **22 sklearn/utils/validation.py** | 803 | 869| 476 | 24392 | 167776 | 
| 69 | 22 sklearn/preprocessing/data.py | 86 | 140| 481 | 24873 | 167776 | 
| 70 | 23 doc/conf.py | 1 | 113| 843 | 25716 | 170187 | 
| 71 | 23 sklearn/utils/testing.py | 405 | 443| 348 | 26064 | 170187 | 
| 72 | 23 sklearn/preprocessing/data.py | 2765 | 2799| 305 | 26369 | 170187 | 
| 73 | 23 sklearn/preprocessing/data.py | 3027 | 3089| 523 | 26892 | 170187 | 
| 74 | 24 sklearn/gaussian_process/gpc.py | 629 | 674| 349 | 27241 | 177232 | 
| 75 | 24 sklearn/feature_extraction/dict_vectorizer.py | 274 | 318| 314 | 27555 | 177232 | 
| 76 | 24 sklearn/mixture/bayesian_mixture.py | 330 | 354| 236 | 27791 | 177232 | 
| 77 | 24 sklearn/utils/estimator_checks.py | 641 | 663| 221 | 28012 | 177232 | 
| 78 | 25 sklearn/utils/multiclass.py | 158 | 172| 137 | 28149 | 181098 | 
| 79 | 25 sklearn/utils/fixes.py | 1 | 71| 443 | 28592 | 181098 | 
| 80 | 26 sklearn/externals/joblib/numpy_pickle.py | 31 | 71| 323 | 28915 | 185949 | 
| 81 | 27 sklearn/utils/arpack.py | 1 | 24| 239 | 29154 | 186188 | 
| 82 | 27 sklearn/utils/testing.py | 117 | 159| 299 | 29453 | 186188 | 
| 83 | 27 sklearn/utils/multiclass.py | 108 | 155| 403 | 29856 | 186188 | 
| 84 | 27 sklearn/metrics/pairwise.py | 1224 | 1251| 286 | 30142 | 186188 | 
| 85 | 27 sklearn/utils/estimator_checks.py | 1164 | 1218| 567 | 30709 | 186188 | 
| 86 | 27 sklearn/utils/estimator_checks.py | 1382 | 1440| 489 | 31198 | 186188 | 
| 87 | 27 sklearn/preprocessing/data.py | 1999 | 2045| 490 | 31688 | 186188 | 
| 88 | 27 sklearn/preprocessing/data.py | 895 | 947| 429 | 32117 | 186188 | 
| 89 | 28 sklearn/impute.py | 36 | 61| 240 | 32357 | 189180 | 
| 90 | 29 sklearn/utils/extmath.py | 55 | 72| 131 | 32488 | 195416 | 
| 91 | 30 sklearn/dummy.py | 162 | 235| 595 | 33083 | 199068 | 


### Hint

```
ping @jnothman 

> This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that check_array(['a', 'b', 'c'], dtype='numeric') works without error and produces an array of strings!

I think you mean #9835 (https://github.com/scikit-learn/scikit-learn/pull/9835#issuecomment-348069380) ?

Yes, from my perspective, if it is not intended, it seems a bug.
it seems memorising four digit numbers is not my speciality

Five now!
And I'm not entirely sure what my intended behavior was, but I agree with your assessment. This should error on strings.
I think, @amueller, for the next little while, the mere knowledge that a
number has five digits leaves the first with distinctly low entropy

Well, I guess it's one more bit, though ;)
@jnothman I'd be happy to give this a go with some limited guidance if no one else is working on it already. Looks like the behavior you noted comes from [this line](https://github.com/scikit-learn/scikit-learn/blob/202b5321f1798c4980abf69ac8c0a0969f01a2ec/sklearn/utils/validation.py#L480), where we're checking the array against the numpy object type when we'd like to check it against string and unicode types as well -- the `[['a', 'b', 'c']]` list in your example appears to be cast to the numpy unicode array type in your example by the time it reaches that line. Sound right?
I'd also be curious to hear what you had in mind in terms of longer term solution, i.e., what would replace `check_array` if deprecated?
> Perhaps we need a deprecation cycle
Something like that. Basically if dtype_numeric and array.dtype is not an
object dtype or a numeric dtype, we should raise.

On 17 January 2018 at 13:17, Ryan <notifications@github.com> wrote:

> @jnothman <https://github.com/jnothman> I'd be happy to give this a go
> with some limited guidance if no one else is working on it already. Looks
> like the behavior you noted comes from this line
> <https://github.com/scikit-learn/scikit-learn/blob/202b5321f1798c4980abf69ac8c0a0969f01a2ec/sklearn/utils/validation.py#L480>,
> where we're checking the array against the numpy object type when we'd like
> to check it against string and unicode types as well -- the [['a', 'b',
> 'c']] list in your example appears to be cast to the numpy unicode array
> type in your example by the time it reaches that line. Sound right?
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/10229#issuecomment-358173231>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz69hcymywNoXaDwoNalOeRc93uF3Uks5tLVg8gaJpZM4QwJIl>
> .
>

We wouldn't deprecate `check_array` entirely, but we would warn for two releases that "In the future, this data with dtype('Uxx') would be rejected because it is not of a numeric dtype."
ping @jnothman 

> This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that check_array(['a', 'b', 'c'], dtype='numeric') works without error and produces an array of strings!

I think you mean #9835 (https://github.com/scikit-learn/scikit-learn/pull/9835#issuecomment-348069380) ?

Yes, from my perspective, if it is not intended, it seems a bug.
it seems memorising four digit numbers is not my speciality

Five now!
And I'm not entirely sure what my intended behavior was, but I agree with your assessment. This should error on strings.
I think, @amueller, for the next little while, the mere knowledge that a
number has five digits leaves the first with distinctly low entropy

Well, I guess it's one more bit, though ;)
@jnothman I'd be happy to give this a go with some limited guidance if no one else is working on it already. Looks like the behavior you noted comes from [this line](https://github.com/scikit-learn/scikit-learn/blob/202b5321f1798c4980abf69ac8c0a0969f01a2ec/sklearn/utils/validation.py#L480), where we're checking the array against the numpy object type when we'd like to check it against string and unicode types as well -- the `[['a', 'b', 'c']]` list in your example appears to be cast to the numpy unicode array type in your example by the time it reaches that line. Sound right?
I'd also be curious to hear what you had in mind in terms of longer term solution, i.e., what would replace `check_array` if deprecated?
> Perhaps we need a deprecation cycle
Something like that. Basically if dtype_numeric and array.dtype is not an
object dtype or a numeric dtype, we should raise.

On 17 January 2018 at 13:17, Ryan <notifications@github.com> wrote:

> @jnothman <https://github.com/jnothman> I'd be happy to give this a go
> with some limited guidance if no one else is working on it already. Looks
> like the behavior you noted comes from this line
> <https://github.com/scikit-learn/scikit-learn/blob/202b5321f1798c4980abf69ac8c0a0969f01a2ec/sklearn/utils/validation.py#L480>,
> where we're checking the array against the numpy object type when we'd like
> to check it against string and unicode types as well -- the [['a', 'b',
> 'c']] list in your example appears to be cast to the numpy unicode array
> type in your example by the time it reaches that line. Sound right?
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/10229#issuecomment-358173231>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz69hcymywNoXaDwoNalOeRc93uF3Uks5tLVg8gaJpZM4QwJIl>
> .
>

We wouldn't deprecate `check_array` entirely, but we would warn for two releases that "In the future, this data with dtype('Uxx') would be rejected because it is not of a numeric dtype."
```

## Patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -516,6 +516,15 @@ def check_array(array, accept_sparse=False, dtype="numeric", order=None,
             # To ensure that array flags are maintained
             array = np.array(array, dtype=dtype, order=order, copy=copy)
 
+        # in the future np.flexible dtypes will be handled like object dtypes
+        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
+            warnings.warn(
+                "Beginning in version 0.22, arrays of strings will be "
+                "interpreted as decimal numbers if parameter 'dtype' is "
+                "'numeric'. It is recommended that you convert the array to "
+                "type np.float64 before passing it to check_array.",
+                FutureWarning)
+
         # make sure we actually converted to numeric:
         if dtype_numeric and array.dtype.kind == "O":
             array = array.astype(np.float64)

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -285,6 +285,42 @@ def test_check_array():
     result = check_array(X_no_array)
     assert_true(isinstance(result, np.ndarray))
 
+    # deprecation warning if string-like array with dtype="numeric"
+    X_str = [['a', 'b'], ['c', 'd']]
+    assert_warns_message(
+        FutureWarning,
+        "arrays of strings will be interpreted as decimal numbers if "
+        "parameter 'dtype' is 'numeric'. It is recommended that you convert "
+        "the array to type np.float64 before passing it to check_array.",
+        check_array, X_str, "numeric")
+    assert_warns_message(
+        FutureWarning,
+        "arrays of strings will be interpreted as decimal numbers if "
+        "parameter 'dtype' is 'numeric'. It is recommended that you convert "
+        "the array to type np.float64 before passing it to check_array.",
+        check_array, np.array(X_str, dtype='U'), "numeric")
+    assert_warns_message(
+        FutureWarning,
+        "arrays of strings will be interpreted as decimal numbers if "
+        "parameter 'dtype' is 'numeric'. It is recommended that you convert "
+        "the array to type np.float64 before passing it to check_array.",
+        check_array, np.array(X_str, dtype='S'), "numeric")
+
+    # deprecation warning if byte-like array with dtype="numeric"
+    X_bytes = [[b'a', b'b'], [b'c', b'd']]
+    assert_warns_message(
+        FutureWarning,
+        "arrays of strings will be interpreted as decimal numbers if "
+        "parameter 'dtype' is 'numeric'. It is recommended that you convert "
+        "the array to type np.float64 before passing it to check_array.",
+        check_array, X_bytes, "numeric")
+    assert_warns_message(
+        FutureWarning,
+        "arrays of strings will be interpreted as decimal numbers if "
+        "parameter 'dtype' is 'numeric'. It is recommended that you convert "
+        "the array to type np.float64 before passing it to check_array.",
+        check_array, np.array(X_bytes, dtype='V1'), "numeric")
+
 
 def test_check_array_pandas_dtype_object_conversion():
     # test that data-frame like objects with dtype object

```


## Code snippets

### 1 - sklearn/utils/validation.py:

Start line: 431, End line: 475

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""
    # ... other code
```
### 2 - sklearn/utils/validation.py:

Start line: 477, End line: 550

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    # ... other code

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.array(array, dtype=dtype, order=order, copy=copy)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happend, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array
```
### 3 - sklearn/utils/validation.py:

Start line: 654, End line: 668

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
```
### 4 - sklearn/utils/validation.py:

Start line: 345, End line: 430

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    """
    # ... other code
```
### 5 - sklearn/utils/estimator_checks.py:

Start line: 508, End line: 543

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    y = (X[:, 0] * 4).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    try:
        estimator.fit(X, y.astype(object))
    except Exception as e:
        if "Unknown label type" not in str(e):
            raise

    X[0, 0] = {'foo': 'bar'}
    msg = "argument must be a string or a number"
    assert_raises_regex(TypeError, msg, estimator.fit, X, y)


def check_complex_data(name, estimator_orig):
    # check that estimators raise an exception on providing complex data
    X = np.random.sample(10) + 1j * np.random.sample(10)
    X = X.reshape(-1, 1)
    y = np.random.sample(10) + 1j * np.random.sample(10)
    estimator = clone(estimator_orig)
    assert_raises_regex(ValueError, "Complex data not supported",
                        estimator.fit, X, y)
```
### 6 - sklearn/utils/validation.py:

Start line: 65, End line: 111

```python
def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        if X.dtype.kind in 'uib' and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        return X.astype(return_dtype)
```
### 7 - sklearn/utils/sparsefuncs.py:

Start line: 6, End line: 26

```python
import scipy.sparse as sp
import numpy as np

from .fixes import sparse_min_max
from .sparsefuncs_fast import (
    csr_mean_variance_axis0 as _csr_mean_var_axis0,
    csc_mean_variance_axis0 as _csc_mean_var_axis0,
    incr_mean_variance_axis0 as _incr_mean_var_axis0)


def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    input_type = X.format if sp.issparse(X) else type(X)
    err = "Expected a CSR or CSC sparse matrix, got %s." % input_type
    raise TypeError(err)


def _raise_error_wrong_axis(axis):
    if axis not in (0, 1):
        raise ValueError(
            "Unknown axis value: %d. Use 0 for rows, or 1 for columns" % axis)
```
### 8 - sklearn/utils/validation.py:

Start line: 34, End line: 62

```python
def _assert_all_finite(X, allow_nan=False):
    """Like assert_all_finite, but only for ndarray."""
    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    is_float = X.dtype.kind in 'fc'
    if is_float and np.isfinite(X.sum()):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(msg_err.format(type_err, X.dtype))


def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix

    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)
```
### 9 - sklearn/utils/validation.py:

Start line: 553, End line: 653

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2d and sparse y.  If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    # ... other code
```
### 10 - sklearn/metrics/pairwise.py:

Start line: 58, End line: 125

```python
def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

        .. versionadded:: 0.18

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y
```
### 14 - sklearn/utils/validation.py:

Start line: 114, End line: 139

```python
def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)
```
### 16 - sklearn/utils/validation.py:

Start line: 748, End line: 800

```python
def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.")
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array
```
### 18 - sklearn/utils/validation.py:

Start line: 206, End line: 221

```python
def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])
```
### 23 - sklearn/utils/validation.py:

Start line: 250, End line: 342

```python
def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]

    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data,
                               allow_nan=force_all_finite == 'allow-nan')

    return spmatrix


def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))
```
### 24 - sklearn/utils/validation.py:

Start line: 671, End line: 697

```python
def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))
```
### 48 - sklearn/utils/validation.py:

Start line: 224, End line: 247

```python
def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if sp.issparse(X):
            result.append(X.tocsr())
        elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length(*result)
    return result
```
### 68 - sklearn/utils/validation.py:

Start line: 803, End line: 869

```python
def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.

    whom : string
        Who passed X to this function.
    """
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)
```
