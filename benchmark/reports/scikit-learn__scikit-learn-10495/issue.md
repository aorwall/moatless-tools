# scikit-learn__scikit-learn-10495

* repo: scikit-learn/scikit-learn
* base_commit: `d6aa098dadc5eddca5287e823cacef474ac0d23f`

## Problem statement

check_array(X, dtype='numeric') should fail if X has strings
Currently, dtype='numeric' is defined as "dtype is preserved unless array.dtype is object". This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that `check_array(['a', 'b', 'c'], dtype='numeric')` works without error and produces an array of strings! This behaviour is not tested and it's hard to believe that it is useful and intended. Perhaps we need a deprecation cycle, but I think dtype='numeric' should raise an error, or attempt to coerce, if the data does not actually have a numeric, real-valued dtype. 
check_array(X, dtype='numeric') should fail if X has strings
Currently, dtype='numeric' is defined as "dtype is preserved unless array.dtype is object". This seems overly lenient and strange behaviour, as in #9342 where @qinhanmin2014 shows that `check_array(['a', 'b', 'c'], dtype='numeric')` works without error and produces an array of strings! This behaviour is not tested and it's hard to believe that it is useful and intended. Perhaps we need a deprecation cycle, but I think dtype='numeric' should raise an error, or attempt to coerce, if the data does not actually have a numeric, real-valued dtype. 


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
