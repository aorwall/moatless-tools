# scikit-learn__scikit-learn-12462

* repo: scikit-learn/scikit-learn
* base_commit: `9ec5a15823dcb924a5cca322f9f97357f9428345`

## Problem statement

SkLearn `.score()` method generating error with Dask DataFrames
When using Dask Dataframes with SkLearn, I used to be able to just ask SkLearn for the score of any given algorithm. It would spit out a nice answer and I'd move on. After updating to the newest versions, all metrics that compute based on (y_true, y_predicted) are failing. I've tested `accuracy_score`, `precision_score`, `r2_score`, and `mean_squared_error.` Work-around shown below, but it's not ideal because it requires me to cast from Dask Arrays to numpy arrays which won't work if the data is huge.

I've asked Dask about it here: https://github.com/dask/dask/issues/4137 and they've said it's an issue with the SkLearn `shape` check, and that they won't be addressing it. It seems like it should be not super complicated to add a `try-except` that says "if shape doesn't return a tuple revert to pretending shape didn't exist". If others think that sounds correct, I can attempt a pull-request, but I don't want to attempt to solve it on my own only to find out others don't deem that an acceptable solutions.

Trace, MWE, versions, and workaround all in-line.

MWE:

```
import dask.dataframe as dd
from sklearn.linear_model import LinearRegression, SGDRegressor

df = dd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
lr = LinearRegression()
X = df.drop('quality', axis=1)
y = df['quality']

lr.fit(X,y)
lr.score(X,y)
```

Output of error:

```
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-5-4eafa0e7fc85> in <module>
      8 
      9 lr.fit(X,y)
---> 10 lr.score(X,y)

~/anaconda3/lib/python3.6/site-packages/sklearn/base.py in score(self, X, y, sample_weight)
    327         from .metrics import r2_score
    328         return r2_score(y, self.predict(X), sample_weight=sample_weight,
--> 329                         multioutput='variance_weighted')
    330 
    331 

~/anaconda3/lib/python3.6/site-packages/sklearn/metrics/regression.py in r2_score(y_true, y_pred, sample_weight, multioutput)
    532     """
    533     y_type, y_true, y_pred, multioutput = _check_reg_targets(
--> 534         y_true, y_pred, multioutput)
    535     check_consistent_length(y_true, y_pred, sample_weight)
    536 

~/anaconda3/lib/python3.6/site-packages/sklearn/metrics/regression.py in _check_reg_targets(y_true, y_pred, multioutput)
     73 
     74     """
---> 75     check_consistent_length(y_true, y_pred)
     76     y_true = check_array(y_true, ensure_2d=False)
     77     y_pred = check_array(y_pred, ensure_2d=False)

~/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    225 
    226     lengths = [_num_samples(X) for X in arrays if X is not None]
--> 227     uniques = np.unique(lengths)
    228     if len(uniques) > 1:
    229         raise ValueError("Found input variables with inconsistent numbers of"

~/anaconda3/lib/python3.6/site-packages/numpy/lib/arraysetops.py in unique(ar, return_index, return_inverse, return_counts, axis)
    229 
    230     """
--> 231     ar = np.asanyarray(ar)
    232     if axis is None:
    233         ret = _unique1d(ar, return_index, return_inverse, return_counts)

~/anaconda3/lib/python3.6/site-packages/numpy/core/numeric.py in asanyarray(a, dtype, order)
    551 
    552     """
--> 553     return array(a, dtype, copy=False, order=order, subok=True)
    554 
    555 

TypeError: int() argument must be a string, a bytes-like object or a number, not 'Scalar'
```

Problem occurs after upgrading as follows:

Before bug:
```
for lib in (sklearn, dask):
    print(f'{lib.__name__} Version: {lib.__version__}')
> sklearn Version: 0.19.1
> dask Version: 0.18.2
```

Update from conda, then bug starts:
```
for lib in (sklearn, dask):
    print(f'{lib.__name__} Version: {lib.__version__}')
> sklearn Version: 0.20.0
> dask Version: 0.19.4
```

Work around:

```
from sklearn.metrics import r2_score
preds = lr.predict(X_test)
r2_score(np.array(y_test), np.array(preds))
```


## Patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -140,7 +140,12 @@ def _num_samples(x):
         if len(x.shape) == 0:
             raise TypeError("Singleton array %r cannot be considered"
                             " a valid collection." % x)
-        return x.shape[0]
+        # Check that shape is returning an integer or default to len
+        # Dask dataframes may not return numeric shape[0] value
+        if isinstance(x.shape[0], numbers.Integral):
+            return x.shape[0]
+        else:
+            return len(x)
     else:
         return len(x)
 

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -41,6 +41,7 @@
     check_memory,
     check_non_negative,
     LARGE_SPARSE_SUPPORTED,
+    _num_samples
 )
 import sklearn
 
@@ -786,3 +787,15 @@ def test_check_X_y_informative_error():
     X = np.ones((2, 2))
     y = None
     assert_raise_message(ValueError, "y cannot be None", check_X_y, X, y)
+
+
+def test_retrieve_samples_from_non_standard_shape():
+    class TestNonNumericShape:
+        def __init__(self):
+            self.shape = ("not numeric",)
+
+        def __len__(self):
+            return len([1, 2, 3])
+
+    X = TestNonNumericShape()
+    assert _num_samples(X) == len(X)

```
