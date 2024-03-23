# scikit-learn__scikit-learn-12938

* repo: scikit-learn/scikit-learn
* base_commit: `acb810647233e40839203ac553429e8663169702`

## Problem statement

AttributeError: 'PrettyPrinter' object has no attribute '_indent_at_name'
There's a failing example in #12654, and here's a piece of code causing it:

```
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid, iid=False)
from tempfile import mkdtemp
from joblib import Memory

# Create a temporary folder to store the transformers of the pipeline
cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=10)
cached_pipe = Pipeline([('reduce_dim', PCA()),
                        ('classify', LinearSVC(dual=False, max_iter=10000))],
                       memory=memory)

# This time, a cached pipeline will be used within the grid search
grid = GridSearchCV(cached_pipe, cv=5, n_jobs=1, param_grid=param_grid,
                    iid=False, error_score='raise')
digits = load_digits()
grid.fit(digits.data, digits.target)
```

With the stack trace:

```
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/path/to//sklearn/model_selection/_search.py", line 683, in fit
    self._run_search(evaluate_candidates)
  File "/path/to//sklearn/model_selection/_search.py", line 1127, in _run_search
    evaluate_candidates(ParameterGrid(self.param_grid))
  File "/path/to//sklearn/model_selection/_search.py", line 672, in evaluate_candidates
    cv.split(X, y, groups)))
  File "/path/to//sklearn/externals/joblib/parallel.py", line 917, in __call__
    if self.dispatch_one_batch(iterator):
  File "/path/to//sklearn/externals/joblib/parallel.py", line 759, in dispatch_one_batch
    self._dispatch(tasks)
  File "/path/to//sklearn/externals/joblib/parallel.py", line 716, in _dispatch
    job = self._backend.apply_async(batch, callback=cb)
  File "/path/to//sklearn/externals/joblib/_parallel_backends.py", line 182, in apply_async
    result = ImmediateResult(func)
  File "/path/to//sklearn/externals/joblib/_parallel_backends.py", line 549, in __init__
    self.results = batch()
  File "/path/to//sklearn/externals/joblib/parallel.py", line 225, in __call__
    for func, args, kwargs in self.items]
  File "/path/to//sklearn/externals/joblib/parallel.py", line 225, in <listcomp>
    for func, args, kwargs in self.items]
  File "/path/to//sklearn/model_selection/_validation.py", line 511, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/path/to//sklearn/pipeline.py", line 279, in fit
    Xt, fit_params = self._fit(X, y, **fit_params)
  File "/path/to//sklearn/pipeline.py", line 244, in _fit
    **fit_params_steps[name])
  File "/path/to/packages/joblib/memory.py", line 555, in __call__
    return self._cached_call(args, kwargs)[0]
  File "/path/to/packages/joblib/memory.py", line 521, in _cached_call
    out, metadata = self.call(*args, **kwargs)
  File "/path/to/packages/joblib/memory.py", line 720, in call
    print(format_call(self.func, args, kwargs))
  File "/path/to/packages/joblib/func_inspect.py", line 356, in format_call
    path, signature = format_signature(func, *args, **kwargs)
  File "/path/to/packages/joblib/func_inspect.py", line 340, in format_signature
    formatted_arg = _format_arg(arg)
  File "/path/to/packages/joblib/func_inspect.py", line 322, in _format_arg
    formatted_arg = pformat(arg, indent=2)
  File "/path/to/packages/joblib/logger.py", line 54, in pformat
    out = pprint.pformat(obj, depth=depth, indent=indent)
  File "/usr/lib64/python3.7/pprint.py", line 58, in pformat
    compact=compact).pformat(object)
  File "/usr/lib64/python3.7/pprint.py", line 144, in pformat
    self._format(object, sio, 0, 0, {}, 0)
  File "/usr/lib64/python3.7/pprint.py", line 167, in _format
    p(self, object, stream, indent, allowance, context, level + 1)
  File "/path/to//sklearn/utils/_pprint.py", line 175, in _pprint_estimator
    if self._indent_at_name:
AttributeError: 'PrettyPrinter' object has no attribute '_indent_at_name'
```


## Patch

```diff
diff --git a/sklearn/utils/_pprint.py b/sklearn/utils/_pprint.py
--- a/sklearn/utils/_pprint.py
+++ b/sklearn/utils/_pprint.py
@@ -321,7 +321,10 @@ def _pprint_key_val_tuple(self, object, stream, indent, allowance, context,
         self._format(v, stream, indent + len(rep) + len(middle), allowance,
                      context, level)
 
-    _dispatch = pprint.PrettyPrinter._dispatch
+    # Note: need to copy _dispatch to prevent instances of the builtin
+    # PrettyPrinter class to call methods of _EstimatorPrettyPrinter (see issue
+    # 12906)
+    _dispatch = pprint.PrettyPrinter._dispatch.copy()
     _dispatch[BaseEstimator.__repr__] = _pprint_estimator
     _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple
 

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_pprint.py b/sklearn/utils/tests/test_pprint.py
--- a/sklearn/utils/tests/test_pprint.py
+++ b/sklearn/utils/tests/test_pprint.py
@@ -1,4 +1,5 @@
 import re
+from pprint import PrettyPrinter
 
 from sklearn.utils._pprint import _EstimatorPrettyPrinter
 from sklearn.pipeline import make_pipeline, Pipeline
@@ -311,3 +312,11 @@ def test_length_constraint():
     vectorizer = CountVectorizer(vocabulary=vocabulary)
     repr_ = vectorizer.__repr__()
     assert '...' in repr_
+
+
+def test_builtin_prettyprinter():
+    # non regression test than ensures we can still use the builtin
+    # PrettyPrinter class for estimators (as done e.g. by joblib).
+    # Used to be a bug
+
+    PrettyPrinter().pprint(LogisticRegression())

```
