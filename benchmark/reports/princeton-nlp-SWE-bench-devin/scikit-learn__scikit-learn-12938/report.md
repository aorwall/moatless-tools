# scikit-learn__scikit-learn-12938

| **scikit-learn/scikit-learn** | `acb810647233e40839203ac553429e8663169702` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 12297 |
| **Any found context length** | 12297 |
| **Avg pos** | 20.0 |
| **Min pos** | 20 |
| **Max pos** | 20 |
| **Top file pos** | 4 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/_pprint.py | 324 | 324 | 20 | 4 | 12297


## Problem Statement

```
AttributeError: 'PrettyPrinter' object has no attribute '_indent_at_name'
There's a failing example in #12654, and here's a piece of code causing it:

\`\`\`
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
\`\`\`

With the stack trace:

\`\`\`
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
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/compose/plot_compare_reduction.py | 1 | 106| 824 | 824 | 1068 | 
| 2 | 2 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 1617 | 2914 | 
| 3 | 3 sklearn/utils/estimator_checks.py | 1 | 80| 698 | 2315 | 24312 | 
| 4 | **4 sklearn/utils/_pprint.py** | 168 | 197| 270 | 2585 | 28371 | 
| 5 | 5 sklearn/model_selection/__init__.py | 1 | 60| 405 | 2990 | 28776 | 
| 6 | 6 sklearn/model_selection/_search.py | 755 | 1124| 570 | 3560 | 41842 | 
| 7 | 6 examples/preprocessing/plot_discretization_classification.py | 109 | 193| 874 | 4434 | 41842 | 
| 8 | 7 examples/compose/plot_column_transformer_mixed_types.py | 1 | 106| 807 | 5241 | 42671 | 
| 9 | 8 benchmarks/bench_mnist.py | 109 | 181| 695 | 5936 | 44399 | 
| 10 | 9 examples/model_selection/grid_search_text_feature_extraction.py | 2 | 97| 663 | 6599 | 45496 | 
| 11 | 9 benchmarks/bench_mnist.py | 85 | 106| 314 | 6913 | 45496 | 
| 12 | 10 examples/model_selection/plot_grid_search_digits.py | 1 | 81| 637 | 7550 | 46133 | 
| 13 | 11 examples/text/plot_document_classification_20newsgroups.py | 250 | 326| 662 | 8212 | 48680 | 
| 14 | 12 benchmarks/bench_covertype.py | 113 | 191| 757 | 8969 | 50581 | 
| 15 | 13 examples/preprocessing/plot_scaling_importance.py | 1 | 82| 762 | 9731 | 51808 | 
| 16 | 13 examples/compose/plot_compare_reduction.py | 107 | 132| 225 | 9956 | 51808 | 
| 17 | 13 examples/preprocessing/plot_scaling_importance.py | 83 | 135| 457 | 10413 | 51808 | 
| 18 | 14 benchmarks/bench_random_projections.py | 88 | 254| 1264 | 11677 | 53558 | 
| 19 | 15 doc/tutorial/text_analytics/skeletons/exercise_02_sentiment.py | 14 | 64| 441 | 12118 | 54092 | 
| **-> 20 <-** | **15 sklearn/utils/_pprint.py** | 309 | 326| 179 | 12297 | 54092 | 
| 21 | 15 sklearn/model_selection/_search.py | 813 | 1108| 3193 | 15490 | 54092 | 
| 22 | 16 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 16123 | 54818 | 
| 23 | 16 examples/model_selection/grid_search_text_feature_extraction.py | 98 | 133| 374 | 16497 | 54818 | 
| 24 | 16 sklearn/utils/estimator_checks.py | 1124 | 1192| 601 | 17098 | 54818 | 
| 25 | 17 examples/preprocessing/plot_all_scaling.py | 1 | 107| 792 | 17890 | 57934 | 
| 26 | 18 sklearn/linear_model/stochastic_gradient.py | 7 | 43| 346 | 18236 | 72336 | 
| 27 | 18 sklearn/utils/estimator_checks.py | 1398 | 1495| 1019 | 19255 | 72336 | 
| 28 | 19 examples/compose/plot_digits_pipe.py | 1 | 79| 572 | 19827 | 72916 | 
| 29 | 20 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 20208 | 74231 | 
| 30 | 21 sklearn/preprocessing/data.py | 10 | 60| 325 | 20533 | 98463 | 
| 31 | 22 examples/model_selection/plot_multi_metric_evaluation.py | 1 | 70| 543 | 21076 | 99344 | 
| 32 | 23 examples/preprocessing/plot_function_transformer.py | 1 | 73| 450 | 21526 | 99794 | 
| 33 | 24 sklearn/preprocessing/__init__.py | 1 | 71| 419 | 21945 | 100213 | 
| 34 | 25 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 22717 | 101377 | 
| 35 | **25 sklearn/utils/_pprint.py** | 104 | 166| 765 | 23482 | 101377 | 
| 36 | 26 examples/model_selection/plot_nested_cross_validation_iris.py | 74 | 118| 433 | 23915 | 102451 | 
| 37 | 26 sklearn/model_selection/_search.py | 1110 | 1436| 184 | 24099 | 102451 | 
| 38 | 27 sklearn/model_selection/_validation.py | 482 | 554| 746 | 24845 | 115537 | 
| 39 | 27 examples/preprocessing/plot_all_scaling.py | 221 | 312| 815 | 25660 | 115537 | 
| 40 | 27 sklearn/utils/estimator_checks.py | 939 | 1013| 702 | 26362 | 115537 | 
| 41 | 27 sklearn/utils/estimator_checks.py | 1016 | 1047| 346 | 26708 | 115537 | 
| 42 | 28 examples/linear_model/plot_sgd_early_stopping.py | 90 | 151| 530 | 27238 | 116838 | 
| 43 | 29 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 27523 | 117632 | 
| 44 | 30 sklearn/__init__.py | 1 | 80| 663 | 28186 | 118428 | 
| 45 | 31 examples/model_selection/plot_precision_recall.py | 100 | 205| 767 | 28953 | 120778 | 
| 46 | 32 sklearn/pipeline.py | 206 | 250| 413 | 29366 | 127694 | 
| 47 | 33 examples/decomposition/plot_faces_decomposition.py | 64 | 143| 628 | 29994 | 129131 | 
| 48 | 33 sklearn/utils/estimator_checks.py | 914 | 936| 248 | 30242 | 129131 | 
| 49 | 33 examples/model_selection/plot_nested_cross_validation_iris.py | 1 | 73| 640 | 30882 | 129131 | 
| 50 | 34 examples/applications/plot_out_of_core_classification.py | 268 | 361| 807 | 31689 | 132438 | 
| 51 | 34 examples/text/plot_document_classification_20newsgroups.py | 122 | 203| 697 | 32386 | 132438 | 
| 52 | 34 examples/applications/plot_out_of_core_classification.py | 362 | 420| 470 | 32856 | 132438 | 
| 53 | 35 sklearn/base.py | 226 | 243| 190 | 33046 | 136497 | 
| 54 | 36 sklearn/linear_model/coordinate_descent.py | 8 | 28| 153 | 33199 | 157652 | 
| 55 | 36 sklearn/pipeline.py | 150 | 176| 248 | 33447 | 157652 | 
| 56 | 37 examples/manifold/plot_lle_digits.py | 73 | 164| 748 | 34195 | 159531 | 
| 57 | 38 examples/linear_model/plot_theilsen.py | 89 | 112| 196 | 34391 | 160535 | 
| 58 | 39 sklearn/utils/__init__.py | 1 | 70| 585 | 34976 | 165263 | 
| 59 | 40 examples/plot_missing_values.py | 34 | 78| 447 | 35423 | 166342 | 
| 60 | 40 sklearn/preprocessing/data.py | 2588 | 2615| 293 | 35716 | 166342 | 
| 61 | 41 benchmarks/bench_tsne_mnist.py | 70 | 170| 1011 | 36727 | 167778 | 
| 62 | 41 examples/preprocessing/plot_all_scaling.py | 313 | 358| 353 | 37080 | 167778 | 
| 63 | 42 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 37626 | 168352 | 
| 64 | 43 examples/plot_anomaly_comparison.py | 81 | 153| 763 | 38389 | 169876 | 
| 65 | 44 examples/svm/plot_rbf_parameters.py | 74 | 160| 732 | 39121 | 171858 | 
| 66 | 45 sklearn/utils/_joblib.py | 1 | 35| 324 | 39445 | 172182 | 
| 67 | 46 examples/classification/plot_classifier_comparison.py | 79 | 145| 710 | 40155 | 173528 | 
| 68 | 46 sklearn/model_selection/_search.py | 360 | 377| 158 | 40313 | 173528 | 
| 69 | 46 sklearn/model_selection/_validation.py | 555 | 584| 341 | 40654 | 173528 | 
| 70 | 47 examples/ensemble/plot_feature_transformation.py | 1 | 83| 740 | 41394 | 174703 | 
| 71 | 48 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 94| 787 | 42181 | 175689 | 
| 72 | 48 sklearn/utils/estimator_checks.py | 848 | 879| 297 | 42478 | 175689 | 
| 73 | 48 sklearn/utils/estimator_checks.py | 1498 | 1563| 567 | 43045 | 175689 | 
| 74 | 49 benchmarks/bench_tree.py | 64 | 125| 523 | 43568 | 176559 | 
| 75 | 49 sklearn/base.py | 77 | 125| 404 | 43972 | 176559 | 
| 76 | 50 benchmarks/bench_glmnet.py | 47 | 129| 796 | 44768 | 177646 | 
| 77 | 50 examples/preprocessing/plot_all_scaling.py | 184 | 218| 381 | 45149 | 177646 | 
| 78 | 50 sklearn/utils/estimator_checks.py | 482 | 528| 467 | 45616 | 177646 | 
| 79 | 50 examples/model_selection/plot_cv_indices.py | 1 | 48| 368 | 45984 | 177646 | 
| 80 | 51 examples/ensemble/plot_bias_variance.py | 116 | 192| 684 | 46668 | 179460 | 
| 81 | 52 sklearn/utils/testing.py | 1 | 115| 782 | 47450 | 187022 | 
| 82 | 52 sklearn/utils/estimator_checks.py | 2179 | 2214| 401 | 47851 | 187022 | 
| 83 | 52 sklearn/utils/estimator_checks.py | 1050 | 1075| 282 | 48133 | 187022 | 
| 84 | 53 sklearn/feature_selection/rfe.py | 9 | 22| 130 | 48263 | 191598 | 
| 85 | 53 sklearn/model_selection/_search.py | 1 | 45| 290 | 48553 | 191598 | 
| 86 | 53 examples/text/plot_document_classification_20newsgroups.py | 1 | 121| 783 | 49336 | 191598 | 
| 87 | 54 examples/model_selection/plot_randomized_search.py | 54 | 87| 310 | 49646 | 192290 | 
| 88 | 55 benchmarks/bench_multilabel_metrics.py | 136 | 191| 543 | 50189 | 193941 | 
| 89 | 55 examples/model_selection/plot_multi_metric_evaluation.py | 72 | 97| 317 | 50506 | 193941 | 
| 90 | 55 sklearn/pipeline.py | 29 | 113| 914 | 51420 | 193941 | 


### Hint

```
So for some reason, the class is `PrettyPrinter` instead of `_EstimatorPrettyPrinter` (which inherits from `PrettyPrinter`). But then 

\`\`\`
  File "/path/to//sklearn/utils/_pprint.py", line 175, in _pprint_estimator
    if self._indent_at_name:
\`\`\`
is a `_EstimatorPrettyPrinter` method, so I don't understand what is going on...
By the way, the example also fails on master, but somehow circle-ci on master is green.
by example, I mean `examples/compose/plot_compare_reduction.py`
#12791 seems to be failing for the same reason.
> By the way, the example also fails on master, but somehow circle-ci on master is green.

I can't see it in the latest build on master.
I think it's because this line should involve a `.copy()`

https://github.com/scikit-learn/scikit-learn/blob/684d8a221d29ba1659e81961425a2380a9930044/sklearn/utils/_pprint.py#L324
That is, we're modifying the dispatch used by `pprint` rather than the local pretty printer.

But then it's also a bit weird that `_pprint_estimator` references a method on the class. This means that configuration of the class cannot affect anything. Rather it should perhaps reference an instancemethod on a configuration singleton??
@NicolasHug do you want to fix this, or should we open it to other contributors?
Thanks @jnothman I didn't see this, I'll take a look
You're right @jnothman we should make a copy of `_dispatch`.

The bug happens because joblib is using calling `PrettyPrinter` on an estimator, but the `_dispatch` dict of `PrettyPrinter` has been updated by `_EstimatorPrettyPrinter` sometime before, which tells the `PrettyPrinter` object to use `_EstimatorPrettyPrinter._pprint_estimator` to render `BaseEstimator` objects.

Pretty sneaky... I'll submit a fix.


However I'm not sure I follow your concern about `_pprint_estimator` being a method.
Minimal reproducing example:

\`\`\`
from pprint import PrettyPrinter
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
PrettyPrinter().pprint(lr)
\`\`\`
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


## Code snippets

### 1 - examples/compose/plot_compare_reduction.py:

Start line: 1, End line: 106

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=================================================================
Selecting dimensionality reduction with Pipeline and GridSearchCV
=================================================================

This example constructs a pipeline that does dimensionality
reduction followed by prediction with a support vector
classifier. It demonstrates the use of ``GridSearchCV`` and
``Pipeline`` to optimize over different classes of estimators in a
single CV run -- unsupervised ``PCA`` and ``NMF`` dimensionality
reductions are compared to univariate feature selection during
the grid search.

Additionally, ``Pipeline`` can be instantiated with the ``memory``
argument to memoize the transformers within the pipeline, avoiding to fit
again the same transformers over and over.

Note that the use of ``memory`` to enable caching becomes interesting when the
fitting of a transformer is costly.
"""

###############################################################################
# Illustration of ``Pipeline`` and ``GridSearchCV``
###############################################################################
# This section illustrates the use of a ``Pipeline`` with
# ``GridSearchCV``

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

print(__doc__)

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC())
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

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
digits = load_digits()
grid.fit(digits.data, digits.target)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()

###############################################################################
# Caching transformers within a ``Pipeline``
###############################################################################
# It is sometimes worthwhile storing the state of a specific transformer
# since it could be used again. Using a pipeline in ``GridSearchCV`` triggers
# such situations. Therefore, we use the argument ``memory`` to enable caching.
#
# .. warning::
#     Note that this example is, however, only an illustration since for this
#     specific case fitting PCA is not necessarily slower than loading the
#     cache. Hence, use the ``memory`` constructor parameter when the fitting
#     of a transformer is costly.

from tempfile import mkdtemp
```
### 2 - examples/preprocessing/plot_discretization_classification.py:

Start line: 1, End line: 89

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
Feature discretization
======================

A demonstration of feature discretization on synthetic classification datasets.
Feature discretization decomposes each feature into a set of bins, here equally
distributed in width. The discrete values are then one-hot encoded, and given
to a linear classifier. This preprocessing enables a non-linear behavior even
though the classifier is linear.

On this example, the first two rows represent linearly non-separable datasets
(moons and concentric circles) while the third is approximately linearly
separable. On the two linearly non-separable datasets, feature discretization
largely increases the performance of linear classifiers. On the linearly
separable dataset, feature discretization decreases the performance of linear
classifiers. Two non-linear classifiers are also shown for comparison.

This example should be taken with a grain of salt, as the intuition conveyed
does not necessarily carry over to real datasets. Particularly in
high-dimensional spaces, data can more easily be separated linearly. Moreover,
using feature discretization and one-hot encoding increases the number of
features, which easily lead to overfitting when the number of samples is small.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
# Code source: Tom Dupré la Tour
# Adapted from plot_classifier_comparison by Gaël Varoquaux and Andreas Müller
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

print(__doc__)

h = .02  # step size in the mesh


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == 'Pipeline':
        name = [get_name(est[1]) for est in estimator.steps]
        name = ' + '.join(name)
    return name


# list of (estimator, param_grid), where param_grid is used in GridSearchCV
classifiers = [
    (LogisticRegression(solver='lbfgs', random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (LinearSVC(random_state=0), {
        'C': np.logspace(-2, 7, 10)
    }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'),
        LogisticRegression(solver='lbfgs', random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'logisticregression__C': np.logspace(-2, 7, 10),
        }),
    (make_pipeline(
        KBinsDiscretizer(encode='onehot'), LinearSVC(random_state=0)), {
            'kbinsdiscretizer__n_bins': np.arange(2, 10),
            'linearsvc__C': np.logspace(-2, 7, 10),
        }),
    (GradientBoostingClassifier(n_estimators=50, random_state=0), {
        'learning_rate': np.logspace(-4, 0, 10)
    }),
    (SVC(random_state=0, gamma='scale'), {
        'C': np.logspace(-2, 7, 10)
    }),
]
```
### 3 - sklearn/utils/estimator_checks.py:

Start line: 1, End line: 80

```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
from functools import partial
from inspect import signature

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from sklearn.utils import IS_PYPY, _IS_32BIT
from sklearn.utils import _joblib
from sklearn.utils._joblib import Memory
from sklearn.utils.testing import assert_raises, _get_args
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import META_ESTIMATORS
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_dict_equal
from sklearn.utils.testing import create_memmap_backed_data
from sklearn.utils import is_scalar_nan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, ClusterMixin,
                          is_classifier, is_regressor, is_outlier_detector)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._validation import _safe_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.validation import has_fit_parameter, _num_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']

ALLOW_NAN = ['Imputer', 'SimpleImputer', 'MissingIndicator',
             'MaxAbsScaler', 'MinMaxScaler', 'RobustScaler', 'StandardScaler',
             'PowerTransformer', 'QuantileTransformer']
```
### 4 - sklearn/utils/_pprint.py:

Start line: 168, End line: 197

```python
class _EstimatorPrettyPrinter(pprint.PrettyPrinter):

    def format(self, object, context, maxlevels, level):
        return _safe_repr(object, context, maxlevels, level,
                          changed_only=self._changed_only)

    def _pprint_estimator(self, object, stream, indent, allowance, context,
                          level):
        stream.write(object.__class__.__name__ + '(')
        if self._indent_at_name:
            indent += len(object.__class__.__name__)

        if self._changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)

        params = OrderedDict((name, val)
                             for (name, val) in sorted(params.items()))

        self._format_params(params.items(), stream, indent, allowance + 1,
                            context, level)
        stream.write(')')

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True)

    def _format_params(self, items, stream, indent, allowance, context, level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=False)
```
### 5 - sklearn/model_selection/__init__.py:

Start line: 1, End line: 60

```python
from ._split import BaseCrossValidator
from ._split import KFold
from ._split import GroupKFold
from ._split import StratifiedKFold
from ._split import TimeSeriesSplit
from ._split import LeaveOneGroupOut
from ._split import LeaveOneOut
from ._split import LeavePGroupsOut
from ._split import LeavePOut
from ._split import RepeatedKFold
from ._split import RepeatedStratifiedKFold
from ._split import ShuffleSplit
from ._split import GroupShuffleSplit
from ._split import StratifiedShuffleSplit
from ._split import PredefinedSplit
from ._split import train_test_split
from ._split import check_cv

from ._validation import cross_val_score
from ._validation import cross_val_predict
from ._validation import cross_validate
from ._validation import learning_curve
from ._validation import permutation_test_score
from ._validation import validation_curve

from ._search import GridSearchCV
from ._search import RandomizedSearchCV
from ._search import ParameterGrid
from ._search import ParameterSampler
from ._search import fit_grid_point

__all__ = ('BaseCrossValidator',
           'GridSearchCV',
           'TimeSeriesSplit',
           'KFold',
           'GroupKFold',
           'GroupShuffleSplit',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedKFold',
           'RepeatedStratifiedKFold',
           'ParameterGrid',
           'ParameterSampler',
           'PredefinedSplit',
           'RandomizedSearchCV',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'check_cv',
           'cross_val_predict',
           'cross_val_score',
           'cross_validate',
           'fit_grid_point',
           'learning_curve',
           'permutation_test_score',
           'train_test_split',
           'validation_curve')
```
### 6 - sklearn/model_selection/_search.py:

Start line: 755, End line: 1124

```python
class BaseSearchCV(BaseEstimator, MetaEstimatorMixin, metaclass=ABCMeta):

    def _format_results(self, candidate_params, scorers, n_splits, out):
        # ... other code

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        iid = self.iid
        if self.iid == 'warn':
            warn = False
            for scorer_name in scorers.keys():
                scores = test_scores[scorer_name].reshape(n_candidates,
                                                          n_splits)
                means_weighted = np.average(scores, axis=1,
                                            weights=test_sample_counts)
                means_unweighted = np.average(scores, axis=1)
                if not np.allclose(means_weighted, means_unweighted,
                                   rtol=1e-4, atol=1e-4):
                    warn = True
                    break

            if warn:
                warnings.warn("The default of the `iid` parameter will change "
                              "from True to False in version 0.22 and will be"
                              " removed in 0.24. This will change numeric"
                              " results when test-set sizes are unequal.",
                              DeprecationWarning)
            iid = True

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if iid else None)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

        return results


class GridSearchCV(BaseSearchCV):
```
### 7 - examples/preprocessing/plot_discretization_classification.py:

Start line: 109, End line: 193

```python
for ds_cnt, (X, y) in enumerate(datasets):
    print('\ndataset %d\n---------' % ds_cnt)

    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.5, random_state=42)

    # create the grid for background colors
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot the dataset first
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    # iterate over classifiers
    for est_idx, (name, (estimator, param_grid)) in \
            enumerate(zip(names, classifiers)):
        ax = axes[ds_cnt, est_idx + 1]

        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5,
                           iid=False)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('%s: %.2f' % (name, score))

        # plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]*[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(name.replace(' + ', '\n'))
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0'), size=15,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                transform=ax.transAxes, horizontalalignment='right')


plt.tight_layout()

# Add suptitles above the figure
plt.subplots_adjust(top=0.90)
suptitles = [
    'Linear classifiers',
    'Feature discretization and linear classifiers',
    'Non-linear classifiers',
]
for i, suptitle in zip([1, 3, 5], suptitles):
    ax = axes[0, i]
    ax.text(1.05, 1.25, suptitle, transform=ax.transAxes,
            horizontalalignment='center', size='x-large')
plt.show()
```
### 8 - examples/compose/plot_column_transformer_mixed_types.py:

Start line: 1, End line: 106

```python
"""
===================================
Column Transformer with Mixed Types
===================================

This example illustrates how to apply different preprocessing and
feature extraction pipelines to different subsets of features,
using :class:`sklearn.compose.ColumnTransformer`.
This is particularly handy for the case of datasets that contain
heterogeneous data types, since we may want to scale the
numeric features and one-hot encode the categorical ones.

In this example, the numeric data is standard-scaled after
mean-imputation, while the categorical data is one-hot
encoded after imputing missing values with a new category
(``'missing'``).

Finally, the preprocessing pipeline is integrated in a
full prediction pipeline using :class:`sklearn.pipeline.Pipeline`,
together with a simple classification model.
"""

from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# Read data from Titanic dataset.
titanic_url = ('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
data = pd.read_csv(titanic_url)

# We will train our classifier with the following features:
# Numeric Features:
# - age: float.
# - fare: float.
# Categorical Features:
# - embarked: categories encoded as strings {'C', 'S', 'Q'}.
# - sex: categories encoded as strings {'female', 'male'}.
# - pclass: ordinal integers {1, 2, 3}.

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


###############################################################################
# Using the prediction pipeline in a grid search
###############################################################################
# Grid search can also be performed on the different preprocessing steps
# defined in the ``ColumnTransformer`` object, together with the classifier's
# hyperparameters as part of the ``Pipeline``.
# We will search for both the imputer strategy of the numeric preprocessing
# and the regularization parameter of the logistic regression using
# :class:`sklearn.model_selection.GridSearchCV`.


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__C': [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
```
### 9 - benchmarks/bench_mnist.py:

Start line: 109, End line: 181

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifiers', nargs="+",
                        choices=ESTIMATORS, type=str,
                        default=['ExtraTrees', 'Nystroem-SVM'],
                        help="list of classifiers to benchmark.")
    parser.add_argument('--n-jobs', nargs="?", default=1, type=int,
                        help="Number of concurrently running workers for "
                             "models that support parallelism.")
    parser.add_argument('--order', nargs="?", default="C", type=str,
                        choices=["F", "C"],
                        help="Allow to choose between fortran and C ordered "
                             "data")
    parser.add_argument('--random-seed', nargs="?", default=0, type=int,
                        help="Common seed used by random number generator.")
    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (size=%dMB)" % ("number of train samples:".ljust(25),
                                 X_train.shape[0], int(X_train.nbytes / 1e6)))
    print("%s %d (size=%dMB)" % ("number of test samples:".ljust(25),
                                 X_test.shape[0], int(X_test.nbytes / 1e6)))

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        estimator.set_params(**{p: args["random_seed"]
                                for p in estimator_params
                                if p.endswith("random_state")})

        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("{0: <24} {1: >10} {2: >11} {3: >12}"
          "".format("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 60)
    for name in sorted(args["classifiers"], key=error.get):

        print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
              "".format(name, train_time[name], test_time[name], error[name]))

    print()
```
### 10 - examples/model_selection/grid_search_text_feature_extraction.py:

Start line: 2, End line: 97

```python
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.

Here is a sample output of a run on a quad-core machine::

  Loading 20 newsgroups dataset for categories:
  ['alt.atheism', 'talk.religion.misc']
  1427 documents
  2 categories

  Performing grid search...
  pipeline: ['vect', 'tfidf', 'clf']
  parameters:
  {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
   'clf__max_iter': (10, 50, 80),
   'clf__penalty': ('l2', 'elasticnet'),
   'tfidf__use_idf': (True, False),
   'vect__max_n': (1, 2),
   'vect__max_df': (0.5, 0.75, 1.0),
   'vect__max_features': (None, 5000, 10000, 50000)}
  done in 1737.030s

  Best score: 0.940
  Best parameters set:
      clf__alpha: 9.9999999999999995e-07
      clf__max_iter: 50
      clf__penalty: 'elasticnet'
      tfidf__use_idf: True
      vect__max_n: 2
      vect__max_df: 0.75
      vect__max_features: 50000

"""

from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

data = fetch_20newsgroups(subset='train', categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
```
### 20 - sklearn/utils/_pprint.py:

Start line: 309, End line: 326

```python
class _EstimatorPrettyPrinter(pprint.PrettyPrinter):

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context,
                              level):
        """Pretty printing for key-value tuples from dict or parameters."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = '='
        else:
            middle = ': '
        stream.write(rep)
        stream.write(middle)
        self._format(v, stream, indent + len(rep) + len(middle), allowance,
                     context, level)

    _dispatch = pprint.PrettyPrinter._dispatch
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple
```
### 35 - sklearn/utils/_pprint.py:

Start line: 104, End line: 166

```python
class _EstimatorPrettyPrinter(pprint.PrettyPrinter):
    """Pretty Printer class for estimator objects.

    This extends the pprint.PrettyPrinter class, because:
    - we need estimators to be printed with their parameters, e.g.
      Estimator(param1=value1, ...) which is not supported by default.
    - the 'compact' parameter of PrettyPrinter is ignored for dicts, which
      may lead to very long representations that we want to avoid.

    Quick overview of pprint.PrettyPrinter (see also
    https://stackoverflow.com/questions/49565047/pprint-with-hex-numbers):

    - the entry point is the _format() method which calls format() (overridden
      here)
    - format() directly calls _safe_repr() for a first try at rendering the
      object
    - _safe_repr formats the whole object reccursively, only calling itself,
      not caring about line length or anything
    - back to _format(), if the output string is too long, _format() then calls
      the appropriate _pprint_TYPE() method (e.g. _pprint_list()) depending on
      the type of the object. This where the line length and the compact
      parameters are taken into account.
    - those _pprint_TYPE() methods will internally use the format() method for
      rendering the nested objects of an object (e.g. the elements of a list)

    In the end, everything has to be implemented twice: in _safe_repr and in
    the custom _pprint_TYPE methods. Unfortunately PrettyPrinter is really not
    straightforward to extend (especially when we want a compact output), so
    the code is a bit convoluted.

    This class overrides:
    - format() to support the changed_only parameter
    - _safe_repr to support printing of estimators (for when they fit on a
      single line)
    - _format_dict_items so that dict are correctly 'compacted'
    - _format_items so that ellipsis is used on long lists and tuples

    When estimators cannot be printed on a single line, the builtin _format()
    will call _pprint_estimator() because it was registered to do so (see
    _dispatch[BaseEstimator.__repr__] = _pprint_estimator).

    both _format_dict_items() and _pprint_estimator() use the
    _format_params_or_dict_items() method that will format parameters and
    key-value pairs respecting the compact parameter. This method needs another
    subroutine _pprint_key_val_tuple() used when a parameter or a key-value
    pair is too long to fit on a single line. This subroutine is called in
    _format() and is registered as well in the _dispatch dict (just like
    _pprint_estimator). We had to create the two classes KeyValTuple and
    KeyValTupleParam for this.
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, indent_at_name=True,
                 n_max_elements_to_show=None):
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # ignore indent param
        self._changed_only = get_config()['print_changed_only']
        # Max number of elements in a list, dict, tuple until we start using
        # ellipsis. This also affects the number of arguments of an estimators
        # (they are treated as dicts)
        self.n_max_elements_to_show = n_max_elements_to_show
```
