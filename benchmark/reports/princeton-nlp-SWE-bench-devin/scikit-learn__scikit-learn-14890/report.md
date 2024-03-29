# scikit-learn__scikit-learn-14890

| **scikit-learn/scikit-learn** | `14f5302b7000e9096de93beef37dcdb08f55f128` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6594 |
| **Any found context length** | 6594 |
| **Avg pos** | 33.0 |
| **Min pos** | 11 |
| **Max pos** | 11 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/compose/_target.py b/sklearn/compose/_target.py
--- a/sklearn/compose/_target.py
+++ b/sklearn/compose/_target.py
@@ -148,7 +148,7 @@ def _fit_transformer(self, y):
                               " you are sure you want to proceed regardless"
                               ", set 'check_inverse=False'", UserWarning)
 
-    def fit(self, X, y, sample_weight=None):
+    def fit(self, X, y, **fit_params):
         """Fit the model according to the given training data.
 
         Parameters
@@ -160,9 +160,10 @@ def fit(self, X, y, sample_weight=None):
         y : array-like, shape (n_samples,)
             Target values.
 
-        sample_weight : array-like, shape (n_samples,) optional
-            Array of weights that are assigned to individual samples.
-            If not provided, then each sample is given unit weight.
+        **fit_params : dict of string -> object
+            Parameters passed to the ``fit`` method of the underlying
+            regressor.
+
 
         Returns
         -------
@@ -197,10 +198,7 @@ def fit(self, X, y, sample_weight=None):
         else:
             self.regressor_ = clone(self.regressor)
 
-        if sample_weight is None:
-            self.regressor_.fit(X, y_trans)
-        else:
-            self.regressor_.fit(X, y_trans, sample_weight=sample_weight)
+        self.regressor_.fit(X, y_trans, **fit_params)
 
         return self
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/compose/_target.py | 151 | 151 | 11 | 2 | 6594
| sklearn/compose/_target.py | 163 | 165 | 11 | 2 | 6594
| sklearn/compose/_target.py | 200 | 203 | 11 | 2 | 6594


## Problem Statement

```
Fitting TransformedTargetRegressor with sample_weight in Pipeline
#### Description

Can't fit a `TransformedTargetRegressor` using `sample_weight`. May be link to #10945 ?

#### Steps/Code to Reproduce

Example:
\`\`\`python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Create dataset
X, y = make_regression(n_samples=10000, noise=100, n_features=10, random_state=2019)
y = np.exp((y + abs(y.min())) / 200)
w = np.random.randn(len(X))
cat_list = ['AA', 'BB', 'CC', 'DD']
cat = np.random.choice(cat_list, len(X), p=[0.3, 0.2, 0.2, 0.3])

df = pd.DataFrame(X, columns=["col_" + str(i) for i in range(1, 11)])
df['sample_weight'] = w
df['my_caterogy'] = cat
df.head()
\`\`\`
![image](https://user-images.githubusercontent.com/8374843/53635914-e169bf00-3c1e-11e9-8d91-e8f474de860c.png)

\`\`\`python
use_col = [col for col in df.columns if col not in ['sample_weight']]


numerical_features = df[use_col].dtypes == 'float'
categorical_features = ~numerical_features

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocess = make_column_transformer(
                                    (RobustScaler(), numerical_features),
                                    (OneHotEncoder(sparse=False), categorical_features)
)

rf = RandomForestRegressor(n_estimators=20)

clf = Pipeline(steps=[
                      ('preprocess', preprocess),
                      ('model', rf)
])

clf_trans = TransformedTargetRegressor(regressor=clf,
                                        func=np.log1p,
                                        inverse_func=np.expm1)

# Work
clf_trans.fit(df[use_col], y)

# Fail
clf_trans.fit(df[use_col], y, sample_weight=df['sample_weight'])
\`\`\`

#### Expected Results
Fitting with `sample_weight`

#### Actual Results
\`\`\`python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-7-366d815659ba> in <module>()
----> 1 clf_trans.fit(df[use_col], y, sample_weight=df['sample_weight'])

~/anaconda3/envs/test_env/lib/python3.5/site-packages/sklearn/compose/_target.py in fit(self, X, y, sample_weight)
    194             self.regressor_.fit(X, y_trans)
    195         else:
--> 196             self.regressor_.fit(X, y_trans, sample_weight=sample_weight)
    197 
    198         return self

~/anaconda3/envs/test_env/lib/python3.5/site-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    263             This estimator
    264         """
--> 265         Xt, fit_params = self._fit(X, y, **fit_params)
    266         if self._final_estimator is not None:
    267             self._final_estimator.fit(Xt, y, **fit_params)

~/anaconda3/envs/test_env/lib/python3.5/site-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params)
    200                                 if step is not None)
    201         for pname, pval in six.iteritems(fit_params):
--> 202             step, param = pname.split('__', 1)
    203             fit_params_steps[step][param] = pval
    204         Xt = X

ValueError: not enough values to unpack (expected 2, got 1)
\`\`\`

#### Versions
\`\`\`python
import sklearn; sklearn.show_versions()
System:
   machine: Linux-4.4.0-127-generic-x86_64-with-debian-stretch-sid
executable: /home/gillesa/anaconda3/envs/test_env/bin/python
    python: 3.5.6 |Anaconda, Inc.| (default, Aug 26 2018, 21:41:56)  [GCC 7.3.0]

BLAS:
cblas_libs: cblas
  lib_dirs: 
    macros: 

Python deps:
   sklearn: 0.20.2
    pandas: 0.24.1
       pip: 19.0.1
setuptools: 40.2.0
     numpy: 1.16.1
    Cython: None
     scipy: 1.2.0
\`\`\`

<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/compose/plot_transformed_target.py | 99 | 179| 749 | 749 | 1832 | 
| 2 | 1 examples/compose/plot_transformed_target.py | 1 | 98| 743 | 1492 | 1832 | 
| 3 | **2 sklearn/compose/_target.py** | 107 | 149| 441 | 1933 | 3778 | 
| 4 | 3 sklearn/pipeline.py | 702 | 744| 285 | 2218 | 11602 | 
| 5 | 3 examples/compose/plot_transformed_target.py | 180 | 208| 312 | 2530 | 11602 | 
| 6 | 4 examples/compose/plot_column_transformer_mixed_types.py | 1 | 103| 797 | 3327 | 12421 | 
| 7 | **4 sklearn/compose/_target.py** | 5 | 106| 805 | 4132 | 12421 | 
| 8 | 5 examples/ensemble/plot_feature_transformation.py | 1 | 87| 761 | 4893 | 13569 | 
| 9 | 5 sklearn/pipeline.py | 258 | 318| 543 | 5436 | 13569 | 
| 10 | 6 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 59 | 124| 700 | 6136 | 14751 | 
| **-> 11 <-** | **6 sklearn/compose/_target.py** | 151 | 205| 458 | 6594 | 14751 | 
| 12 | 7 examples/preprocessing/plot_function_transformer.py | 1 | 73| 450 | 7044 | 15201 | 
| 13 | 8 examples/compose/plot_column_transformer.py | 87 | 136| 379 | 7423 | 16197 | 
| 14 | 9 sklearn/utils/estimator_checks.py | 1124 | 1197| 650 | 8073 | 39369 | 
| 15 | 10 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 8888 | 42477 | 
| 16 | 10 sklearn/pipeline.py | 587 | 624| 278 | 9166 | 42477 | 
| 17 | 11 examples/compose/plot_compare_reduction.py | 1 | 105| 822 | 9988 | 43509 | 
| 18 | 11 sklearn/pipeline.py | 353 | 389| 304 | 10292 | 43509 | 
| 19 | 12 sklearn/preprocessing/data.py | 2747 | 2774| 294 | 10586 | 69062 | 
| 20 | 12 sklearn/pipeline.py | 29 | 123| 993 | 11579 | 69062 | 
| 21 | 12 sklearn/pipeline.py | 489 | 506| 125 | 11704 | 69062 | 
| 22 | 12 sklearn/pipeline.py | 508 | 525| 134 | 11838 | 69062 | 
| 23 | 13 sklearn/compose/_column_transformer.py | 482 | 535| 453 | 12291 | 75362 | 
| 24 | 14 benchmarks/bench_random_projections.py | 85 | 251| 1264 | 13555 | 77097 | 
| 25 | 14 sklearn/pipeline.py | 391 | 416| 207 | 13762 | 77097 | 
| 26 | 14 sklearn/pipeline.py | 320 | 351| 241 | 14003 | 77097 | 
| 27 | 14 sklearn/compose/_column_transformer.py | 459 | 480| 151 | 14154 | 77097 | 
| 28 | 15 sklearn/ensemble/gradient_boosting.py | 1478 | 1558| 757 | 14911 | 99064 | 
| 29 | 15 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 15695 | 99064 | 
| 30 | 15 sklearn/utils/estimator_checks.py | 1096 | 1121| 283 | 15978 | 99064 | 
| 31 | 16 sklearn/linear_model/ransac.py | 327 | 423| 806 | 16784 | 103112 | 
| 32 | 17 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 775 | 17559 | 104932 | 
| 33 | 18 benchmarks/bench_hist_gradient_boosting.py | 66 | 139| 728 | 18287 | 107217 | 
| 34 | 18 sklearn/utils/estimator_checks.py | 672 | 695| 292 | 18579 | 107217 | 
| 35 | 18 sklearn/pipeline.py | 470 | 487| 131 | 18710 | 107217 | 
| 36 | 19 examples/inspection/plot_partial_dependence.py | 1 | 103| 872 | 19582 | 109177 | 
| 37 | 20 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 20336 | 110396 | 
| 38 | 21 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 86| 769 | 21105 | 111549 | 
| 39 | 21 sklearn/compose/_column_transformer.py | 427 | 457| 284 | 21389 | 111549 | 
| 40 | 21 sklearn/utils/estimator_checks.py | 1229 | 1257| 306 | 21695 | 111549 | 
| 41 | 22 sklearn/cross_decomposition/pls_.py | 293 | 355| 839 | 22534 | 119760 | 
| 42 | 22 sklearn/pipeline.py | 162 | 188| 248 | 22782 | 119760 | 
| 43 | 22 sklearn/linear_model/ransac.py | 424 | 453| 347 | 23129 | 119760 | 
| 44 | 22 sklearn/pipeline.py | 451 | 468| 131 | 23260 | 119760 | 
| 45 | 22 examples/preprocessing/plot_all_scaling.py | 311 | 356| 353 | 23613 | 119760 | 
| 46 | 22 sklearn/utils/estimator_checks.py | 698 | 714| 189 | 23802 | 119760 | 
| 47 | 22 sklearn/cross_decomposition/pls_.py | 239 | 292| 580 | 24382 | 119760 | 
| 48 | 22 sklearn/utils/estimator_checks.py | 992 | 1023| 275 | 24657 | 119760 | 
| 49 | 23 sklearn/neighbors/regression.py | 138 | 191| 452 | 25109 | 122781 | 
| 50 | 23 sklearn/ensemble/gradient_boosting.py | 2103 | 2113| 125 | 25234 | 122781 | 
| 51 | 23 sklearn/pipeline.py | 527 | 554| 190 | 25424 | 122781 | 
| 52 | 24 examples/linear_model/plot_robust_fit.py | 1 | 73| 509 | 25933 | 123575 | 
| 53 | 24 sklearn/preprocessing/data.py | 683 | 747| 572 | 26505 | 123575 | 
| 54 | 24 sklearn/utils/estimator_checks.py | 1 | 60| 431 | 26936 | 123575 | 
| 55 | 24 sklearn/linear_model/ransac.py | 230 | 326| 807 | 27743 | 123575 | 
| 56 | 24 benchmarks/bench_random_projections.py | 44 | 63| 122 | 27865 | 123575 | 
| 57 | 25 sklearn/metrics/regression.py | 546 | 593| 476 | 28341 | 130609 | 
| 58 | 25 sklearn/utils/estimator_checks.py | 717 | 752| 468 | 28809 | 130609 | 
| 59 | 25 sklearn/utils/estimator_checks.py | 1308 | 1376| 597 | 29406 | 130609 | 
| 60 | 26 examples/compose/plot_digits_pipe.py | 1 | 77| 559 | 29965 | 131176 | 
| 61 | 27 benchmarks/bench_sample_without_replacement.py | 34 | 202| 1229 | 31194 | 132568 | 
| 62 | 27 sklearn/pipeline.py | 903 | 932| 245 | 31439 | 132568 | 
| 63 | 27 examples/compose/plot_column_transformer.py | 1 | 54| 371 | 31810 | 132568 | 
| 64 | 27 examples/preprocessing/plot_scaling_importance.py | 82 | 134| 457 | 32267 | 132568 | 
| 65 | 28 sklearn/isotonic.py | 302 | 345| 356 | 32623 | 135974 | 
| 66 | 28 sklearn/pipeline.py | 934 | 950| 178 | 32801 | 135974 | 
| 67 | 29 benchmarks/bench_mnist.py | 84 | 105| 306 | 33107 | 137685 | 
| 68 | 30 sklearn/metrics/classification.py | 44 | 112| 564 | 33671 | 159632 | 
| 69 | 30 sklearn/pipeline.py | 879 | 901| 138 | 33809 | 159632 | 
| 70 | 31 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 772 | 34581 | 160553 | 
| 71 | 32 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 34954 | 162498 | 
| 72 | 32 sklearn/pipeline.py | 418 | 449| 270 | 35224 | 162498 | 
| 73 | 32 sklearn/utils/estimator_checks.py | 1026 | 1057| 289 | 35513 | 162498 | 
| 74 | 32 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 35798 | 162498 | 
| 75 | 33 sklearn/calibration.py | 318 | 361| 307 | 36105 | 167392 | 
| 76 | 34 sklearn/datasets/samples_generator.py | 534 | 574| 372 | 36477 | 181402 | 
| 77 | 34 sklearn/utils/estimator_checks.py | 2507 | 2538| 340 | 36817 | 181402 | 
| 78 | 35 examples/linear_model/plot_theilsen.py | 1 | 88| 785 | 37602 | 182406 | 
| 79 | 36 sklearn/linear_model/base.py | 177 | 191| 155 | 37757 | 187180 | 
| 80 | **36 sklearn/compose/_target.py** | 207 | 239| 239 | 37996 | 187180 | 
| 81 | 37 sklearn/ensemble/voting.py | 440 | 462| 176 | 38172 | 191169 | 
| 82 | 38 benchmarks/bench_plot_nmf.py | 230 | 278| 535 | 38707 | 195059 | 
| 83 | 39 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 39274 | 195626 | 
| 84 | 39 sklearn/preprocessing/data.py | 11 | 60| 312 | 39586 | 195626 | 
| 85 | 39 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 40118 | 195626 | 
| 86 | 39 sklearn/preprocessing/data.py | 2941 | 2984| 389 | 40507 | 195626 | 
| 87 | 39 sklearn/utils/estimator_checks.py | 1982 | 2024| 462 | 40969 | 195626 | 


### Hint

```
This has nothing to do with TransformedTargetRegressor. Pipeline requires
you to pass model__sample_weight, not just sample_weight... But the error
message is terrible! We should improve it.

Thank you for your prompt reply @jnothman 

### Second try : 
\`\`\`python
clf_trans.fit(X_train[use_col], y_train,
              model__sample_weight=X_train['weight']
             )

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-25-aa3242bb1603> in <module>()
----> 1 clf_trans.fit(df[use_col], y, model__sample_weight=df['sample_weight'])

TypeError: fit() got an unexpected keyword argument 'model__sample_weight'
\`\`\`

Did i miss something or anything ?

By the way I used this kind of pipeline typo (don't know how to call it) in `GridSearchCV` and it work's well !

\`\`\`python
from sklearn.model_selection import GridSearchCV

param_grid = { 
    'regressor__model__n_estimators': [20, 50, 100, 200]
}

GDCV = GridSearchCV(estimator=clf_trans, param_grid=param_grid, cv=5,
                    n_jobs=-1, scoring='neg_mean_absolute_error',
                    return_train_score=True, verbose=True)
GDCV.fit(X[use_col], y)
\`\`\`

Ps : Fill free to rename title if it can help community
You're right. we don't yet seem to properly support fit parameters in TransformedTargetRegressor. And perhaps we should...
> This has nothing to do with TransformedTargetRegressor. Pipeline requires you to pass model__sample_weight, not just sample_weight... But the error message is terrible! We should improve it.

That's true but what @armgilles asked in the first example was the sample_weight, a parameter that it's passed in the fit call. From my knowledge, specifying model__sample_weight just sets internal attributes of the model step in the pipeline but doesn't modify any parameters passed to the fit method

Should we implement both parameters, meaning the parameter of the model (like we do in GridSearchCV) and parameter of the fit (eg. sample_weight, i don't know if there are more that could be passed in fit call) ?
No, the comment *is* about fit parameters. TransformedTargetRegressor
currently accepts sample_weight, but to support pipelines it needs to
support **fit_params

Cool, I'll give it a try then
I am having the same problem here using the `Pipeline` along with `CatBoostRegressor`. The only hacky way I found so far to accomplish this is to do something like:
\`\`\`
pipeline.named_steps['reg'].regressor.set_params(**fit_params)
# Or alternatively 
pipeline.set_params({"reg_regressor_param": value})
\`\`\`
And then call 
\`\`\`
pipeline.fit(X, y)
\`\`\`

Where `reg` is the step containing the `TransformedTargetRegressor`. is there a cleaner way? 
That's not about a fit parameter like sample_weight at all. For that you
should be able to set_params directly from the TransformedTargetRegressor
instance. Call its get_params to find the right key.

@jnothman thanks for your response . Please let me know if I am doing something wrong. From what I understand there are 3 issues here:


1.  `TransformedTargetRegressor` fit only passes sample_weight to the underlying regressor. Which you can argue that's what is has to do. Other estimators, (not sklearn based but compatible). might  support receiving other  prams in the `fit` method. 

2. `TransformedTargetRegressor` only support sample_weight as a parameter and d[oes not support passing arbitrary parameters](https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/compose/_target.py#L200-L205) to the underlying `regressor` fit method as `Pipeline` does (i.e. using `<component>__<parameter>` convention ). 

3. Now, when using a Pipeline  and I want to pass a parameter to the regressor inside a `TransformedTargetRegressor` at fit time this fails. 

Some examples:

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor 
import numpy as np

tr_regressor = TransformedTargetRegressor(
            CatBoostRegressor(),
             func=np.log, inverse_func=np.exp
)

pipeline = Pipeline(steps=[
              ('reg', tr_regressor)
])

X = np.arange(4).reshape(-1, 1)
y = np.exp(2 * X).ravel()

pipeline.fit(X, y, reg__regressor__verbose=False)
---
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
     17 y = np.exp(2 * X).ravel()
     18 
---> 19 pipeline.fit(X, y, reg__regressor__verbose=False)

~/development/order_prediction/ord_pred_env/lib/python3.6/site-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    354                                  self._log_message(len(self.steps) - 1)):
    355             if self._final_estimator != 'passthrough':
--> 356                 self._final_estimator.fit(Xt, y, **fit_params)
    357         return self
    358 

TypeError: fit() got an unexpected keyword argument 'regressor__verbose'
\`\`\`

This also fails:

\`\`\`python
pipeline.named_steps['reg'].fit(X, y, regressor__verbose=False)

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-19-fd09c06db732> in <module>
----> 1 pipeline.named_steps['reg'].fit(X, y, regressor__verbose=False)

TypeError: fit() got an unexpected keyword argument 'regressor__verbose'
\`\`\`

This actually works:

\`\`\`python
pipeline.named_steps['reg'].regressor.fit(X, y, verbose=False)
\`\`\`
And this will also work:
\`\`\`python
pipeline.set_params(**{'reg__regressor__verbose': False})
pipeline.fit(X, y)
\`\`\`

So I have a question:

Shouldn't `TransformedTargetRegressor` `fit` method support `**fit_params` as the `Pipeline`does? i.e. passing parameters to the underlying regressor via the `<component>__<parameter>` syntax? 

Maybe I missing something or  expecting something from the API I should not be expecting here. Thanks in advance for the help :). 




I think the discussion started from the opposite way around: using a `Pipeline `as the `regressor `parameter of the `TransformedTargetRegressor`. The problem is the same: you cannot pass fit parameters to the underlying regressor apart from the `sample_weight`.

Another question is if there are cases where you would want to pass fit parameters to the transformer too because the current fit logic calls fit for the transformer too.
>  The problem is the same: you cannot pass fit parameters to the
underlying regressor apart from the sample_weight.

Yes, let's fix this and assume all fit params should be passed to the
regressor.

> Another question is if there are cases where you would want to pass fit
parameters to the transformer too because the current fit logic calls fit
for the transformer too.

We'll deal with this in the world where
https://github.com/scikit-learn/enhancement_proposals/pull/16 eventually
gets completed, approved, merged and implemented!!

Pull request welcome.

i will start working 
```

## Patch

```diff
diff --git a/sklearn/compose/_target.py b/sklearn/compose/_target.py
--- a/sklearn/compose/_target.py
+++ b/sklearn/compose/_target.py
@@ -148,7 +148,7 @@ def _fit_transformer(self, y):
                               " you are sure you want to proceed regardless"
                               ", set 'check_inverse=False'", UserWarning)
 
-    def fit(self, X, y, sample_weight=None):
+    def fit(self, X, y, **fit_params):
         """Fit the model according to the given training data.
 
         Parameters
@@ -160,9 +160,10 @@ def fit(self, X, y, sample_weight=None):
         y : array-like, shape (n_samples,)
             Target values.
 
-        sample_weight : array-like, shape (n_samples,) optional
-            Array of weights that are assigned to individual samples.
-            If not provided, then each sample is given unit weight.
+        **fit_params : dict of string -> object
+            Parameters passed to the ``fit`` method of the underlying
+            regressor.
+
 
         Returns
         -------
@@ -197,10 +198,7 @@ def fit(self, X, y, sample_weight=None):
         else:
             self.regressor_ = clone(self.regressor)
 
-        if sample_weight is None:
-            self.regressor_.fit(X, y_trans)
-        else:
-            self.regressor_.fit(X, y_trans, sample_weight=sample_weight)
+        self.regressor_.fit(X, y_trans, **fit_params)
 
         return self
 

```

## Test Patch

```diff
diff --git a/sklearn/compose/tests/test_target.py b/sklearn/compose/tests/test_target.py
--- a/sklearn/compose/tests/test_target.py
+++ b/sklearn/compose/tests/test_target.py
@@ -14,6 +14,8 @@
 from sklearn.preprocessing import FunctionTransformer
 from sklearn.preprocessing import StandardScaler
 
+from sklearn.pipeline import Pipeline
+
 from sklearn.linear_model import LinearRegression, Lasso
 
 from sklearn import datasets
@@ -294,3 +296,39 @@ def test_transform_target_regressor_count_fit(check_inverse):
     )
     ttr.fit(X, y)
     assert ttr.transformer_.fit_counter == 1
+
+
+class DummyRegressorWithExtraFitParams(DummyRegressor):
+    def fit(self, X, y, sample_weight=None, check_input=True):
+        # on the test below we force this to false, we make sure this is
+        # actually passed to the regressor
+        assert not check_input
+        return super().fit(X, y, sample_weight)
+
+
+def test_transform_target_regressor_pass_fit_parameters():
+    X, y = friedman
+    regr = TransformedTargetRegressor(
+        regressor=DummyRegressorWithExtraFitParams(),
+        transformer=DummyTransformer()
+    )
+
+    regr.fit(X, y, check_input=False)
+    assert regr.transformer_.fit_counter == 1
+
+
+def test_transform_target_regressor_route_pipeline():
+    X, y = friedman
+
+    regr = TransformedTargetRegressor(
+        regressor=DummyRegressorWithExtraFitParams(),
+        transformer=DummyTransformer()
+    )
+    estimators = [
+        ('normalize', StandardScaler()), ('est', regr)
+    ]
+
+    pip = Pipeline(estimators)
+    pip.fit(X, y, **{'est__check_input': False})
+
+    assert regr.transformer_.fit_counter == 1

```


## Code snippets

### 1 - examples/compose/plot_transformed_target.py:

Start line: 99, End line: 179

```python
ax0.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax0.set_xlim([0, 2000])
ax0.set_ylim([0, 2000])

regr_trans = TransformedTargetRegressor(regressor=RidgeCV(),
                                        func=np.log1p,
                                        inverse_func=np.expm1)
regr_trans.fit(X_train, y_train)
y_pred = regr_trans.predict(X_test)

ax1.scatter(y_test, y_pred)
ax1.plot([0, 2000], [0, 2000], '--k')
ax1.set_ylabel('Target predicted')
ax1.set_xlabel('True Target')
ax1.set_title('Ridge regression \n with target transformation')
ax1.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax1.set_xlim([0, 2000])
ax1.set_ylim([0, 2000])

f.suptitle("Synthetic data", y=0.035)
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

###############################################################################
# Real-world data set
###############################################################################

###############################################################################
# In a similar manner, the boston housing data set is used to show the impact
# of transforming the targets before learning a model. In this example, the
# targets to be predicted corresponds to the weighted distances to the five
# Boston employment centers.

from sklearn.datasets import load_boston
from sklearn.preprocessing import QuantileTransformer, quantile_transform

dataset = load_boston()
target = np.array(dataset.feature_names) == "DIS"
X = dataset.data[:, np.logical_not(target)]
y = dataset.data[:, target].squeeze()
y_trans = quantile_transform(dataset.data[:, target],
                             n_quantiles=300,
                             output_distribution='normal',
                             copy=True).squeeze()

###############################################################################
# A :class:`sklearn.preprocessing.QuantileTransformer` is used such that the
# targets follows a normal distribution before applying a
# :class:`sklearn.linear_model.RidgeCV` model.

f, (ax0, ax1) = plt.subplots(1, 2)

ax0.hist(y, bins=100, **density_param)
ax0.set_ylabel('Probability')
ax0.set_xlabel('Target')
ax0.set_title('Target distribution')

ax1.hist(y_trans, bins=100, **density_param)
ax1.set_ylabel('Probability')
ax1.set_xlabel('Target')
ax1.set_title('Transformed target distribution')

f.suptitle("Boston housing data: distance to employment centers", y=0.035)
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

###############################################################################
# The effect of the transformer is weaker than on the synthetic data. However,
# the transform induces a decrease of the MAE.

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

regr = RidgeCV()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

ax0.scatter(y_test, y_pred)
ax0.plot([0, 10], [0, 10], '--k')
ax0.set_ylabel('Target predicted')
```
### 2 - examples/compose/plot_transformed_target.py:

Start line: 1, End line: 98

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
======================================================
Effect of transforming the targets in regression model
======================================================

In this example, we give an overview of the
:class:`sklearn.compose.TransformedTargetRegressor`. Two examples
illustrate the benefit of transforming the targets before learning a linear
regression model. The first example uses synthetic data while the second
example is based on the Boston housing data set.

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion

print(__doc__)

###############################################################################
# Synthetic example
###############################################################################

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import median_absolute_error, r2_score


# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}

###############################################################################
# A synthetic random regression problem is generated. The targets ``y`` are
# modified by: (i) translating all targets such that all entries are
# non-negative and (ii) applying an exponential function to obtain non-linear
# targets which cannot be fitted using a simple linear model.
#
# Therefore, a logarithmic (`np.log1p`) and an exponential function
# (`np.expm1`) will be used to transform the targets before training a linear
# regression model and using it for prediction.

X, y = make_regression(n_samples=10000, noise=100, random_state=0)
y = np.exp((y + abs(y.min())) / 200)
y_trans = np.log1p(y)

###############################################################################
# The following illustrate the probability density functions of the target
# before and after applying the logarithmic functions.

f, (ax0, ax1) = plt.subplots(1, 2)

ax0.hist(y, bins=100, **density_param)
ax0.set_xlim([0, 2000])
ax0.set_ylabel('Probability')
ax0.set_xlabel('Target')
ax0.set_title('Target distribution')

ax1.hist(y_trans, bins=100, **density_param)
ax1.set_ylabel('Probability')
ax1.set_xlabel('Target')
ax1.set_title('Transformed target distribution')

f.suptitle("Synthetic data", y=0.035)
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

###############################################################################
# At first, a linear model will be applied on the original targets. Due to the
# non-linearity, the model trained will not be precise during the
# prediction. Subsequently, a logarithmic function is used to linearize the
# targets, allowing better prediction even with a similar linear model as
# reported by the median absolute error (MAE).

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

regr = RidgeCV()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

ax0.scatter(y_test, y_pred)
ax0.plot([0, 2000], [0, 2000], '--k')
ax0.set_ylabel('Target predicted')
ax0.set_xlabel('True Target')
ax0.set_title('Ridge regression \n without target transformation')
```
### 3 - sklearn/compose/_target.py:

Start line: 107, End line: 149

```python
class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, regressor=None, transformer=None,
                 func=None, inverse_func=None, check_inverse=True):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse

    def _fit_transformer(self, y):
        """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        """
        if (self.transformer is not None and
                (self.func is not None or self.inverse_func is not None)):
            raise ValueError("'transformer' and functions 'func'/"
                             "'inverse_func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must"
                                 " also be provided")
            self.transformer_ = FunctionTransformer(
                func=self.func, inverse_func=self.inverse_func, validate=True,
                check_inverse=self.check_inverse)
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel,
                               self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are"
                              " not strictly inverse of each other. If"
                              " you are sure you want to proceed regardless"
                              ", set 'check_inverse=False'", UserWarning)
```
### 4 - sklearn/pipeline.py:

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
### 5 - examples/compose/plot_transformed_target.py:

Start line: 180, End line: 208

```python
ax0.set_xlabel('True Target')
ax0.set_title('Ridge regression \n without target transformation')
ax0.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax0.set_xlim([0, 10])
ax0.set_ylim([0, 10])

regr_trans = TransformedTargetRegressor(
    regressor=RidgeCV(),
    transformer=QuantileTransformer(n_quantiles=300,
                                    output_distribution='normal'))
regr_trans.fit(X_train, y_train)
y_pred = regr_trans.predict(X_test)

ax1.scatter(y_test, y_pred)
ax1.plot([0, 10], [0, 10], '--k')
ax1.set_ylabel('Target predicted')
ax1.set_xlabel('True Target')
ax1.set_title('Ridge regression \n with target transformation')
ax1.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
ax1.set_xlim([0, 10])
ax1.set_ylim([0, 10])

f.suptitle("Boston housing data: distance to employment centers", y=0.035)
f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

plt.show()
```
### 6 - examples/compose/plot_column_transformer_mixed_types.py:

Start line: 1, End line: 103

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

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Alternatively X and y can be obtained directly from the frame attribute:
# X = titanic.frame.drop('survived', axis=1)
# y = titanic.frame['survived']

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
                      ('classifier', LogisticRegression())])

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

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)

print(("best logistic regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))
```
### 7 - sklearn/compose/_target.py:

Start line: 5, End line: 106

```python
import warnings

import numpy as np

from ..base import BaseEstimator, RegressorMixin, clone
from ..utils.validation import check_is_fitted
from ..utils import check_array, safe_indexing
from ..preprocessing import FunctionTransformer

__all__ = ['TransformedTargetRegressor']


class TransformedTargetRegressor(RegressorMixin, BaseEstimator):
    """Meta-estimator to regress on a transformed target.

    Useful for applying a non-linear transformation in regression
    problems. This transformation can be given as a Transformer such as the
    QuantileTransformer or as a function and its inverse such as ``log`` and
    ``exp``.

    The computation during ``fit`` is::

        regressor.fit(X, func(y))

    or::

        regressor.fit(X, transformer.transform(y))

    The computation during ``predict`` is::

        inverse_func(regressor.predict(X))

    or::

        transformer.inverse_transform(regressor.predict(X))

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Parameters
    ----------
    regressor : object, default=LinearRegression()
        Regressor object such as derived from ``RegressorMixin``. This
        regressor will automatically be cloned each time prior to fitting.

    transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Cannot be
        set at the same time as ``func`` and ``inverse_func``. If
        ``transformer`` is ``None`` as well as ``func`` and ``inverse_func``,
        the transformer will be an identity transformer. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.

    func : function, optional
        Function to apply to ``y`` before passing to ``fit``. Cannot be set at
        the same time as ``transformer``. The function needs to return a
        2-dimensional array. If ``func`` is ``None``, the function used will be
        the identity function.

    inverse_func : function, optional
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as ``transformer`` as well. The function needs to return
        a 2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.

    check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        or ``func`` followed by ``inverse_func`` leads to the original targets.

    Attributes
    ----------
    regressor_ : object
        Fitted regressor.

    transformer_ : object
        Transformer used in ``fit`` and ``predict``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> tt = TransformedTargetRegressor(regressor=LinearRegression(),
    ...                                 func=np.log, inverse_func=np.exp)
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = np.exp(2 * X).ravel()
    >>> tt.fit(X, y)
    TransformedTargetRegressor(...)
    >>> tt.score(X, y)
    1.0
    >>> tt.regressor_.coef_
    array([2.])

    Notes
    -----
    Internally, the target ``y`` is always converted into a 2-dimensional array
    to be used by scikit-learn transformers. At the time of prediction, the
    output will be reshaped to a have the same number of dimensions as ``y``.

    See :ref:`examples/compose/plot_transformed_target.py
    <sphx_glr_auto_examples_compose_plot_transformed_target.py>`.

    """
```
### 8 - examples/ensemble/plot_feature_transformation.py:

Start line: 1, End line: 87

```python
"""
===============================================
Feature transformations with ensembles of trees
===============================================

Transform your features into a higher dimensional, sparse space. Then
train a linear model on these features.

First fit an ensemble of trees (totally random trees, a random
forest, or gradient boosted trees) on the training set. Then each leaf
of each tree in the ensemble is assigned a fixed arbitrary feature
index in a new feature space. These leaf indices are then encoded in a
one-hot fashion.

Each sample goes through the decisions of each tree of the ensemble
and ends up in one leaf per tree. The sample is encoded by setting
feature values for these leaves to 1 and the other feature values to 0.

The resulting transformer has then learned a supervised, sparse,
high-dimensional categorical embedding of the data.

"""

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression(max_iter=1000)
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# Supervised transformation based on gradient boosted trees
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression(max_iter=1000)
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
```
### 9 - sklearn/pipeline.py:

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
### 10 - benchmarks/bench_hist_gradient_boosting_higgsboson.py:

Start line: 59, End line: 124

```python
df = load_data()
target = df.values[:, 0]
data = np.ascontiguousarray(df.values[:, 1:])
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.2, random_state=0)

if subsample is not None:
    data_train, target_train = data_train[:subsample], target_train[:subsample]

n_samples, n_features = data_train.shape
print(f"Training set with {n_samples} records with {n_features} features.")

print("Fitting a sklearn model...")
tic = time()
est = HistGradientBoostingClassifier(loss='binary_crossentropy',
                                     learning_rate=lr,
                                     max_iter=n_trees,
                                     max_bins=max_bins,
                                     max_leaf_nodes=n_leaf_nodes,
                                     n_iter_no_change=None,
                                     random_state=0,
                                     verbose=1)
est.fit(data_train, target_train)
toc = time()
predicted_test = est.predict(data_test)
predicted_proba_test = est.predict_proba(data_test)
roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
acc = accuracy_score(target_test, predicted_test)
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.lightgbm:
    print("Fitting a LightGBM model...")
    tic = time()
    lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')
    lightgbm_est.fit(data_train, target_train)
    toc = time()
    predicted_test = lightgbm_est.predict(data_test)
    predicted_proba_test = lightgbm_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.xgboost:
    print("Fitting an XGBoost model...")
    tic = time()
    xgboost_est = get_equivalent_estimator(est, lib='xgboost')
    xgboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = xgboost_est.predict(data_test)
    predicted_proba_test = xgboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.catboost:
    print("Fitting a Catboost model...")
    tic = time()
    catboost_est = get_equivalent_estimator(est, lib='catboost')
    catboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = catboost_est.predict(data_test)
    predicted_proba_test = catboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")
```
### 11 - sklearn/compose/_target.py:

Start line: 151, End line: 205

```python
class TransformedTargetRegressor(RegressorMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        y = check_array(y, accept_sparse=False, force_all_finite=True,
                        ensure_2d=False, dtype='numeric')

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from ..linear_model import LinearRegression
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        if sample_weight is None:
            self.regressor_.fit(X, y_trans)
        else:
            self.regressor_.fit(X, y_trans, sample_weight=sample_weight)

        return self
```
### 80 - sklearn/compose/_target.py:

Start line: 207, End line: 239

```python
class TransformedTargetRegressor(RegressorMixin, BaseEstimator):

    def predict(self, X):
        """Predict using the base regressor, applying inverse.

        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.

        """
        check_is_fitted(self)
        pred = self.regressor_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (self._training_dim == 1 and
                pred_trans.ndim == 2 and pred_trans.shape[1] == 1):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
```
