# scikit-learn__scikit-learn-10198

| **scikit-learn/scikit-learn** | `726fa36f2556e0d604d85a1de48ba56a8b6550db` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 2691 |
| **Avg pos** | 6.0 |
| **Min pos** | 6 |
| **Max pos** | 6 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/preprocessing/_encoders.py b/sklearn/preprocessing/_encoders.py
--- a/sklearn/preprocessing/_encoders.py
+++ b/sklearn/preprocessing/_encoders.py
@@ -240,6 +240,8 @@ class OneHotEncoder(_BaseEncoder):
     >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
     array([['Male', 1],
            [None, 2]], dtype=object)
+    >>> enc.get_feature_names()
+    array(['x0_Female', 'x0_Male', 'x1_1', 'x1_2', 'x1_3'], dtype=object)
 
     See also
     --------
@@ -639,6 +641,38 @@ def inverse_transform(self, X):
 
         return X_tr
 
+    def get_feature_names(self, input_features=None):
+        """Return feature names for output features.
+
+        Parameters
+        ----------
+        input_features : list of string, length n_features, optional
+            String names for input features if available. By default,
+            "x0", "x1", ... "xn_features" is used.
+
+        Returns
+        -------
+        output_feature_names : array of string, length n_output_features
+
+        """
+        check_is_fitted(self, 'categories_')
+        cats = self.categories_
+        if input_features is None:
+            input_features = ['x%d' % i for i in range(len(cats))]
+        elif(len(input_features) != len(self.categories_)):
+            raise ValueError(
+                "input_features should have length equal to number of "
+                "features ({}), got {}".format(len(self.categories_),
+                                               len(input_features)))
+
+        feature_names = []
+        for i in range(len(cats)):
+            names = [
+                input_features[i] + '_' + six.text_type(t) for t in cats[i]]
+            feature_names.extend(names)
+
+        return np.array(feature_names, dtype=object)
+
 
 class OrdinalEncoder(_BaseEncoder):
     """Encode categorical features as an integer array.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/preprocessing/_encoders.py | 243 | 243 | 6 | 2 | 2691
| sklearn/preprocessing/_encoders.py | 642 | 642 | - | 2 | -


## Problem Statement

```
add get_feature_names to CategoricalEncoder
We should add a ``get_feature_names`` to the new CategoricalEncoder, as discussed [here](https://github.com/scikit-learn/scikit-learn/pull/9151#issuecomment-345830056). I think it would be good to be consistent with the PolynomialFeature which allows passing in original feature names to map them to new feature names. Also see #6425.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/preprocessing/data.py | 1348 | 1386| 294 | 294 | 21857 | 
| 2 | **2 sklearn/preprocessing/_encoders.py** | 33 | 74| 328 | 622 | 28299 | 
| 3 | **2 sklearn/preprocessing/_encoders.py** | 259 | 290| 266 | 888 | 28299 | 
| 4 | **2 sklearn/preprocessing/_encoders.py** | 525 | 553| 262 | 1150 | 28299 | 
| 5 | 2 sklearn/preprocessing/data.py | 2672 | 2684| 113 | 1263 | 28299 | 
| **-> 6 <-** | **2 sklearn/preprocessing/_encoders.py** | 111 | 257| 1428 | 2691 | 28299 | 
| 7 | **2 sklearn/preprocessing/_encoders.py** | 292 | 375| 797 | 3488 | 28299 | 
| 8 | 3 sklearn/pipeline.py | 713 | 729| 136 | 3624 | 34892 | 
| 9 | **3 sklearn/preprocessing/_encoders.py** | 643 | 743| 746 | 4370 | 34892 | 
| 10 | **3 sklearn/preprocessing/_encoders.py** | 405 | 458| 562 | 4932 | 34892 | 
| 11 | **3 sklearn/preprocessing/_encoders.py** | 485 | 523| 410 | 5342 | 34892 | 
| 12 | **3 sklearn/preprocessing/_encoders.py** | 76 | 108| 297 | 5639 | 34892 | 
| 13 | 4 sklearn/feature_extraction/text.py | 825 | 837| 122 | 5761 | 47604 | 
| 14 | 4 sklearn/pipeline.py | 515 | 533| 142 | 5903 | 47604 | 
| 15 | 5 sklearn/feature_extraction/dict_vectorizer.py | 274 | 318| 314 | 6217 | 50328 | 
| 16 | 5 sklearn/feature_extraction/dict_vectorizer.py | 103 | 135| 213 | 6430 | 50328 | 
| 17 | 5 sklearn/feature_extraction/text.py | 1 | 42| 185 | 6615 | 50328 | 
| 18 | **5 sklearn/preprocessing/_encoders.py** | 5 | 30| 117 | 6732 | 50328 | 
| 19 | 5 sklearn/preprocessing/data.py | 1410 | 1452| 348 | 7080 | 50328 | 
| 20 | 5 sklearn/preprocessing/data.py | 1833 | 1889| 546 | 7626 | 50328 | 
| 21 | 6 sklearn/compose/_column_transformer.py | 262 | 298| 279 | 7905 | 55453 | 
| 22 | 7 sklearn/preprocessing/__init__.py | 1 | 73| 441 | 8346 | 55894 | 
| 23 | 8 examples/ensemble/plot_feature_transformation.py | 1 | 91| 765 | 9111 | 57017 | 
| 24 | 9 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 9904 | 58877 | 
| 25 | 10 sklearn/preprocessing/label.py | 77 | 110| 276 | 10180 | 66349 | 
| 26 | 10 sklearn/feature_extraction/dict_vectorizer.py | 26 | 101| 683 | 10863 | 66349 | 
| 27 | 10 sklearn/preprocessing/label.py | 40 | 57| 138 | 11001 | 66349 | 
| 28 | 10 sklearn/feature_extraction/dict_vectorizer.py | 137 | 211| 606 | 11607 | 66349 | 
| 29 | 11 sklearn/ensemble/gradient_boosting.py | 926 | 955| 321 | 11928 | 84627 | 
| 30 | 12 sklearn/feature_extraction/hashing.py | 87 | 99| 139 | 12067 | 86157 | 
| 31 | 12 sklearn/preprocessing/data.py | 10 | 66| 348 | 12415 | 86157 | 
| 32 | 13 sklearn/utils/estimator_checks.py | 1201 | 1224| 185 | 12600 | 106633 | 
| 33 | 14 sklearn/base.py | 128 | 164| 314 | 12914 | 110526 | 
| 34 | 14 sklearn/preprocessing/label.py | 60 | 74| 133 | 13047 | 110526 | 
| 35 | 15 sklearn/impute.py | 693 | 735| 428 | 13475 | 120319 | 
| 36 | **15 sklearn/preprocessing/_encoders.py** | 575 | 640| 515 | 13990 | 120319 | 
| 37 | 15 sklearn/utils/estimator_checks.py | 798 | 829| 297 | 14287 | 120319 | 
| 38 | 16 sklearn/feature_extraction/__init__.py | 1 | 14| 0 | 14287 | 120418 | 
| 39 | 16 sklearn/preprocessing/data.py | 2510 | 2527| 144 | 14431 | 120418 | 
| 40 | 16 sklearn/utils/estimator_checks.py | 1952 | 1972| 233 | 14664 | 120418 | 
| 41 | 17 sklearn/utils/metaestimators.py | 61 | 72| 151 | 14815 | 122071 | 
| 42 | 17 sklearn/utils/estimator_checks.py | 1574 | 1603| 324 | 15139 | 122071 | 
| 43 | 17 sklearn/utils/estimator_checks.py | 1651 | 1686| 399 | 15538 | 122071 | 
| 44 | 17 sklearn/feature_extraction/dict_vectorizer.py | 320 | 366| 349 | 15887 | 122071 | 
| 45 | 17 sklearn/utils/estimator_checks.py | 1975 | 1989| 211 | 16098 | 122071 | 
| 46 | 17 sklearn/utils/estimator_checks.py | 639 | 687| 441 | 16539 | 122071 | 
| 47 | 18 sklearn/feature_selection/__init__.py | 1 | 44| 286 | 16825 | 122357 | 
| 48 | 18 sklearn/preprocessing/data.py | 1273 | 1346| 733 | 17558 | 122357 | 
| 49 | 19 sklearn/feature_selection/rfe.py | 142 | 223| 662 | 18220 | 126372 | 
| 50 | **19 sklearn/preprocessing/_encoders.py** | 745 | 780| 266 | 18486 | 126372 | 
| 51 | 19 sklearn/feature_extraction/text.py | 115 | 144| 170 | 18656 | 126372 | 
| 52 | 19 sklearn/impute.py | 653 | 691| 330 | 18986 | 126372 | 
| 53 | 19 sklearn/feature_extraction/text.py | 228 | 256| 252 | 19238 | 126372 | 
| 54 | 20 sklearn/feature_selection/from_model.py | 4 | 34| 228 | 19466 | 128174 | 
| 55 | 20 sklearn/utils/estimator_checks.py | 1885 | 1920| 352 | 19818 | 128174 | 
| 56 | 20 sklearn/pipeline.py | 688 | 711| 195 | 20013 | 128174 | 
| 57 | 20 sklearn/ensemble/gradient_boosting.py | 1219 | 1247| 237 | 20250 | 128174 | 
| 58 | 20 sklearn/feature_extraction/text.py | 1248 | 1586| 140 | 20390 | 128174 | 
| 59 | 21 sklearn/cluster/_feature_agglomeration.py | 1 | 59| 390 | 20780 | 128745 | 
| 60 | 21 sklearn/feature_extraction/text.py | 294 | 320| 230 | 21010 | 128745 | 
| 61 | 21 sklearn/feature_extraction/dict_vectorizer.py | 213 | 231| 144 | 21154 | 128745 | 
| 62 | 22 examples/preprocessing/plot_scaling_importance.py | 1 | 82| 762 | 21916 | 129948 | 
| 63 | 23 examples/ensemble/plot_forest_importances.py | 1 | 55| 373 | 22289 | 130321 | 
| 64 | 24 sklearn/ensemble/forest.py | 358 | 388| 239 | 22528 | 147316 | 
| 65 | 25 sklearn/ensemble/weight_boosting.py | 237 | 272| 283 | 22811 | 156086 | 
| 66 | 26 sklearn/decomposition/dict_learning.py | 866 | 902| 267 | 23078 | 167384 | 
| 67 | 26 sklearn/utils/estimator_checks.py | 850 | 864| 169 | 23247 | 167384 | 
| 68 | 26 sklearn/feature_selection/rfe.py | 36 | 116| 759 | 24006 | 167384 | 
| 69 | 26 sklearn/utils/estimator_checks.py | 557 | 592| 328 | 24334 | 167384 | 
| 70 | 27 examples/compose/plot_feature_union.py | 1 | 61| 411 | 24745 | 167819 | 
| 71 | 27 sklearn/preprocessing/data.py | 1388 | 1408| 133 | 24878 | 167819 | 
| 72 | 27 sklearn/utils/estimator_checks.py | 541 | 554| 166 | 25044 | 167819 | 
| 73 | 27 sklearn/utils/estimator_checks.py | 85 | 118| 257 | 25301 | 167819 | 
| 74 | 27 sklearn/feature_selection/rfe.py | 279 | 384| 955 | 26256 | 167819 | 
| 75 | 27 sklearn/utils/estimator_checks.py | 517 | 538| 272 | 26528 | 167819 | 
| 76 | 27 sklearn/utils/estimator_checks.py | 1 | 82| 708 | 27236 | 167819 | 
| 77 | 27 sklearn/utils/estimator_checks.py | 867 | 889| 248 | 27484 | 167819 | 
| 78 | 27 sklearn/preprocessing/data.py | 2392 | 2468| 649 | 28133 | 167819 | 
| 79 | 27 sklearn/feature_extraction/text.py | 839 | 878| 376 | 28509 | 167819 | 
| 80 | 27 sklearn/decomposition/dict_learning.py | 113 | 181| 692 | 29201 | 167819 | 
| 81 | 27 sklearn/feature_extraction/text.py | 792 | 823| 319 | 29520 | 167819 | 
| 82 | 27 sklearn/feature_extraction/text.py | 880 | 940| 467 | 29987 | 167819 | 
| 83 | 27 sklearn/utils/estimator_checks.py | 1717 | 1759| 474 | 30461 | 167819 | 
| 84 | 28 sklearn/cluster/hierarchical.py | 844 | 920| 744 | 31205 | 176085 | 
| 85 | 28 sklearn/utils/estimator_checks.py | 1786 | 1828| 449 | 31654 | 176085 | 
| 86 | 28 sklearn/feature_extraction/hashing.py | 4 | 85| 795 | 32449 | 176085 | 
| 87 | 28 sklearn/utils/estimator_checks.py | 1305 | 1336| 258 | 32707 | 176085 | 
| 88 | 28 sklearn/utils/estimator_checks.py | 1034 | 1057| 269 | 32976 | 176085 | 
| 89 | 28 sklearn/feature_extraction/hashing.py | 134 | 172| 347 | 33323 | 176085 | 
| 90 | 29 sklearn/metrics/pairwise.py | 1467 | 1477| 107 | 33430 | 189222 | 
| 91 | 30 sklearn/decomposition/online_lda.py | 316 | 341| 243 | 33673 | 195812 | 
| 92 | 31 examples/compose/plot_compare_reduction.py | 1 | 105| 809 | 34482 | 196868 | 
| 93 | 31 sklearn/ensemble/gradient_boosting.py | 1286 | 1527| 2411 | 36893 | 196868 | 
| 94 | **31 sklearn/preprocessing/_encoders.py** | 555 | 573| 121 | 37014 | 196868 | 
| 95 | 31 sklearn/cluster/hierarchical.py | 581 | 601| 134 | 37148 | 196868 | 
| 96 | 31 sklearn/utils/estimator_checks.py | 969 | 1003| 372 | 37520 | 196868 | 
| 97 | 31 sklearn/feature_selection/from_model.py | 81 | 140| 590 | 38110 | 196868 | 
| 98 | 31 sklearn/utils/estimator_checks.py | 892 | 966| 702 | 38812 | 196868 | 
| 99 | 31 sklearn/utils/estimator_checks.py | 1689 | 1714| 277 | 39089 | 196868 | 
| 100 | **31 sklearn/preprocessing/_encoders.py** | 377 | 403| 178 | 39267 | 196868 | 


### Hint

```
I'd like to try this one.
If you haven't contributed before, I suggest you try an issue labeled "good first issue". Though this one isn't too hard, eigher.
@amueller 
I think I can handle it.
So we want something like this right?

    enc.fit([['male',0], ['female', 1]])
    enc.get_feature_names()

    >> ['female', 'male', 0, 1]

Can you please give an example of how original feature names can map to new feature names? I have seen the `get_feature_names()` from PolynomialFeatures, but I don't understand what that means in this case.
I think the idea is that if you have multiple input features containing the
value "hello" they need to be distinguished in the feature names listed for
output. so you prefix the value with the input feature name, defaulting to
x1 etc as in polynomial. clearer?

@jnothman Is this what you mean?

    enc.fit(  [ [ 'male' ,    0,  1],
                 [ 'female' ,  1 , 0]  ] )

    enc.get_feature_names(['one','two','three'])

    >> ['one_female', 'one_male' , 'two_0' , 'two_1' , 'three_0' , 'three_1']


And in case I don't pass any strings, it should just use `x0` , `x1` and so on for the prefixes right?
Precisely.

>
>

I like the idea to be able to specify input feature names.

Regarding syntax of combining the two names, as prior art we have eg `DictVectorizer` that does something like `['0=female', '0=male', '1=0', '1=1']` (assuming we use 0 and 1 as the column names for arrays) or Pipelines that uses double underscores (`['0__female', '0__male', '1__0', '1__1']`). Others? 
I personally like the `__` a bit more I think, but the fact that this is used by pipelines is for me actually a reason to use `=` in this case. Eg in combination with the ColumnTransformer (assuming this would use the `__` syntax like pipeline), you could then get a feature name like `'cat__0=male'` instead of `'cat__0__male'`.
Additional question:

- if the input is a pandas DataFrame, do we want to preserve the column names (to use instead of 0, 1, ..)? 
  (ideally yes IMO, but this would require some extra code as currently it is not detected whether a DataFrame is passed or not, it is just coerced to array)
no, we shouldn't use column names automatically. it's hard for us to keep
them and easy for the user to pass them.

>  it's hard for us to keep them

It's not really 'hard':

\`\`\`
class CategoricalEncoder():

    def fit(self, X, ...):
        ...
        if hasattr(X, 'iloc'):
            self._input_features = X.columns
        ...

    def get_feature_names(self, input_features=None):
        if input_features is None:
            input_features = self._input_features
        ...
\`\`\`

but of course it is added complexity, and more explicit support for pandas dataframes, which is not necessarily something we want to add (I just don't think 'hard' is the correct reason :-)).

But eg if you combine multiple sets of columns and transformers in a ColumnTransformer, it is not always that straightforward for the user to keep track of IMO, because you then need to combine the different sets of selected column into one list to pass to `get_feature_names`.
No, then you just need get_feature_names implemented everywhere and let
Pipeline's (not yet) implementation of get_feature_names handle it for you.
(Note: There remain some problems with this design in a meta-estimator
context.) I've implemented similar within the eli5 package, but we also got
somewhat stuck when it came to making arbitrary decisions about how to make
feature names for linear transforms like PCA. A structured representation
rather than a string name might be nice...

On 23 November 2017 at 10:00, Joris Van den Bossche <
notifications@github.com> wrote:

> it's hard for us to keep them
>
> It's not really 'hard':
>
> class CategoricalEncoder():
>
>     def fit(self, X, ...):
>         ...
>         if hasattr(X, 'iloc'):
>             self._input_features = X.columns
>         ...
>
>     def get_feature_names(self, input_features=None):
>         if input_features is None:
>             input_features = self._input_features
>         ...
>
> but of course it is added complexity, and more explicit support for pandas
> dataframes, which is not necessarily something we want to add (I just don't
> think 'hard' is the correct reason :-)).
>
> But eg if you combine multiple sets of columns and transformers in a
> ColumnTransformer, it is not always that straightforward for the user to
> keep track of IMO, because you then need to combine the different sets of
> selected column into one list to pass to get_feature_names.
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/10181#issuecomment-346495657>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz62rb6pYYTi80NzltL4u4biA3_-ARks5s5KePgaJpZM4Ql59C>
> .
>

```

## Patch

```diff
diff --git a/sklearn/preprocessing/_encoders.py b/sklearn/preprocessing/_encoders.py
--- a/sklearn/preprocessing/_encoders.py
+++ b/sklearn/preprocessing/_encoders.py
@@ -240,6 +240,8 @@ class OneHotEncoder(_BaseEncoder):
     >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
     array([['Male', 1],
            [None, 2]], dtype=object)
+    >>> enc.get_feature_names()
+    array(['x0_Female', 'x0_Male', 'x1_1', 'x1_2', 'x1_3'], dtype=object)
 
     See also
     --------
@@ -639,6 +641,38 @@ def inverse_transform(self, X):
 
         return X_tr
 
+    def get_feature_names(self, input_features=None):
+        """Return feature names for output features.
+
+        Parameters
+        ----------
+        input_features : list of string, length n_features, optional
+            String names for input features if available. By default,
+            "x0", "x1", ... "xn_features" is used.
+
+        Returns
+        -------
+        output_feature_names : array of string, length n_output_features
+
+        """
+        check_is_fitted(self, 'categories_')
+        cats = self.categories_
+        if input_features is None:
+            input_features = ['x%d' % i for i in range(len(cats))]
+        elif(len(input_features) != len(self.categories_)):
+            raise ValueError(
+                "input_features should have length equal to number of "
+                "features ({}), got {}".format(len(self.categories_),
+                                               len(input_features)))
+
+        feature_names = []
+        for i in range(len(cats)):
+            names = [
+                input_features[i] + '_' + six.text_type(t) for t in cats[i]]
+            feature_names.extend(names)
+
+        return np.array(feature_names, dtype=object)
+
 
 class OrdinalEncoder(_BaseEncoder):
     """Encode categorical features as an integer array.

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_encoders.py b/sklearn/preprocessing/tests/test_encoders.py
--- a/sklearn/preprocessing/tests/test_encoders.py
+++ b/sklearn/preprocessing/tests/test_encoders.py
@@ -1,3 +1,4 @@
+# -*- coding: utf-8 -*-
 from __future__ import division
 
 import re
@@ -455,6 +456,47 @@ def test_one_hot_encoder_pandas():
     assert_allclose(Xtr, [[1, 0, 1, 0], [0, 1, 0, 1]])
 
 
+def test_one_hot_encoder_feature_names():
+    enc = OneHotEncoder()
+    X = [['Male', 1, 'girl', 2, 3],
+         ['Female', 41, 'girl', 1, 10],
+         ['Male', 51, 'boy', 12, 3],
+         ['Male', 91, 'girl', 21, 30]]
+
+    enc.fit(X)
+    feature_names = enc.get_feature_names()
+    assert isinstance(feature_names, np.ndarray)
+
+    assert_array_equal(['x0_Female', 'x0_Male',
+                        'x1_1', 'x1_41', 'x1_51', 'x1_91',
+                        'x2_boy', 'x2_girl',
+                        'x3_1', 'x3_2', 'x3_12', 'x3_21',
+                        'x4_3',
+                        'x4_10', 'x4_30'], feature_names)
+
+    feature_names2 = enc.get_feature_names(['one', 'two',
+                                            'three', 'four', 'five'])
+
+    assert_array_equal(['one_Female', 'one_Male',
+                        'two_1', 'two_41', 'two_51', 'two_91',
+                        'three_boy', 'three_girl',
+                        'four_1', 'four_2', 'four_12', 'four_21',
+                        'five_3', 'five_10', 'five_30'], feature_names2)
+
+    with pytest.raises(ValueError, match="input_features should have length"):
+        enc.get_feature_names(['one', 'two'])
+
+
+def test_one_hot_encoder_feature_names_unicode():
+    enc = OneHotEncoder()
+    X = np.array([[u'c❤t1', u'dat2']], dtype=object).T
+    enc.fit(X)
+    feature_names = enc.get_feature_names()
+    assert_array_equal([u'x0_c❤t1', u'x0_dat2'], feature_names)
+    feature_names = enc.get_feature_names(input_features=[u'n👍me'])
+    assert_array_equal([u'n👍me_c❤t1', u'n👍me_dat2'], feature_names)
+
+
 @pytest.mark.parametrize("X", [
     [['abc', 2, 55], ['def', 1, 55]],
     np.array([[10, 2, 55], [20, 1, 55]]),

```


## Code snippets

### 1 - sklearn/preprocessing/data.py:

Start line: 1348, End line: 1386

```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):

    @property
    def powers_(self):
        check_is_fitted(self, 'n_input_features_')

        combinations = self._combinations(self.n_input_features_, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        return np.vstack(np.bincount(c, minlength=self.n_input_features_)
                         for c in combinations)

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features

        """
        powers = self.powers_
        if input_features is None:
            input_features = ['x%d' % i for i in range(powers.shape[1])]
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join("%s^%d" % (input_features[ind], exp)
                                if exp != 1 else input_features[ind]
                                for ind, exp in zip(inds, row[inds]))
            else:
                name = "1"
            feature_names.append(name)
        return feature_names
```
### 2 - sklearn/preprocessing/_encoders.py:

Start line: 33, End line: 74

```python
class _BaseEncoder(BaseEstimator, TransformerMixin):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _fit(self, X, handle_unknown='error'):

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        if self._categories != 'auto':
            if X.dtype != object:
                for cats in self._categories:
                    if not np.all(np.sort(cats) == np.array(cats)):
                        raise ValueError("Unsorted categories are not "
                                         "supported for numerical categories")
            if len(self._categories) != n_features:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")

        self.categories_ = []

        for i in range(n_features):
            Xi = X[:, i]
            if self._categories == 'auto':
                cats = _encode(Xi)
            else:
                cats = np.array(self._categories[i], dtype=X.dtype)
                if self.handle_unknown == 'error':
                    diff = _encode_check_unknown(Xi, cats)
                    if diff:
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
            self.categories_.append(cats)
```
### 3 - sklearn/preprocessing/_encoders.py:

Start line: 259, End line: 290

```python
class OneHotEncoder(_BaseEncoder):

    def __init__(self, n_values=None, categorical_features=None,
                 categories=None, sparse=True, dtype=np.float64,
                 handle_unknown='error'):
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.n_values = n_values
        self.categorical_features = categorical_features

    # Deprecated attributes

    @property
    @deprecated("The ``active_features_`` attribute was deprecated in version "
                "0.20 and will be removed 0.22.")
    def active_features_(self):
        check_is_fitted(self, 'categories_')
        return self._active_features_

    @property
    @deprecated("The ``feature_indices_`` attribute was deprecated in version "
                "0.20 and will be removed 0.22.")
    def feature_indices_(self):
        check_is_fitted(self, 'categories_')
        return self._feature_indices_

    @property
    @deprecated("The ``n_values_`` attribute was deprecated in version "
                "0.20 and will be removed 0.22.")
    def n_values_(self):
        check_is_fitted(self, 'categories_')
        return self._n_values_
```
### 4 - sklearn/preprocessing/_encoders.py:

Start line: 525, End line: 553

```python
class OneHotEncoder(_BaseEncoder):

    def _transform_new(self, X):
        """New implementation assuming categorical input"""
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if not self.sparse:
            return out.toarray()
        else:
            return out
```
### 5 - sklearn/preprocessing/data.py:

Start line: 2672, End line: 2684

```python
class CategoricalEncoder:
    """
    CategoricalEncoder briefly existed in 0.20dev. Its functionality
    has been rolled into the OneHotEncoder and OrdinalEncoder.
    This stub will be removed in version 0.21.
    """

    def __init__(*args, **kwargs):
        raise RuntimeError(
            "CategoricalEncoder briefly existed in 0.20dev. Its functionality "
            "has been rolled into the OneHotEncoder and OrdinalEncoder. "
            "This stub will be removed in version 0.21.")
```
### 6 - sklearn/preprocessing/_encoders.py:

Start line: 111, End line: 257

```python
class OneHotEncoder(_BaseEncoder):
    """Encode categorical integer features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array.

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.
    The OneHotEncoder previously assumed that the input features take on
    values in the range [0, max(values)). This behaviour is deprecated.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=np.float
        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    n_values : 'auto', int or array of ints
        Number of values per feature.

        - 'auto' : determine value range from training data.
        - int : number of categorical values per feature.
                Each feature value should be in ``range(n_values)``
        - array : ``n_values[i]`` is the number of categorical values in
                  ``X[:, i]``. Each feature value should be
                  in ``range(n_values[i])``

        .. deprecated:: 0.20
            The `n_values` keyword was deprecated in version 0.20 and will
            be removed in 0.22. Use `categories` instead.

    categorical_features : "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

        .. deprecated:: 0.20
            The `categorical_features` keyword was deprecated in version
            0.20 and will be removed in 0.22.
            You can use the ``ColumnTransformer`` instead.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    active_features_ : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

        .. deprecated:: 0.20
            The ``active_features_`` attribute was deprecated in version
            0.20 and will be removed in 0.22.

    feature_indices_ : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by ``active_features_`` afterwards)

        .. deprecated:: 0.20
            The ``feature_indices_`` attribute was deprecated in version
            0.20 and will be removed in 0.22.

    n_values_ : array of shape (n_features,)
        Maximum number of values per feature.

        .. deprecated:: 0.20
            The ``n_values_`` attribute was deprecated in version
            0.20 and will be removed in 0.22.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features=None, categories=None,
           dtype=<... 'numpy.float64'>, handle_unknown='ignore',
           n_values=None, sparse=True)

    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)

    See also
    --------
    sklearn.preprocessing.OrdinalEncoder : performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    sklearn.preprocessing.LabelBinarizer : binarizes labels in a one-vs-all
      fashion.
    sklearn.preprocessing.MultiLabelBinarizer : transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.
    """
```
### 7 - sklearn/preprocessing/_encoders.py:

Start line: 292, End line: 375

```python
class OneHotEncoder(_BaseEncoder):

    def _handle_deprecations(self, X):

        # internal version of the attributes to handle deprecations
        self._categories = getattr(self, '_categories', None)
        self._categorical_features = getattr(self, '_categorical_features',
                                             None)

        # user manually set the categories or second fit -> never legacy mode
        if self.categories is not None or self._categories is not None:
            self._legacy_mode = False
            if self.categories is not None:
                self._categories = self.categories

        # categories not set -> infer if we need legacy mode or not
        elif self.n_values is not None and self.n_values != 'auto':
            msg = (
                "Passing 'n_values' is deprecated in version 0.20 and will be "
                "removed in 0.22. You can use the 'categories' keyword "
                "instead. 'n_values=n' corresponds to 'categories=[range(n)]'."
            )
            warnings.warn(msg, DeprecationWarning)
            self._legacy_mode = True

        else:  # n_values = 'auto'
            if self.handle_unknown == 'ignore':
                # no change in behaviour, no need to raise deprecation warning
                self._legacy_mode = False
                self._categories = 'auto'
                if self.n_values == 'auto':
                    # user manually specified this
                    msg = (
                        "Passing 'n_values' is deprecated in version 0.20 and "
                        "will be removed in 0.22. n_values='auto' can be "
                        "replaced with categories='auto'."
                    )
                    warnings.warn(msg, DeprecationWarning)
            else:

                # check if we have integer or categorical input
                try:
                    X = check_array(X, dtype=np.int)
                except ValueError:
                    self._legacy_mode = False
                    self._categories = 'auto'
                else:
                    msg = (
                        "The handling of integer data will change in version "
                        "0.22. Currently, the categories are determined "
                        "based on the range [0, max(values)], while in the "
                        "future they will be determined based on the unique "
                        "values.\nIf you want the future behaviour and "
                        "silence this warning, you can specify "
                        "\"categories='auto'\".\n"
                        "In case you used a LabelEncoder before this "
                        "OneHotEncoder to convert the categories to integers, "
                        "then you can now use the OneHotEncoder directly."
                    )
                    warnings.warn(msg, FutureWarning)
                    self._legacy_mode = True
                    self.n_values = 'auto'

        # if user specified categorical_features -> always use legacy mode
        if self.categorical_features is not None:
            if (isinstance(self.categorical_features, six.string_types)
                    and self.categorical_features == 'all'):
                warnings.warn(
                    "The 'categorical_features' keyword is deprecated in "
                    "version 0.20 and will be removed in 0.22. The passed "
                    "value of 'all' is the default and can simply be removed.",
                    DeprecationWarning)
            else:
                if self.categories is not None:
                    raise ValueError(
                        "The 'categorical_features' keyword is deprecated, "
                        "and cannot be used together with specifying "
                        "'categories'.")
                warnings.warn(
                    "The 'categorical_features' keyword is deprecated in "
                    "version 0.20 and will be removed in 0.22. You can "
                    "use the ColumnTransformer instead.", DeprecationWarning)
                self._legacy_mode = True
            self._categorical_features = self.categorical_features
        else:
            self._categorical_features = 'all'
```
### 8 - sklearn/pipeline.py:

Start line: 713, End line: 729

```python
class FeatureUnion(_BaseComposition, TransformerMixin):

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
### 9 - sklearn/preprocessing/_encoders.py:

Start line: 643, End line: 743

```python
class OrdinalEncoder(_BaseEncoder):
    """Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    OrdinalEncoder(categories='auto', dtype=<... 'numpy.float64'>)
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      categorical features.
    sklearn.preprocessing.LabelEncoder : encodes target labels with values
      between 0 and n_classes-1.
    """

    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self

        """
        # base classes uses _categories to deal with deprecations in
        # OneHoteEncoder: can be removed once deprecations are removed
        self._categories = self.categories
        self._fit(X)

        return self

    def transform(self, X):
        """Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.

        """
        X_int, _ = self._transform(X)
        return X_int.astype(self.dtype, copy=False)
```
### 10 - sklearn/preprocessing/_encoders.py:

Start line: 405, End line: 458

```python
class OneHotEncoder(_BaseEncoder):

    def _legacy_fit_transform(self, X):
        """Assumes X contains only categorical features."""
        dtype = getattr(X, 'dtype', None)
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape
        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            if (np.max(X, axis=0) >= self.n_values).any():
                raise ValueError("Feature out of bounds for n_values=%d"
                                 % self.n_values)
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected"
                                " 'auto', int or array of ints, got %r"
                                % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")

        self._n_values_ = n_values
        self.categories_ = [np.arange(n_val - 1, dtype=dtype)
                            for n_val in n_values]
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self._feature_indices_ = indices

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()

        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self._active_features_ = active_features

            self.categories_ = [
                np.unique(X[:, i]).astype(dtype) if dtype
                else np.unique(X[:, i]) for i in range(n_features)]

        return out if self.sparse else out.toarray()
```
### 11 - sklearn/preprocessing/_encoders.py:

Start line: 485, End line: 523

```python
class OneHotEncoder(_BaseEncoder):

    def _legacy_transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape

        indices = self._feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        # We use only those categorical features of X that are known using fit.
        # i.e lesser than n_values_ using mask.
        # This means, if self.handle_unknown is "ignore", the row_indices and
        # col_indices corresponding to the unknown categorical feature are
        # ignored.
        mask = (X < self._n_values_).ravel()
        if np.any(~mask):
            if self.handle_unknown not in ['error', 'ignore']:
                raise ValueError("handle_unknown should be either error or "
                                 "unknown got %s" % self.handle_unknown)
            if self.handle_unknown == 'error':
                raise ValueError("unknown categorical feature present %s "
                                 "during transform." % X.ravel()[~mask])

        column_indices = (X + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(np.sum(mask))
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if (isinstance(self.n_values, six.string_types) and
                self.n_values == 'auto'):
            out = out[:, self._active_features_]

        return out if self.sparse else out.toarray()
```
### 12 - sklearn/preprocessing/_encoders.py:

Start line: 76, End line: 108

```python
class _BaseEncoder(BaseEstimator, TransformerMixin):

    def _transform(self, X, handle_unknown='error'):

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        _, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            diff, valid_mask = _encode_check_unknown(Xi, self.categories_[i],
                                                     return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            _, encoded = _encode(Xi, self.categories_[i], encode=True)
            X_int[:, i] = encoded

        return X_int, X_mask
```
### 18 - sklearn/preprocessing/_encoders.py:

Start line: 5, End line: 30

```python
from __future__ import division

import numbers
import warnings

import numpy as np
from scipy import sparse

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..utils import check_array
from ..utils import deprecated
from ..utils.fixes import _argmax
from ..utils.validation import check_is_fitted

from .base import _transform_selected
from .label import _encode, _encode_check_unknown


range = six.moves.range


__all__ = [
    'OneHotEncoder',
    'OrdinalEncoder'
]
```
### 36 - sklearn/preprocessing/_encoders.py:

Start line: 575, End line: 640

```python
class OneHotEncoder(_BaseEncoder):

    def inverse_transform(self, X):
        """Convert the back data to the original representation.

        In case unknown categories are encountered (all zero's in the
        one-hot encoding), ``None`` is used to represent this category.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.

        """
        # if self._legacy_mode:
        #     raise ValueError("only supported for categorical features")

        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if X.shape[1] != n_transformed_features:
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        j = 0
        found_unknown = {}

        for i in range(n_features):
            n_categories = len(self.categories_[i])
            sub = X[:, j:j + n_categories]

            # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(_argmax(sub, axis=1)).flatten()
            X_tr[:, i] = self.categories_[i][labels]

            if self.handle_unknown == 'ignore':
                # ignored unknown categories: we have a row of all zero's
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                if unknown.any():
                    found_unknown[i] = unknown

            j += n_categories

        # if ignored are found: potentially need to upcast result to
        # insert None values
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr
```
### 50 - sklearn/preprocessing/_encoders.py:

Start line: 745, End line: 780

```python
class OrdinalEncoder(_BaseEncoder):

    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.

        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        for i in range(n_features):
            labels = X[:, i].astype('int64')
            X_tr[:, i] = self.categories_[i][labels]

        return X_tr
```
### 94 - sklearn/preprocessing/_encoders.py:

Start line: 555, End line: 573

```python
class OneHotEncoder(_BaseEncoder):

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.
        """
        if self._legacy_mode:
            return _transform_selected(X, self._legacy_transform, self.dtype,
                                       self._categorical_features,
                                       copy=True)
        else:
            return self._transform_new(X)
```
### 100 - sklearn/preprocessing/_encoders.py:

Start line: 377, End line: 403

```python
class OneHotEncoder(_BaseEncoder):

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        if self.handle_unknown not in ('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', "
                   "got {0}.".format(self.handle_unknown))
            raise ValueError(msg)

        self._handle_deprecations(X)

        if self._legacy_mode:
            _transform_selected(X, self._legacy_fit_transform, self.dtype,
                                self._categorical_features,
                                copy=True)
            return self
        else:
            self._fit(X, handle_unknown=self.handle_unknown)
            return self
```
