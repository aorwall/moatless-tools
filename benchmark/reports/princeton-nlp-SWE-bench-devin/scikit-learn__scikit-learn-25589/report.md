# scikit-learn__scikit-learn-25589

| **scikit-learn/scikit-learn** | `53e0d95cb10cba5827751657e487f792afd94329` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6795 |
| **Any found context length** | 2722 |
| **Avg pos** | 187.0 |
| **Min pos** | 1 |
| **Max pos** | 23 |
| **Top file pos** | 1 |
| **Missing snippets** | 11 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/preprocessing/_encoders.py b/sklearn/preprocessing/_encoders.py
--- a/sklearn/preprocessing/_encoders.py
+++ b/sklearn/preprocessing/_encoders.py
@@ -270,6 +270,10 @@ class OneHotEncoder(_BaseEncoder):
         - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
           should be dropped.
 
+        When `max_categories` or `min_frequency` is configured to group
+        infrequent categories, the dropping behavior is handled after the
+        grouping.
+
         .. versionadded:: 0.21
            The parameter `drop` was added in 0.21.
 
@@ -544,7 +548,7 @@ def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
         """Convert `drop_idx` into the index for infrequent categories.
 
         If there are no infrequent categories, then `drop_idx` is
-        returned. This method is called in `_compute_drop_idx` when the `drop`
+        returned. This method is called in `_set_drop_idx` when the `drop`
         parameter is an array-like.
         """
         if not self._infrequent_enabled:
@@ -564,24 +568,35 @@ def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
             )
         return default_to_infrequent[drop_idx]
 
-    def _compute_drop_idx(self):
+    def _set_drop_idx(self):
         """Compute the drop indices associated with `self.categories_`.
 
         If `self.drop` is:
-        - `None`, returns `None`.
-        - `'first'`, returns all zeros to drop the first category.
-        - `'if_binary'`, returns zero if the category is binary and `None`
+        - `None`, No categories have been dropped.
+        - `'first'`, All zeros to drop the first category.
+        - `'if_binary'`, All zeros if the category is binary and `None`
           otherwise.
-        - array-like, returns the indices of the categories that match the
+        - array-like, The indices of the categories that match the
           categories in `self.drop`. If the dropped category is an infrequent
           category, then the index for the infrequent category is used. This
           means that the entire infrequent category is dropped.
+
+        This methods defines a public `drop_idx_` and a private
+        `_drop_idx_after_grouping`.
+
+        - `drop_idx_`: Public facing API that references the drop category in
+          `self.categories_`.
+        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
+          infrequent categories are grouped together.
+
+        If there are no infrequent categories or drop is `None`, then
+        `drop_idx_=_drop_idx_after_grouping`.
         """
         if self.drop is None:
-            return None
+            drop_idx_after_grouping = None
         elif isinstance(self.drop, str):
             if self.drop == "first":
-                return np.zeros(len(self.categories_), dtype=object)
+                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
             elif self.drop == "if_binary":
                 n_features_out_no_drop = [len(cat) for cat in self.categories_]
                 if self._infrequent_enabled:
@@ -590,7 +605,7 @@ def _compute_drop_idx(self):
                             continue
                         n_features_out_no_drop[i] -= infreq_idx.size - 1
 
-                return np.array(
+                drop_idx_after_grouping = np.array(
                     [
                         0 if n_features_out == 2 else None
                         for n_features_out in n_features_out_no_drop
@@ -647,7 +662,29 @@ def _compute_drop_idx(self):
                     )
                 )
                 raise ValueError(msg)
-            return np.array(drop_indices, dtype=object)
+            drop_idx_after_grouping = np.array(drop_indices, dtype=object)
+
+        # `_drop_idx_after_grouping` are the categories to drop *after* the infrequent
+        # categories are grouped together. If needed, we remap `drop_idx` back
+        # to the categories seen in `self.categories_`.
+        self._drop_idx_after_grouping = drop_idx_after_grouping
+
+        if not self._infrequent_enabled or drop_idx_after_grouping is None:
+            self.drop_idx_ = self._drop_idx_after_grouping
+        else:
+            drop_idx_ = []
+            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
+                default_to_infrequent = self._default_to_infrequent_mappings[
+                    feature_idx
+                ]
+                if drop_idx is None or default_to_infrequent is None:
+                    orig_drop_idx = drop_idx
+                else:
+                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]
+
+                drop_idx_.append(orig_drop_idx)
+
+            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)
 
     def _identify_infrequent(self, category_count, n_samples, col_idx):
         """Compute the infrequent indices.
@@ -809,16 +846,19 @@ def _compute_transformed_categories(self, i, remove_dropped=True):
 
     def _remove_dropped_categories(self, categories, i):
         """Remove dropped categories."""
-        if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
-            return np.delete(categories, self.drop_idx_[i])
+        if (
+            self._drop_idx_after_grouping is not None
+            and self._drop_idx_after_grouping[i] is not None
+        ):
+            return np.delete(categories, self._drop_idx_after_grouping[i])
         return categories
 
     def _compute_n_features_outs(self):
         """Compute the n_features_out for each input feature."""
         output = [len(cats) for cats in self.categories_]
 
-        if self.drop_idx_ is not None:
-            for i, drop_idx in enumerate(self.drop_idx_):
+        if self._drop_idx_after_grouping is not None:
+            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                 if drop_idx is not None:
                     output[i] -= 1
 
@@ -875,7 +915,7 @@ def fit(self, X, y=None):
             self._fit_infrequent_category_mapping(
                 fit_results["n_samples"], fit_results["category_counts"]
             )
-        self.drop_idx_ = self._compute_drop_idx()
+        self._set_drop_idx()
         self._n_features_outs = self._compute_n_features_outs()
         return self
 
@@ -914,8 +954,8 @@ def transform(self, X):
 
         n_samples, n_features = X_int.shape
 
-        if self.drop_idx_ is not None:
-            to_drop = self.drop_idx_.copy()
+        if self._drop_idx_after_grouping is not None:
+            to_drop = self._drop_idx_after_grouping.copy()
             # We remove all the dropped categories from mask, and decrement all
             # categories that occur after them to avoid an empty column.
             keep_cells = X_int != to_drop
@@ -1014,7 +1054,7 @@ def inverse_transform(self, X):
             # category. In this case we just fill the column with this
             # unique category value.
             if n_categories == 0:
-                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
+                X_tr[:, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
                 j += n_categories
                 continue
             sub = X[:, j : j + n_categories]
@@ -1031,14 +1071,19 @@ def inverse_transform(self, X):
                 if unknown.any():
                     # if categories were dropped then unknown categories will
                     # be mapped to the dropped category
-                    if self.drop_idx_ is None or self.drop_idx_[i] is None:
+                    if (
+                        self._drop_idx_after_grouping is None
+                        or self._drop_idx_after_grouping[i] is None
+                    ):
                         found_unknown[i] = unknown
                     else:
-                        X_tr[unknown, i] = self.categories_[i][self.drop_idx_[i]]
+                        X_tr[unknown, i] = self.categories_[i][
+                            self._drop_idx_after_grouping[i]
+                        ]
             else:
                 dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                 if dropped.any():
-                    if self.drop_idx_ is None:
+                    if self._drop_idx_after_grouping is None:
                         all_zero_samples = np.flatnonzero(dropped)
                         raise ValueError(
                             f"Samples {all_zero_samples} can not be inverted "
@@ -1047,7 +1092,7 @@ def inverse_transform(self, X):
                         )
                     # we can safely assume that all of the nulls in each column
                     # are the dropped value
-                    drop_idx = self.drop_idx_[i]
+                    drop_idx = self._drop_idx_after_grouping[i]
                     X_tr[dropped, i] = transformed_features[i][drop_idx]
 
             j += n_categories

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/preprocessing/_encoders.py | 273 | 273 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 547 | 547 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 567 | 584 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 593 | 593 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 650 | 650 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 812 | 821 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 878 | 878 | 23 | 1 | 13553
| sklearn/preprocessing/_encoders.py | 917 | 918 | 12 | 1 | 7354
| sklearn/preprocessing/_encoders.py | 1017 | 1017 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 1034 | 1041 | 11 | 1 | 6795
| sklearn/preprocessing/_encoders.py | 1050 | 1050 | 11 | 1 | 6795


## Problem Statement

```
OneHotEncoder `drop_idx_` attribute description in presence of infrequent categories
### Describe the issue linked to the documentation

### Issue summary

In the OneHotEncoder documentation both for [v1.2](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) and [v1.1](https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.OneHotEncoder.html?highlight=one+hot+encoder#sklearn.preprocessing.OneHotEncoder), the description of attribute `drop_idx_` in presence of infrequent categories reads as follows:

> If infrequent categories are enabled by setting `min_frequency` or `max_categories` to a non-default value and `drop_idx[i]` corresponds to a infrequent category, then the entire infrequent category is dropped.`

### User interpretation

My understanding of this description is that when `drop_idx_[i]` corresponds to an infrequent category for column `i`, then the expected encoded column `i_infrequent_sklearn` is dropped. For example, suppose we have the following situation:
\`\`\`
>>> X = np.array([['a'] * 2 + ['b'] * 4 + ['c'] * 4
...               + ['d'] * 4 + ['e'] * 4], dtype=object).T
>>> enc = preprocessing.OneHotEncoder(min_frequency=4, sparse_output=False, drop='first')
\`\`\`
Here `X` is a column with five categories where category `a` is considered infrequent. If the above interpretation is correct, then the expected output will consist of four columns, namely, `x0_b`, `x0_c`, `x0_d` and `x0_e`. This is because `a` is both the first category to get dropped due to `drop='first'` as well as an infrequent one. However, the transform output is as follows:
\`\`\`
>>> Xt = enc.fit_transform(X)
>>> pd.DataFrame(Xt, columns = enc.get_feature_names_out())
ent_categories_
    x0_c  x0_d  x0_e  x0_infrequent_sklearn
0    0.0   0.0   0.0                    1.0
1    0.0   0.0   0.0                    1.0
2    0.0   0.0   0.0                    0.0
3    0.0   0.0   0.0                    0.0
4    0.0   0.0   0.0                    0.0
5    0.0   0.0   0.0                    0.0
6    1.0   0.0   0.0                    0.0
7    1.0   0.0   0.0                    0.0
8    1.0   0.0   0.0                    0.0
9    1.0   0.0   0.0                    0.0
10   0.0   1.0   0.0                    0.0
11   0.0   1.0   0.0                    0.0
12   0.0   1.0   0.0                    0.0
13   0.0   1.0   0.0                    0.0
14   0.0   0.0   1.0                    0.0
15   0.0   0.0   1.0                    0.0
16   0.0   0.0   1.0                    0.0
17   0.0   0.0   1.0                    0.0
\`\`\`
This means that category `a` is part of the `x0_infrequent_sklearn` column, which takes the value of `1` when `X=='a'`. Category `b` is dropped, this is expected since the `drop='first'` functionality drops the column indexed `0` and after the `_encode` function is applied, categories are remapped based on their sorting order and infrequent ones are mapped last. Meaning that `'a'->4, 'b'->0, 'c'->1, 'd'->2, 'e'->3. This can be verified by the following objects:
\`\`\`
>>> enc.categories_
[array(['a', 'b', 'c', 'd', 'e'], dtype=object)]
>>> enc._default_to_infrequent_mappings
[array([4, 0, 1, 2, 3])]
\`\`\`
Notice how at transform the values of the encoded columns are `0` when `X=='b'`. Finally, columns `x0_c`, `x0_d` and `x0_e` are encoded as expected.

### Suggest a potential alternative/fix

### Correct suggestive description based on what is actually happening.

> If infrequent categories are enabled by setting `min_frequency` or `max_categories` to a non-default value and `drop_idx_[i]` corresponds to a infrequent category, then the "first", i.e., indexed `0`, frequent category is dropped after `_encode` is applied during `_transform`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/preprocessing/_encoders.py** | 216 | 482| 2722 | 2722 | 12137 | 
| **-> 2 <-** | **1 sklearn/preprocessing/_encoders.py** | 543 | 565| 229 | 2951 | 12137 | 
| **-> 3 <-** | **1 sklearn/preprocessing/_encoders.py** | 567 | 650| 667 | 3618 | 12137 | 
| **-> 4 <-** | **1 sklearn/preprocessing/_encoders.py** | 1007 | 1064| 539 | 4157 | 12137 | 
| 5 | **1 sklearn/preprocessing/_encoders.py** | 787 | 808| 199 | 4356 | 12137 | 
| 6 | **1 sklearn/preprocessing/_encoders.py** | 484 | 541| 484 | 4840 | 12137 | 
| **-> 7 <-** | **1 sklearn/preprocessing/_encoders.py** | 810 | 835| 220 | 5060 | 12137 | 
| 8 | **1 sklearn/preprocessing/_encoders.py** | 691 | 747| 546 | 5606 | 12137 | 
| 9 | **1 sklearn/preprocessing/_encoders.py** | 652 | 689| 310 | 5916 | 12137 | 
| 10 | **1 sklearn/preprocessing/_encoders.py** | 749 | 785| 329 | 6245 | 12137 | 
| **-> 11 <-** | **1 sklearn/preprocessing/_encoders.py** | 151 | 1111| 550 | 6795 | 12137 | 
| **-> 12 <-** | **1 sklearn/preprocessing/_encoders.py** | 882 | 950| 559 | 7354 | 12137 | 
| 13 | **1 sklearn/preprocessing/_encoders.py** | 952 | 1005| 407 | 7761 | 12137 | 
| 14 | **1 sklearn/preprocessing/_encoders.py** | 69 | 149| 606 | 8367 | 12137 | 
| 15 | 2 examples/release_highlights/plot_release_highlights_1_1_0.py | 98 | 177| 847 | 9214 | 14274 | 
| 16 | 3 examples/ensemble/plot_gradient_boosting_categorical.py | 1 | 104| 744 | 9958 | 16561 | 
| 17 | **3 sklearn/preprocessing/_encoders.py** | 1235 | 1256| 175 | 10133 | 16561 | 
| 18 | **3 sklearn/preprocessing/_encoders.py** | 1258 | 1360| 780 | 10913 | 16561 | 
| 19 | **3 sklearn/preprocessing/_encoders.py** | 1114 | 1233| 1081 | 11994 | 16561 | 
| 20 | 3 examples/ensemble/plot_gradient_boosting_categorical.py | 218 | 278| 687 | 12681 | 16561 | 
| 21 | **3 sklearn/preprocessing/_encoders.py** | 1390 | 1450| 452 | 13133 | 16561 | 
| 22 | **3 sklearn/preprocessing/_encoders.py** | 1101 | 1111| 128 | 13261 | 16561 | 
| **-> 23 <-** | **3 sklearn/preprocessing/_encoders.py** | 837 | 880| 292 | 13553 | 16561 | 
| 24 | **3 sklearn/preprocessing/_encoders.py** | 1362 | 1388| 208 | 13761 | 16561 | 
| 25 | **3 sklearn/preprocessing/_encoders.py** | 25 | 67| 391 | 14152 | 16561 | 
| 26 | 4 examples/applications/plot_cyclical_feature_engineering.py | 450 | 523| 747 | 14899 | 23703 | 
| 27 | 4 examples/ensemble/plot_gradient_boosting_categorical.py | 106 | 178| 592 | 15491 | 23703 | 
| 28 | 5 sklearn/preprocessing/_discretization.py | 258 | 327| 702 | 16193 | 27612 | 
| 29 | 6 examples/inspection/plot_partial_dependence.py | 88 | 174| 808 | 17001 | 32561 | 
| 30 | **6 sklearn/preprocessing/_encoders.py** | 1066 | 1099| 293 | 17294 | 32561 | 
| 31 | 6 examples/applications/plot_cyclical_feature_engineering.py | 244 | 357| 1067 | 18361 | 32561 | 
| 32 | 7 sklearn/ensemble/_hist_gradient_boosting/binning.py | 247 | 280| 305 | 18666 | 35506 | 
| 33 | **7 sklearn/preprocessing/_encoders.py** | 5 | 22| 133 | 18799 | 35506 | 
| 34 | 7 examples/applications/plot_cyclical_feature_engineering.py | 524 | 621| 899 | 19698 | 35506 | 
| 35 | 7 sklearn/preprocessing/_discretization.py | 24 | 155| 1332 | 21030 | 35506 | 
| 36 | 8 sklearn/naive_bayes.py | 1463 | 1483| 211 | 21241 | 48562 | 
| 37 | 9 sklearn/decomposition/_dict_learning.py | 339 | 371| 492 | 21733 | 67522 | 
| 38 | 9 sklearn/decomposition/_dict_learning.py | 117 | 199| 711 | 22444 | 67522 | 
| 39 | 10 examples/inspection/plot_permutation_importance.py | 116 | 197| 760 | 23204 | 69716 | 
| 40 | 11 sklearn/impute/_iterative.py | 55 | 281| 2395 | 25599 | 77545 | 
| 41 | 11 sklearn/preprocessing/_discretization.py | 354 | 391| 298 | 25897 | 77545 | 
| 42 | 12 sklearn/impute/_base.py | 876 | 908| 279 | 26176 | 86106 | 
| 43 | 13 examples/release_highlights/plot_release_highlights_1_0_0.py | 92 | 165| 775 | 26951 | 88441 | 
| 44 | 13 sklearn/ensemble/_hist_gradient_boosting/binning.py | 70 | 150| 993 | 27944 | 88441 | 
| 45 | 14 sklearn/decomposition/_fastica.py | 602 | 695| 814 | 28758 | 94955 | 
| 46 | 15 sklearn/preprocessing/_data.py | 2645 | 2708| 677 | 29435 | 122514 | 
| 47 | 15 sklearn/impute/_base.py | 719 | 800| 742 | 30177 | 122514 | 
| 48 | 15 sklearn/impute/_iterative.py | 425 | 461| 455 | 30632 | 122514 | 
| 49 | 16 sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py | 182 | 266| 713 | 31345 | 139717 | 
| 50 | 16 sklearn/impute/_base.py | 973 | 1012| 332 | 31677 | 139717 | 
| 51 | 16 sklearn/impute/_base.py | 535 | 621| 692 | 32369 | 139717 | 
| 52 | 16 examples/inspection/plot_partial_dependence.py | 1 | 87| 795 | 33164 | 139717 | 
| 53 | 17 sklearn/inspection/_partial_dependence.py | 92 | 138| 443 | 33607 | 144970 | 
| 54 | 18 examples/release_highlights/plot_release_highlights_0_22_0.py | 90 | 193| 896 | 34503 | 147380 | 
| 55 | 19 sklearn/feature_extraction/_dict_vectorizer.py | 190 | 287| 729 | 35232 | 150583 | 
| 56 | 20 examples/compose/plot_column_transformer_mixed_types.py | 1 | 122| 879 | 36111 | 152380 | 
| 57 | 20 examples/inspection/plot_partial_dependence.py | 269 | 355| 849 | 36960 | 152380 | 
| 58 | 20 examples/release_highlights/plot_release_highlights_1_1_0.py | 1 | 96| 775 | 37735 | 152380 | 
| 59 | 20 sklearn/decomposition/_dict_learning.py | 374 | 451| 494 | 38229 | 152380 | 
| 60 | 21 sklearn/model_selection/_split.py | 1157 | 1204| 481 | 38710 | 175483 | 
| 61 | 22 sklearn/compose/_column_transformer.py | 624 | 649| 247 | 38957 | 184789 | 
| 62 | 23 examples/preprocessing/plot_all_scaling.py | 250 | 351| 964 | 39921 | 188240 | 
| 63 | 23 sklearn/impute/_base.py | 142 | 270| 1272 | 41193 | 188240 | 
| 64 | 23 sklearn/decomposition/_dict_learning.py | 1288 | 1306| 137 | 41330 | 188240 | 
| 65 | 24 examples/release_highlights/plot_release_highlights_1_2_0.py | 92 | 167| 718 | 42048 | 189725 | 
| 66 | 24 sklearn/preprocessing/_data.py | 2765 | 2784| 179 | 42227 | 189725 | 
| 67 | 24 sklearn/naive_bayes.py | 1485 | 1513| 302 | 42529 | 189725 | 
| 68 | 25 examples/text/plot_hashing_vs_dict_vectorizer.py | 121 | 210| 775 | 43304 | 193223 | 
| 69 | 25 sklearn/impute/_iterative.py | 806 | 858| 429 | 43733 | 193223 | 
| 70 | 25 sklearn/decomposition/_dict_learning.py | 202 | 338| 1186 | 44919 | 193223 | 
| 71 | 25 examples/inspection/plot_permutation_importance.py | 1 | 115| 984 | 45903 | 193223 | 
| 72 | 26 sklearn/utils/_encode.py | 167 | 190| 198 | 46101 | 195747 | 
| 73 | 26 sklearn/preprocessing/_data.py | 3105 | 3134| 303 | 46404 | 195747 | 
| 74 | 26 sklearn/impute/_iterative.py | 755 | 804| 401 | 46805 | 195747 | 


### Hint

```
Thank you for opening the issue! In this case, API-wise I think `drop_idx` is defined incorrectly and should be `1` point to `b`, because it is the categorical that is actually dropped. 

There seems to be a bigger issue with how `drop_idx` is defined when there are any infrequent categories. I am looking into a fix.
```

## Patch

```diff
diff --git a/sklearn/preprocessing/_encoders.py b/sklearn/preprocessing/_encoders.py
--- a/sklearn/preprocessing/_encoders.py
+++ b/sklearn/preprocessing/_encoders.py
@@ -270,6 +270,10 @@ class OneHotEncoder(_BaseEncoder):
         - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
           should be dropped.
 
+        When `max_categories` or `min_frequency` is configured to group
+        infrequent categories, the dropping behavior is handled after the
+        grouping.
+
         .. versionadded:: 0.21
            The parameter `drop` was added in 0.21.
 
@@ -544,7 +548,7 @@ def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
         """Convert `drop_idx` into the index for infrequent categories.
 
         If there are no infrequent categories, then `drop_idx` is
-        returned. This method is called in `_compute_drop_idx` when the `drop`
+        returned. This method is called in `_set_drop_idx` when the `drop`
         parameter is an array-like.
         """
         if not self._infrequent_enabled:
@@ -564,24 +568,35 @@ def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
             )
         return default_to_infrequent[drop_idx]
 
-    def _compute_drop_idx(self):
+    def _set_drop_idx(self):
         """Compute the drop indices associated with `self.categories_`.
 
         If `self.drop` is:
-        - `None`, returns `None`.
-        - `'first'`, returns all zeros to drop the first category.
-        - `'if_binary'`, returns zero if the category is binary and `None`
+        - `None`, No categories have been dropped.
+        - `'first'`, All zeros to drop the first category.
+        - `'if_binary'`, All zeros if the category is binary and `None`
           otherwise.
-        - array-like, returns the indices of the categories that match the
+        - array-like, The indices of the categories that match the
           categories in `self.drop`. If the dropped category is an infrequent
           category, then the index for the infrequent category is used. This
           means that the entire infrequent category is dropped.
+
+        This methods defines a public `drop_idx_` and a private
+        `_drop_idx_after_grouping`.
+
+        - `drop_idx_`: Public facing API that references the drop category in
+          `self.categories_`.
+        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
+          infrequent categories are grouped together.
+
+        If there are no infrequent categories or drop is `None`, then
+        `drop_idx_=_drop_idx_after_grouping`.
         """
         if self.drop is None:
-            return None
+            drop_idx_after_grouping = None
         elif isinstance(self.drop, str):
             if self.drop == "first":
-                return np.zeros(len(self.categories_), dtype=object)
+                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
             elif self.drop == "if_binary":
                 n_features_out_no_drop = [len(cat) for cat in self.categories_]
                 if self._infrequent_enabled:
@@ -590,7 +605,7 @@ def _compute_drop_idx(self):
                             continue
                         n_features_out_no_drop[i] -= infreq_idx.size - 1
 
-                return np.array(
+                drop_idx_after_grouping = np.array(
                     [
                         0 if n_features_out == 2 else None
                         for n_features_out in n_features_out_no_drop
@@ -647,7 +662,29 @@ def _compute_drop_idx(self):
                     )
                 )
                 raise ValueError(msg)
-            return np.array(drop_indices, dtype=object)
+            drop_idx_after_grouping = np.array(drop_indices, dtype=object)
+
+        # `_drop_idx_after_grouping` are the categories to drop *after* the infrequent
+        # categories are grouped together. If needed, we remap `drop_idx` back
+        # to the categories seen in `self.categories_`.
+        self._drop_idx_after_grouping = drop_idx_after_grouping
+
+        if not self._infrequent_enabled or drop_idx_after_grouping is None:
+            self.drop_idx_ = self._drop_idx_after_grouping
+        else:
+            drop_idx_ = []
+            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
+                default_to_infrequent = self._default_to_infrequent_mappings[
+                    feature_idx
+                ]
+                if drop_idx is None or default_to_infrequent is None:
+                    orig_drop_idx = drop_idx
+                else:
+                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]
+
+                drop_idx_.append(orig_drop_idx)
+
+            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)
 
     def _identify_infrequent(self, category_count, n_samples, col_idx):
         """Compute the infrequent indices.
@@ -809,16 +846,19 @@ def _compute_transformed_categories(self, i, remove_dropped=True):
 
     def _remove_dropped_categories(self, categories, i):
         """Remove dropped categories."""
-        if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
-            return np.delete(categories, self.drop_idx_[i])
+        if (
+            self._drop_idx_after_grouping is not None
+            and self._drop_idx_after_grouping[i] is not None
+        ):
+            return np.delete(categories, self._drop_idx_after_grouping[i])
         return categories
 
     def _compute_n_features_outs(self):
         """Compute the n_features_out for each input feature."""
         output = [len(cats) for cats in self.categories_]
 
-        if self.drop_idx_ is not None:
-            for i, drop_idx in enumerate(self.drop_idx_):
+        if self._drop_idx_after_grouping is not None:
+            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                 if drop_idx is not None:
                     output[i] -= 1
 
@@ -875,7 +915,7 @@ def fit(self, X, y=None):
             self._fit_infrequent_category_mapping(
                 fit_results["n_samples"], fit_results["category_counts"]
             )
-        self.drop_idx_ = self._compute_drop_idx()
+        self._set_drop_idx()
         self._n_features_outs = self._compute_n_features_outs()
         return self
 
@@ -914,8 +954,8 @@ def transform(self, X):
 
         n_samples, n_features = X_int.shape
 
-        if self.drop_idx_ is not None:
-            to_drop = self.drop_idx_.copy()
+        if self._drop_idx_after_grouping is not None:
+            to_drop = self._drop_idx_after_grouping.copy()
             # We remove all the dropped categories from mask, and decrement all
             # categories that occur after them to avoid an empty column.
             keep_cells = X_int != to_drop
@@ -1014,7 +1054,7 @@ def inverse_transform(self, X):
             # category. In this case we just fill the column with this
             # unique category value.
             if n_categories == 0:
-                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
+                X_tr[:, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
                 j += n_categories
                 continue
             sub = X[:, j : j + n_categories]
@@ -1031,14 +1071,19 @@ def inverse_transform(self, X):
                 if unknown.any():
                     # if categories were dropped then unknown categories will
                     # be mapped to the dropped category
-                    if self.drop_idx_ is None or self.drop_idx_[i] is None:
+                    if (
+                        self._drop_idx_after_grouping is None
+                        or self._drop_idx_after_grouping[i] is None
+                    ):
                         found_unknown[i] = unknown
                     else:
-                        X_tr[unknown, i] = self.categories_[i][self.drop_idx_[i]]
+                        X_tr[unknown, i] = self.categories_[i][
+                            self._drop_idx_after_grouping[i]
+                        ]
             else:
                 dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                 if dropped.any():
-                    if self.drop_idx_ is None:
+                    if self._drop_idx_after_grouping is None:
                         all_zero_samples = np.flatnonzero(dropped)
                         raise ValueError(
                             f"Samples {all_zero_samples} can not be inverted "
@@ -1047,7 +1092,7 @@ def inverse_transform(self, X):
                         )
                     # we can safely assume that all of the nulls in each column
                     # are the dropped value
-                    drop_idx = self.drop_idx_[i]
+                    drop_idx = self._drop_idx_after_grouping[i]
                     X_tr[dropped, i] = transformed_features[i][drop_idx]
 
             j += n_categories

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_encoders.py b/sklearn/preprocessing/tests/test_encoders.py
--- a/sklearn/preprocessing/tests/test_encoders.py
+++ b/sklearn/preprocessing/tests/test_encoders.py
@@ -929,7 +929,7 @@ def test_ohe_infrequent_two_levels_drop_frequent(drop):
         max_categories=2,
         drop=drop,
     ).fit(X_train)
-    assert_array_equal(ohe.drop_idx_, [0])
+    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"
 
     X_test = np.array([["b"], ["c"]])
     X_trans = ohe.transform(X_test)
@@ -2015,3 +2015,39 @@ def test_ordinal_encoder_missing_unknown_encoding_max():
     X_test = np.array([["snake"]])
     X_trans = enc.transform(X_test)
     assert_allclose(X_trans, [[2]])
+
+
+def test_drop_idx_infrequent_categories():
+    """Check drop_idx is defined correctly with infrequent categories.
+
+    Non-regression test for gh-25550.
+    """
+    X = np.array(
+        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
+    ).T
+    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="first").fit(X)
+    assert_array_equal(
+        ohe.get_feature_names_out(), ["x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"]
+    )
+    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"
+
+    X = np.array([["a"] * 2 + ["b"] * 2 + ["c"] * 10], dtype=object).T
+    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="if_binary").fit(X)
+    assert_array_equal(ohe.get_feature_names_out(), ["x0_infrequent_sklearn"])
+    assert ohe.categories_[0][ohe.drop_idx_[0]] == "c"
+
+    X = np.array(
+        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
+    ).T
+    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=["d"]).fit(X)
+    assert_array_equal(
+        ohe.get_feature_names_out(), ["x0_b", "x0_c", "x0_e", "x0_infrequent_sklearn"]
+    )
+    assert ohe.categories_[0][ohe.drop_idx_[0]] == "d"
+
+    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=None).fit(X)
+    assert_array_equal(
+        ohe.get_feature_names_out(),
+        ["x0_b", "x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"],
+    )
+    assert ohe.drop_idx_ is None

```


## Code snippets

### 1 - sklearn/preprocessing/_encoders.py:

Start line: 216, End line: 482

```python
class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter)

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

        .. versionadded:: 0.20

    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), \
            default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

        .. versionadded:: 0.21
           The parameter `drop` was added in 0.21.

        .. versionchanged:: 0.23
           The option `drop='if_binary'` was added in 0.23.

        .. versionchanged:: 1.1
            Support for dropping infrequent categories.

    sparse : bool, default=True
        Will return sparse matrix if set True else will return an array.

        .. deprecated:: 1.2
           `sparse` is deprecated in 1.2 and will be removed in 1.4. Use
           `sparse_output` instead.

    sparse_output : bool, default=True
        Will return sparse matrix if set True else will return an array.

        .. versionadded:: 1.2
           `sparse` was renamed to `sparse_output`

    dtype : number type, default=float
        Desired dtype of output.

    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'}, \
                     default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is present during transform.
        - 'ignore' : When an unknown category is encountered during
          transform, the resulting one-hot encoded columns for this feature
          will be all zeros. In the inverse transform, an unknown category
          will be denoted as None.
        - 'infrequent_if_exist' : When an unknown category is encountered
          during transform, the resulting one-hot encoded columns for this
          feature will map to the infrequent category if it exists. The
          infrequent category will be mapped to the last position in the
          encoding. During inverse transform, an unknown category will be
          mapped to the category denoted `'infrequent'` if it exists. If the
          `'infrequent'` category does not exist, then :meth:`transform` and
          :meth:`inverse_transform` will handle an unknown category as with
          `handle_unknown='ignore'`. Infrequent categories exist based on
          `min_frequency` and `max_categories`. Read more in the
          :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

        .. versionchanged:: 1.1
            `'infrequent_if_exist'` was added to automatically handle unknown
            categories and infrequent categories.

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

    max_categories : int, default=None
        Specifies an upper limit to the number of output features for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

    feature_name_combiner : "concat" or callable, default="concat"
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        `"concat"` concatenates encoded feature name and category with
        `feature + "_" + str(category)`.E.g. feature X with values 1, 6, 7 create
        feature names `X_1, X_6, X_7`.

        .. versionadded:: 1.3

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

        If infrequent categories are enabled by setting `min_frequency` or
        `max_categories` to a non-default value and `drop_idx[i]` corresponds
        to a infrequent category, then the entire infrequent category is
        dropped.

        .. versionchanged:: 0.23
           Added the possibility to contain `None` values.

    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    feature_name_combiner : callable or None
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        .. versionadded:: 1.3

    See Also
    --------
    OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
      encoding of dictionary items or strings.
    LabelBinarizer : Binarizes labels in a one-vs-all
      fashion.
    MultiLabelBinarizer : Transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder

    One can discard categories not seen during `fit`:

    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)

    One can always drop the first column for each feature:

    >>> drop_enc = OneHotEncoder(drop='first').fit(X)
    >>> drop_enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 0., 0.],
           [1., 1., 0.]])

    Or drop a column for feature only having 2 categories:

    >>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    >>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 1., 0.]])

    One can change the way feature names are created.

    >>> def custom_combiner(feature, category):
    ...     return str(feature) + "_" + type(category).__name__ + "_" + str(category)
    >>> custom_fnames_enc = OneHotEncoder(feature_name_combiner=custom_combiner).fit(X)
    >>> custom_fnames_enc.get_feature_names_out()
    array(['x0_str_Female', 'x0_str_Male', 'x1_int_1', 'x1_int_2', 'x1_int_3'],
          dtype=object)

    Infrequent categories are enabled by setting `max_categories` or `min_frequency`.

    >>> import numpy as np
    >>> X = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
    >>> ohe = OneHotEncoder(max_categories=3, sparse_output=False).fit(X)
    >>> ohe.infrequent_categories_
    [array(['a', 'd'], dtype=object)]
    >>> ohe.transform([["a"], ["b"]])
    array([[0., 0., 1.],
           [1., 0., 0.]])
    """
```
### 2 - sklearn/preprocessing/_encoders.py:

Start line: 543, End line: 565

```python
class OneHotEncoder(_BaseEncoder):

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        """Convert `drop_idx` into the index for infrequent categories.

        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_compute_drop_idx` when the `drop`
        parameter is an array-like.
        """
        if not self._infrequent_enabled:
            return drop_idx

        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx

        # Raise error when explicitly dropping a category that is infrequent
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(
                f"Unable to drop category {categories[drop_idx]!r} from feature"
                f" {feature_idx} because it is infrequent"
            )
        return default_to_infrequent[drop_idx]
```
### 3 - sklearn/preprocessing/_encoders.py:

Start line: 567, End line: 650

```python
class OneHotEncoder(_BaseEncoder):

    def _compute_drop_idx(self):
        """Compute the drop indices associated with `self.categories_`.

        If `self.drop` is:
        - `None`, returns `None`.
        - `'first'`, returns all zeros to drop the first category.
        - `'if_binary'`, returns zero if the category is binary and `None`
          otherwise.
        - array-like, returns the indices of the categories that match the
          categories in `self.drop`. If the dropped category is an infrequent
          category, then the index for the infrequent category is used. This
          means that the entire infrequent category is dropped.
        """
        if self.drop is None:
            return None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                return np.zeros(len(self.categories_), dtype=object)
            elif self.drop == "if_binary":
                n_features_out_no_drop = [len(cat) for cat in self.categories_]
                if self._infrequent_enabled:
                    for i, infreq_idx in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1

                return np.array(
                    [
                        0 if n_features_out == 2 else None
                        for n_features_out in n_features_out_no_drop
                    ],
                    dtype=object,
                )

        else:
            drop_array = np.asarray(self.drop, dtype=object)
            droplen = len(drop_array)

            if droplen != len(self.categories_):
                msg = (
                    "`drop` should have length equal to the number "
                    "of features ({}), got {}"
                )
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for feature_idx, (drop_val, cat_list) in enumerate(
                zip(drop_array, self.categories_)
            ):
                if not is_scalar_nan(drop_val):
                    drop_idx = np.where(cat_list == drop_val)[0]
                    if drop_idx.size:  # found drop idx
                        drop_indices.append(
                            self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0])
                        )
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue

                # drop_val is nan, find nan in categories manually
                for cat_idx, cat in enumerate(cat_list):
                    if is_scalar_nan(cat):
                        drop_indices.append(
                            self._map_drop_idx_to_infrequent(feature_idx, cat_idx)
                        )
                        break
                else:  # loop did not break thus drop is missing
                    missing_drops.append((feature_idx, drop_val))

            if any(missing_drops):
                msg = (
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format(
                        "\n".join(
                            [
                                "Category: {}, Feature: {}".format(c, v)
                                for c, v in missing_drops
                            ]
                        )
                    )
                )
                raise ValueError(msg)
            return np.array(drop_indices, dtype=object)
```
### 4 - sklearn/preprocessing/_encoders.py:

Start line: 1007, End line: 1064

```python
class OneHotEncoder(_BaseEncoder):

    def inverse_transform(self, X):
        # ... other code

        for i in range(n_features):
            cats_wo_dropped = self._remove_dropped_categories(
                transformed_features[i], i
            )
            n_categories = cats_wo_dropped.shape[0]

            # Only happens if there was a column with a unique
            # category. In this case we just fill the column with this
            # unique category value.
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
                j += n_categories
                continue
            sub = X[:, j : j + n_categories]
            # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats_wo_dropped[labels]

            if self.handle_unknown == "ignore" or (
                self.handle_unknown == "infrequent_if_exist"
                and infrequent_indices[i] is None
            ):
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                # ignored unknown categories: we have a row of all zero
                if unknown.any():
                    # if categories were dropped then unknown categories will
                    # be mapped to the dropped category
                    if self.drop_idx_ is None or self.drop_idx_[i] is None:
                        found_unknown[i] = unknown
                    else:
                        X_tr[unknown, i] = self.categories_[i][self.drop_idx_[i]]
            else:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    if self.drop_idx_ is None:
                        all_zero_samples = np.flatnonzero(dropped)
                        raise ValueError(
                            f"Samples {all_zero_samples} can not be inverted "
                            "when drop=None and handle_unknown='error' "
                            "because they contain all zeros"
                        )
                    # we can safely assume that all of the nulls in each column
                    # are the dropped value
                    drop_idx = self.drop_idx_[i]
                    X_tr[dropped, i] = transformed_features[i][drop_idx]

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
### 5 - sklearn/preprocessing/_encoders.py:

Start line: 787, End line: 808

```python
class OneHotEncoder(_BaseEncoder):

    def _compute_transformed_categories(self, i, remove_dropped=True):
        """Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named
        'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        """
        cats = self.categories_[i]

        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = "infrequent_sklearn"
                # infrequent category is always at the end
                cats = np.concatenate(
                    (cats[frequent_mask], np.array([infrequent_cat], dtype=object))
                )

        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats
```
### 6 - sklearn/preprocessing/_encoders.py:

Start line: 484, End line: 541

```python
class OneHotEncoder(_BaseEncoder):

    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        "drop": [StrOptions({"first", "if_binary"}), "array-like", None],
        "dtype": "no_validation",  # validation delegated to numpy
        "handle_unknown": [StrOptions({"error", "ignore", "infrequent_if_exist"})],
        "max_categories": [Interval(Integral, 1, None, closed="left"), None],
        "min_frequency": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="neither"),
            None,
        ],
        "sparse": [Hidden(StrOptions({"deprecated"})), "boolean"],  # deprecated
        "sparse_output": ["boolean"],
        "feature_name_combiner": [StrOptions({"concat"}), callable],
    }

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse="deprecated",
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
        feature_name_combiner="concat",
    ):
        self.categories = categories
        # TODO(1.4): Remove self.sparse
        self.sparse = sparse
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    @property
    def infrequent_categories_(self):
        """Infrequent categories for each feature."""
        # raises an AttributeError if `_infrequent_indices` is not defined
        infrequent_indices = self._infrequent_indices
        return [
            None if indices is None else category[indices]
            for category, indices in zip(self.categories_, infrequent_indices)
        ]

    def _check_infrequent_enabled(self):
        """
        This functions checks whether _infrequent_enabled is True or False.
        This has to be called after parameter validation in the fit function.
        """
        self._infrequent_enabled = (
            self.max_categories is not None and self.max_categories >= 1
        ) or self.min_frequency is not None
```
### 7 - sklearn/preprocessing/_encoders.py:

Start line: 810, End line: 835

```python
class OneHotEncoder(_BaseEncoder):

    def _remove_dropped_categories(self, categories, i):
        """Remove dropped categories."""
        if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
            return np.delete(categories, self.drop_idx_[i])
        return categories

    def _compute_n_features_outs(self):
        """Compute the n_features_out for each input feature."""
        output = [len(cats) for cats in self.categories_]

        if self.drop_idx_ is not None:
            for i, drop_idx in enumerate(self.drop_idx_):
                if drop_idx is not None:
                    output[i] -= 1

        if not self._infrequent_enabled:
            return output

        # infrequent is enabled, the number of features out are reduced
        # because the infrequent categories are grouped together
        for i, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1

        return output
```
### 8 - sklearn/preprocessing/_encoders.py:

Start line: 691, End line: 747

```python
class OneHotEncoder(_BaseEncoder):

    def _fit_infrequent_category_mapping(self, n_samples, category_counts):
        """Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is an array of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        .. versionadded:: 1.1

        Parameters
        ----------
        n_samples : int
            Number of samples in training set.
        category_counts: list of ndarray
            `category_counts[i]` is the category counts corresponding to
            `self.categories_[i]`.
        """
        self._infrequent_indices = [
            self._identify_infrequent(category_count, n_samples, col_idx)
            for col_idx, category_count in enumerate(category_counts)
        ]

        # compute mapping from default mapping to infrequent mapping
        self._default_to_infrequent_mappings = []

        for cats, infreq_idx in zip(self.categories_, self._infrequent_indices):
            # no infrequent categories
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue

            n_cats = len(cats)
            # infrequent indices exist
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size

            # infrequent categories are mapped to the last element.
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats

            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)

            self._default_to_infrequent_mappings.append(mapping)
```
### 9 - sklearn/preprocessing/_encoders.py:

Start line: 652, End line: 689

```python
class OneHotEncoder(_BaseEncoder):

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        """Compute the infrequent indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Category counts.

        n_samples : int
            Number of samples.

        col_idx : int
            Index of the current category. Only used for the error message.

        Returns
        -------
        output : ndarray of shape (n_infrequent_categories,) or None
            If there are infrequent categories, indices of infrequent
            categories. Otherwise None.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)

        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            # stable sort to preserve original count order
            smallest_levels = np.argsort(category_count, kind="mergesort")[
                : -self.max_categories + 1
            ]
            infrequent_mask[smallest_levels] = True

        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None
```
### 10 - sklearn/preprocessing/_encoders.py:

Start line: 749, End line: 785

```python
class OneHotEncoder(_BaseEncoder):

    def _map_infrequent_categories(self, X_int, X_mask):
        """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.

        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.
        """
        if not self._infrequent_enabled:
            return

        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue

            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == "infrequent_if_exist":
                # All the unknown values are now mapped to the
                # infrequent_idx[0], which makes the unknown values valid
                # This is needed in `transform` when the encoding is formed
                # using `X_mask`.
                X_mask[:, col_idx] = True

        # Remaps encoding in `X_int` where the infrequent categories are
        # grouped together.
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue
            X_int[:, i] = np.take(mapping, X_int[:, i])
```
### 11 - sklearn/preprocessing/_encoders.py:

Start line: 151, End line: 1111

```python
class _BaseEncoder(TransformerMixin, BaseEstimator):

    def _transform(
        self, X, handle_unknown="error", force_all_finite=True, warn_on_unknown=False
    ):
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )

        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == "O" and Xi.dtype.kind == "U":
                        # categories are objects and Xi are numpy strings.
                        # Cast Xi to an object dtype to prevent truncation
                        # when setting invalid values.
                        Xi = Xi.astype("O")
                    else:
                        Xi = Xi.copy()

                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _check_unknown was
            # already called above.
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(
                "Found unknown categories in columns "
                f"{columns_with_unknown} during transform. These "
                "unknown categories will be encoded as all zeros",
                UserWarning,
            )

        return X_int, X_mask

    def _more_tags(self):
        return {"X_types": ["categorical"]}


class OneHotEncoder(_BaseEncoder):
```
### 12 - sklearn/preprocessing/_encoders.py:

Start line: 882, End line: 950

```python
class OneHotEncoder(_BaseEncoder):

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        If there are infrequent categories for a feature, the infrequent
        categories will be grouped into a single category.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse_output=True`, a sparse matrix will be
            returned.
        """
        check_is_fitted(self)
        # validation of X happens in _check_X called by _transform
        warn_on_unknown = self.drop is not None and self.handle_unknown in {
            "ignore",
            "infrequent_if_exist",
        }
        X_int, X_mask = self._transform(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            warn_on_unknown=warn_on_unknown,
        )
        self._map_infrequent_categories(X_int, X_mask)

        n_samples, n_features = X_int.shape

        if self.drop_idx_ is not None:
            to_drop = self.drop_idx_.copy()
            # We remove all the dropped categories from mask, and decrement all
            # categories that occur after them to avoid an empty column.
            keep_cells = X_int != to_drop
            for i, cats in enumerate(self.categories_):
                # drop='if_binary' but feature isn't binary
                if to_drop[i] is None:
                    # set to cardinality to not drop from X_int
                    to_drop[i] = len(cats)

            to_drop = to_drop.reshape(1, -1)
            X_int[X_int > to_drop] -= 1
            X_mask &= keep_cells

        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + self._n_features_outs)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]

        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])

        out = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, feature_indices[-1]),
            dtype=self.dtype,
        )
        if not self.sparse_output:
            return out.toarray()
        else:
            return out
```
### 13 - sklearn/preprocessing/_encoders.py:

Start line: 952, End line: 1005

```python
class OneHotEncoder(_BaseEncoder):

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        When unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category. If the
        feature with the unknown category has a dropped category, the dropped
        category will be its inverse.

        For a given input feature, if there is an infrequent category,
        'infrequent_sklearn' will be used to represent the infrequent category.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        n_features_out = np.sum(self._n_features_outs)

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_features_out:
            raise ValueError(msg.format(n_features_out, X.shape[1]))

        transformed_features = [
            self._compute_transformed_categories(i, remove_dropped=False)
            for i, _ in enumerate(self.categories_)
        ]

        # create resulting array of appropriate dtype
        dt = np.result_type(*[cat.dtype for cat in transformed_features])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        j = 0
        found_unknown = {}

        if self._infrequent_enabled:
            infrequent_indices = self._infrequent_indices
        else:
            infrequent_indices = [None] * n_features
        # ... other code
```
### 14 - sklearn/preprocessing/_encoders.py:

Start line: 69, End line: 149

```python
class _BaseEncoder(TransformerMixin, BaseEstimator):

    def _fit(
        self, X, handle_unknown="error", force_all_finite=True, return_counts=False
    ):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )
        self.n_features_in_ = n_features

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []
        category_counts = []

        for i in range(n_features):
            Xi = X_list[i]

            if self.categories == "auto":
                result = _unique(Xi, return_counts=return_counts)
                if return_counts:
                    cats, counts = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                if np.issubdtype(Xi.dtype, np.str_):
                    # Always convert string categories to objects to avoid
                    # unexpected string truncation for longer category labels
                    # passed in the constructor.
                    Xi_dtype = object
                else:
                    Xi_dtype = Xi.dtype

                cats = np.array(self.categories[i], dtype=Xi_dtype)
                if (
                    cats.dtype == object
                    and isinstance(cats[0], bytes)
                    and Xi.dtype.kind != "S"
                ):
                    msg = (
                        f"In column {i}, the predefined categories have type 'bytes'"
                        " which is incompatible with values of type"
                        f" '{type(Xi[0]).__name__}'."
                    )
                    raise ValueError(msg)

                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                        np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
                    ):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                if return_counts:
                    category_counts.append(_get_counts(Xi, cats))

            self.categories_.append(cats)

        output = {"n_samples": n_samples}
        if return_counts:
            output["category_counts"] = category_counts
        return output
```
### 17 - sklearn/preprocessing/_encoders.py:

Start line: 1235, End line: 1256

```python
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):

    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        "dtype": "no_validation",  # validation delegated to numpy
        "encoded_missing_value": [Integral, type(np.nan)],
        "handle_unknown": [StrOptions({"error", "use_encoded_value"})],
        "unknown_value": [Integral, type(np.nan), None],
    }

    def __init__(
        self,
        *,
        categories="auto",
        dtype=np.float64,
        handle_unknown="error",
        unknown_value=None,
        encoded_missing_value=np.nan,
    ):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
```
### 18 - sklearn/preprocessing/_encoders.py:

Start line: 1258, End line: 1360

```python
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):

    def fit(self, X, y=None):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        self._validate_params()

        if self.handle_unknown == "use_encoded_value":
            if is_scalar_nan(self.unknown_value):
                if np.dtype(self.dtype).kind != "f":
                    raise ValueError(
                        "When unknown_value is np.nan, the dtype "
                        "parameter should be "
                        f"a float dtype. Got {self.dtype}."
                    )
            elif not isinstance(self.unknown_value, numbers.Integral):
                raise TypeError(
                    "unknown_value should be an integer or "
                    "np.nan when "
                    "handle_unknown is 'use_encoded_value', "
                    f"got {self.unknown_value}."
                )
        elif self.unknown_value is not None:
            raise TypeError(
                "unknown_value should only be set when "
                "handle_unknown is 'use_encoded_value', "
                f"got {self.unknown_value}."
            )

        # `_fit` will only raise an error when `self.handle_unknown="error"`
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan")

        cardinalities = [len(categories) for categories in self.categories_]

        # stores the missing indices per category
        self._missing_indices = {}
        for cat_idx, categories_for_idx in enumerate(self.categories_):
            for i, cat in enumerate(categories_for_idx):
                if is_scalar_nan(cat):
                    self._missing_indices[cat_idx] = i

                    # missing values are not considered part of the cardinality
                    # when considering unknown categories or encoded_missing_value
                    cardinalities[cat_idx] -= 1
                    continue

        if self.handle_unknown == "use_encoded_value":
            for cardinality in cardinalities:
                if 0 <= self.unknown_value < cardinality:
                    raise ValueError(
                        "The used value for unknown_value "
                        f"{self.unknown_value} is one of the "
                        "values already used for encoding the "
                        "seen categories."
                    )

        if self._missing_indices:
            if np.dtype(self.dtype).kind != "f" and is_scalar_nan(
                self.encoded_missing_value
            ):
                raise ValueError(
                    "There are missing values in features "
                    f"{list(self._missing_indices)}. For OrdinalEncoder to "
                    f"encode missing values with dtype: {self.dtype}, set "
                    "encoded_missing_value to a non-nan value, or "
                    "set dtype to a float"
                )

            if not is_scalar_nan(self.encoded_missing_value):
                # Features are invalid when they contain a missing category
                # and encoded_missing_value was already used to encode a
                # known category
                invalid_features = [
                    cat_idx
                    for cat_idx, cardinality in enumerate(cardinalities)
                    if cat_idx in self._missing_indices
                    and 0 <= self.encoded_missing_value < cardinality
                ]

                if invalid_features:
                    # Use feature names if they are avaliable
                    if hasattr(self, "feature_names_in_"):
                        invalid_features = self.feature_names_in_[invalid_features]
                    raise ValueError(
                        f"encoded_missing_value ({self.encoded_missing_value}) "
                        "is already used to encode a known category in features: "
                        f"{invalid_features}"
                    )

        return self
```
### 19 - sklearn/preprocessing/_encoders.py:

Start line: 1114, End line: 1233

```python
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

        .. versionadded:: 0.24

    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.

        .. versionadded:: 0.24

    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.

        .. versionadded:: 1.1

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OneHotEncoder : Performs a one-hot encoding of categorical features.
    LabelEncoder : Encodes target labels with values between 0 and
        ``n_classes-1``.

    Notes
    -----
    With a high proportion of `nan` values, inferring categories becomes slow with
    Python versions before 3.10. The handling of `nan` values was improved
    from Python 3.10 onwards, (c.f.
    `bpo-43475 <https://github.com/python/cpython/issues/87641>`_).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    By default, :class:`OrdinalEncoder` is lenient towards missing values by
    propagating them.

    >>> import numpy as np
    >>> X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
    >>> enc.fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., nan]])

    You can use the parameter `encoded_missing_value` to encode missing values.

    >>> enc.set_params(encoded_missing_value=-1).fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., -1.]])
    """
```
### 21 - sklearn/preprocessing/_encoders.py:

Start line: 1390, End line: 1450

```python
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan")

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.result_type(*[cat.dtype for cat in self.categories_])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        found_unknown = {}

        for i in range(n_features):
            labels = X[:, i]

            # replace values of X[:, i] that were nan with actual indices
            if i in self._missing_indices:
                X_i_mask = _get_mask(labels, self.encoded_missing_value)
                labels[X_i_mask] = self._missing_indices[i]

            if self.handle_unknown == "use_encoded_value":
                unknown_labels = _get_mask(labels, self.unknown_value)

                known_labels = ~unknown_labels
                X_tr[known_labels, i] = self.categories_[i][
                    labels[known_labels].astype("int64", copy=False)
                ]
                found_unknown[i] = unknown_labels
            else:
                X_tr[:, i] = self.categories_[i][labels.astype("int64", copy=False)]

        # insert None values for unknown values
        if found_unknown:
            X_tr = X_tr.astype(object, copy=False)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr
```
### 22 - sklearn/preprocessing/_encoders.py:

Start line: 1101, End line: 1111

```python
class OneHotEncoder(_BaseEncoder):

    def _check_get_feature_name_combiner(self):
        if self.feature_name_combiner == "concat":
            return lambda feature, category: feature + "_" + str(category)
        else:  # callable
            dry_run_combiner = self.feature_name_combiner("feature", "category")
            if not isinstance(dry_run_combiner, str):
                raise TypeError(
                    "When `feature_name_combiner` is a callable, it should return a "
                    f"Python string. Got {type(dry_run_combiner)} instead."
                )
            return self.feature_name_combiner
```
### 23 - sklearn/preprocessing/_encoders.py:

Start line: 837, End line: 880

```python
class OneHotEncoder(_BaseEncoder):

    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
            Fitted encoder.
        """
        self._validate_params()

        if self.sparse != "deprecated":
            warnings.warn(
                "`sparse` was renamed to `sparse_output` in version 1.2 and "
                "will be removed in 1.4. `sparse_output` is ignored unless you "
                "leave `sparse` to its default value.",
                FutureWarning,
            )
            self.sparse_output = self.sparse

        self._check_infrequent_enabled()

        fit_results = self._fit(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            return_counts=self._infrequent_enabled,
        )
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(
                fit_results["n_samples"], fit_results["category_counts"]
            )
        self.drop_idx_ = self._compute_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self
```
### 24 - sklearn/preprocessing/_encoders.py:

Start line: 1362, End line: 1388

```python
class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):

    def transform(self, X):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_int, X_mask = self._transform(
            X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan"
        )
        X_trans = X_int.astype(self.dtype, copy=False)

        for cat_idx, missing_idx in self._missing_indices.items():
            X_missing_mask = X_int[:, cat_idx] == missing_idx
            X_trans[X_missing_mask, cat_idx] = self.encoded_missing_value

        # create separate category for unknown values
        if self.handle_unknown == "use_encoded_value":
            X_trans[~X_mask] = self.unknown_value
        return X_trans
```
### 25 - sklearn/preprocessing/_encoders.py:

Start line: 25, End line: 67

```python
class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, force_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = force_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features
```
### 30 - sklearn/preprocessing/_encoders.py:

Start line: 1066, End line: 1099

```python
class OneHotEncoder(_BaseEncoder):

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        cats = [
            self._compute_transformed_categories(i)
            for i, _ in enumerate(self.categories_)
        ]

        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)
```
### 33 - sklearn/preprocessing/_encoders.py:

Start line: 5, End line: 22

```python
import numbers
from numbers import Integral, Real
import warnings

import numpy as np
from scipy import sparse

from ..base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from ..utils import check_array, is_scalar_nan, _safe_indexing
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_feature_names_in
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..utils._mask import _get_mask

from ..utils._encode import _encode, _check_unknown, _unique, _get_counts


__all__ = ["OneHotEncoder", "OrdinalEncoder"]
```
