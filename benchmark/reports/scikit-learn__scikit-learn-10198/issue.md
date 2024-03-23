# scikit-learn__scikit-learn-10198

* repo: scikit-learn/scikit-learn
* base_commit: `726fa36f2556e0d604d85a1de48ba56a8b6550db`

## Problem statement

add get_feature_names to CategoricalEncoder
We should add a ``get_feature_names`` to the new CategoricalEncoder, as discussed [here](https://github.com/scikit-learn/scikit-learn/pull/9151#issuecomment-345830056). I think it would be good to be consistent with the PolynomialFeature which allows passing in original feature names to map them to new feature names. Also see #6425.


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
