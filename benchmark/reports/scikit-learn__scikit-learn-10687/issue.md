# scikit-learn__scikit-learn-10687

* repo: scikit-learn/scikit-learn
* base_commit: `69e9111b437084f99011dde6ab8ccc848c8c3783`

## Problem statement

Shape of `coef_` wrong for linear_model.Lasso when using `fit_intercept=False` 
<!--
If your issue is a usage question, submit it here instead:
- StackOverflow with the scikit-learn tag: http://stackoverflow.com/questions/tagged/scikit-learn
- Mailing List: https://mail.python.org/mailman/listinfo/scikit-learn
For more information, see User Questions: http://scikit-learn.org/stable/support.html#user-questions
-->

<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
Shape of `coef_` wrong for linear_model.Lasso when using `fit_intercept=False` 

#### Steps/Code to Reproduce
Example:

```python
import numpy as np
from sklearn import linear_model

est_intercept = linear_model.Lasso(fit_intercept=True)
est_intercept.fit(np.c_[np.ones(3)], np.ones(3))
assert est_intercept.coef_.shape  == (1,)
```

```python
import numpy as np
from sklearn import linear_model

est_no_intercept = linear_model.Lasso(fit_intercept=False)
est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
assert est_no_intercept.coef_.shape  == (1,)
```

#### Expected Results
the second snippet should not raise, but it does. The first snippet is ok. I pasted it as a reference

#### Actual Results
```python
In [2]: %paste
import numpy as np
from sklearn import linear_model
est_intercept = linear_model.Lasso(fit_intercept=True)
est_intercept.fit(np.c_[np.ones(3)], np.ones(3))
assert est_intercept.coef_.shape  == (1,)



In [3]: %paste
import numpy as np
from sklearn import linear_model

est_no_intercept = linear_model.Lasso(fit_intercept=False)
est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
assert est_no_intercept.coef_.shape  == (1,)


---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-3-5ffa9cfd4df7> in <module>()
      4 est_no_intercept = linear_model.Lasso(fit_intercept=False)
      5 est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
----> 6 assert est_no_intercept.coef_.shape  == (1,)

AssertionError:
```

#### Versions
Linux-3.2.0-4-amd64-x86_64-with-debian-7.11
('Python', '2.7.3 (default, Mar 13 2014, 11:03:55) \n[GCC 4.7.2]')
('NumPy', '1.13.3')
('SciPy', '0.19.1')
('Scikit-Learn', '0.18.2')



<!-- Thanks for contributing! -->

[MRG] Shape of `coef_` wrong for linear_model.Lasso when using `fit_intercept=False`
<!--
Thanks for contributing a pull request! Please ensure you have taken a look at
the contribution guidelines: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#Contributing-Pull-Requests
-->
#### Reference Issue
Fixes #10571 


## Patch

```diff
diff --git a/sklearn/linear_model/coordinate_descent.py b/sklearn/linear_model/coordinate_descent.py
--- a/sklearn/linear_model/coordinate_descent.py
+++ b/sklearn/linear_model/coordinate_descent.py
@@ -762,8 +762,12 @@ def fit(self, X, y, check_input=True):
 
         if n_targets == 1:
             self.n_iter_ = self.n_iter_[0]
+            self.coef_ = coef_[0]
+            self.dual_gap_ = dual_gaps_[0]
+        else:
+            self.coef_ = coef_
+            self.dual_gap_ = dual_gaps_
 
-        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
         self._set_intercept(X_offset, y_offset, X_scale)
 
         # workaround since _set_intercept will cast self.coef_ into X.dtype

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_coordinate_descent.py b/sklearn/linear_model/tests/test_coordinate_descent.py
--- a/sklearn/linear_model/tests/test_coordinate_descent.py
+++ b/sklearn/linear_model/tests/test_coordinate_descent.py
@@ -803,3 +803,9 @@ def test_enet_l1_ratio():
         est.fit(X, y[:, None])
         est_desired.fit(X, y[:, None])
     assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)
+
+
+def test_coef_shape_not_zero():
+    est_no_intercept = Lasso(fit_intercept=False)
+    est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
+    assert est_no_intercept.coef_.shape == (1,)

```
