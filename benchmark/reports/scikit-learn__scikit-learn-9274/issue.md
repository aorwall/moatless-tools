# scikit-learn__scikit-learn-9274

* repo: scikit-learn/scikit-learn
* base_commit: `faa940608befaeca99db501609c6db796739f30f`

## Problem statement

Training MLP using l-bfgs limited to default l-bfgs maxiter value
#### Description

Training an MLP regressor (or classifier) using l-bfgs currently cannot run for more than (approx) 15000 iterations.
This artificial limit is caused by the call site to l-bfgs passing the MLP argument value "max_iters" to the argument for "maxfun" (maximum number of function calls), but not for "maxiter" (maximum number of iterations), so that no matter how large a number you pass as "max_iters" to train for MLP, the iterations are capped by the default value for maxiter (15000).

#### Steps/Code to Reproduce
Fit an MLP for a problem that requires > 15000 iterations

Here is an example (tested in python 2.7):
https://gist.github.com/daniel-perry/d9e356a03936673e58e0ce47d5fc70ef

(you will need data.npy from the gist linked to above)

````
from __future__ import print_function
import numpy as np
from sklearn.neural_network import MLPRegressor

train = np.load("data.npy").tolist()

max_iter = 18000
clf = MLPRegressor(max_iter=max_iter, activation='relu', solver='lbfgs', verbose=True)

clf.fit(train["train_x"],train["train_y"])

print("score: ", clf.score(train["train_x"],train["train_y"]))
print("iters: ", clf.n_iter_, " / ", max_iter)
````

#### Expected Results

The training should run for 18000 iterations.

#### Actual Results

The training runs for 15000 iterations.

#### Versions

Here are my local version details, though the problem appears to exist on the current head, and so should exist for any python/sklearn versions.

'Python', '2.7.12 (default, Jul  1 2016, 15:12:24) \n[GCC 5.4.0 20160609]'
'NumPy', '1.13.0'
'SciPy', '0.19.1'
'Scikit-Learn', '0.18'



[WIP] FIX: use maxiter rather than maxfun in MultiLayerPerceptron with solver='lbfgs'
In my limited experience with LBFGS, the number of function calls is greater than the number of iterations.

The impact of this bug is that with solver='lbfgs' is probably not doing as many iterations as it should in master although I am not sure it matters that much in practice.

To get an idea how much funtion calls differ from iterations, I tweaked `examples/neural_networks/plot_mnist_filters.py` to be able to run for a few hundred iterations:

```py
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                    solver='lbfgs', verbose=10, tol=1e-16, random_state=1,
                    learning_rate_init=.1)
```

The result: 393 iterations and 414 function calls.

Not sure whether we nest to test this, and how to test it, suggestions more than welcome!

- [ ] add a whats_new entry once there is agreement


## Patch

```diff
diff --git a/sklearn/neural_network/multilayer_perceptron.py b/sklearn/neural_network/multilayer_perceptron.py
--- a/sklearn/neural_network/multilayer_perceptron.py
+++ b/sklearn/neural_network/multilayer_perceptron.py
@@ -51,7 +51,7 @@ def __init__(self, hidden_layer_sizes, activation, solver,
                  max_iter, loss, shuffle, random_state, tol, verbose,
                  warm_start, momentum, nesterovs_momentum, early_stopping,
                  validation_fraction, beta_1, beta_2, epsilon,
-                 n_iter_no_change):
+                 n_iter_no_change, max_fun):
         self.activation = activation
         self.solver = solver
         self.alpha = alpha
@@ -75,6 +75,7 @@ def __init__(self, hidden_layer_sizes, activation, solver,
         self.beta_2 = beta_2
         self.epsilon = epsilon
         self.n_iter_no_change = n_iter_no_change
+        self.max_fun = max_fun
 
     def _unpack(self, packed_parameters):
         """Extract the coefficients and intercepts from packed_parameters."""
@@ -172,7 +173,6 @@ def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
         self._unpack(packed_coef_inter)
         loss, coef_grads, intercept_grads = self._backprop(
             X, y, activations, deltas, coef_grads, intercept_grads)
-        self.n_iter_ += 1
         grad = _pack(coef_grads, intercept_grads)
         return loss, grad
 
@@ -381,6 +381,8 @@ def _validate_hyperparameters(self):
                              self.shuffle)
         if self.max_iter <= 0:
             raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
+        if self.max_fun <= 0:
+            raise ValueError("max_fun must be > 0, got %s." % self.max_fun)
         if self.alpha < 0.0:
             raise ValueError("alpha must be >= 0, got %s." % self.alpha)
         if (self.learning_rate in ["constant", "invscaling", "adaptive"] and
@@ -459,10 +461,29 @@ def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
         optimal_parameters, self.loss_, d = fmin_l_bfgs_b(
             x0=packed_coef_inter,
             func=self._loss_grad_lbfgs,
-            maxfun=self.max_iter,
+            maxfun=self.max_fun,
+            maxiter=self.max_iter,
             iprint=iprint,
             pgtol=self.tol,
             args=(X, y, activations, deltas, coef_grads, intercept_grads))
+        self.n_iter_ = d['nit']
+        if d['warnflag'] == 1:
+            if d['nit'] >= self.max_iter:
+                warnings.warn(
+                    "LBFGS Optimizer: Maximum iterations (%d) "
+                    "reached and the optimization hasn't converged yet."
+                    % self.max_iter, ConvergenceWarning)
+            if d['funcalls'] >= self.max_fun:
+                warnings.warn(
+                    "LBFGS Optimizer: Maximum function evaluations (%d) "
+                    "reached and the optimization hasn't converged yet."
+                    % self.max_fun, ConvergenceWarning)
+        elif d['warnflag'] == 2:
+            warnings.warn(
+                "LBFGS Optimizer: Optimization hasn't converged yet, "
+                "cause of LBFGS stopping: %s."
+                % d['task'], ConvergenceWarning)
+
 
         self._unpack(optimal_parameters)
 
@@ -833,6 +854,15 @@ class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):
 
         .. versionadded:: 0.20
 
+    max_fun : int, optional, default 15000
+        Only used when solver='lbfgs'. Maximum number of loss function calls.
+        The solver iterates until convergence (determined by 'tol'), number
+        of iterations reaches max_iter, or this number of loss function calls.
+        Note that number of loss function calls will be greater than or equal
+        to the number of iterations for the `MLPClassifier`.
+
+        .. versionadded:: 0.22
+
     Attributes
     ----------
     classes_ : array or list of array of shape (n_classes,)
@@ -898,8 +928,7 @@ def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                  verbose=False, warm_start=False, momentum=0.9,
                  nesterovs_momentum=True, early_stopping=False,
                  validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
-                 epsilon=1e-8, n_iter_no_change=10):
-
+                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
         super().__init__(
             hidden_layer_sizes=hidden_layer_sizes,
             activation=activation, solver=solver, alpha=alpha,
@@ -912,7 +941,7 @@ def __init__(self, hidden_layer_sizes=(100,), activation="relu",
             early_stopping=early_stopping,
             validation_fraction=validation_fraction,
             beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
-            n_iter_no_change=n_iter_no_change)
+            n_iter_no_change=n_iter_no_change, max_fun=max_fun)
 
     def _validate_input(self, X, y, incremental):
         X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
@@ -1216,6 +1245,15 @@ class MLPRegressor(BaseMultilayerPerceptron, RegressorMixin):
 
         .. versionadded:: 0.20
 
+    max_fun : int, optional, default 15000
+        Only used when solver='lbfgs'. Maximum number of function calls.
+        The solver iterates until convergence (determined by 'tol'), number
+        of iterations reaches max_iter, or this number of function calls.
+        Note that number of function calls will be greater than or equal to
+        the number of iterations for the MLPRegressor.
+
+        .. versionadded:: 0.22
+
     Attributes
     ----------
     loss_ : float
@@ -1279,8 +1317,7 @@ def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                  verbose=False, warm_start=False, momentum=0.9,
                  nesterovs_momentum=True, early_stopping=False,
                  validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
-                 epsilon=1e-8, n_iter_no_change=10):
-
+                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
         super().__init__(
             hidden_layer_sizes=hidden_layer_sizes,
             activation=activation, solver=solver, alpha=alpha,
@@ -1293,7 +1330,7 @@ def __init__(self, hidden_layer_sizes=(100,), activation="relu",
             early_stopping=early_stopping,
             validation_fraction=validation_fraction,
             beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
-            n_iter_no_change=n_iter_no_change)
+            n_iter_no_change=n_iter_no_change, max_fun=max_fun)
 
     def predict(self, X):
         """Predict using the multi-layer perceptron model.

```

## Test Patch

```diff
diff --git a/sklearn/neural_network/tests/test_mlp.py b/sklearn/neural_network/tests/test_mlp.py
--- a/sklearn/neural_network/tests/test_mlp.py
+++ b/sklearn/neural_network/tests/test_mlp.py
@@ -48,6 +48,8 @@
 Xboston = StandardScaler().fit_transform(boston.data)[: 200]
 yboston = boston.target[:200]
 
+regression_datasets = [(Xboston, yboston)]
+
 iris = load_iris()
 
 X_iris = iris.data
@@ -228,32 +230,30 @@ def loss_grad_fun(t):
             assert_almost_equal(numgrad, grad)
 
 
-def test_lbfgs_classification():
+@pytest.mark.parametrize('X,y', classification_datasets)
+def test_lbfgs_classification(X, y):
     # Test lbfgs on classification.
     # It should achieve a score higher than 0.95 for the binary and multi-class
     # versions of the digits dataset.
-    for X, y in classification_datasets:
-        X_train = X[:150]
-        y_train = y[:150]
-        X_test = X[150:]
-
-        expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)
-
-        for activation in ACTIVATION_TYPES:
-            mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
-                                max_iter=150, shuffle=True, random_state=1,
-                                activation=activation)
-            mlp.fit(X_train, y_train)
-            y_predict = mlp.predict(X_test)
-            assert mlp.score(X_train, y_train) > 0.95
-            assert ((y_predict.shape[0], y_predict.dtype.kind) ==
-                         expected_shape_dtype)
+    X_train = X[:150]
+    y_train = y[:150]
+    X_test = X[150:]
+    expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)
 
-
-def test_lbfgs_regression():
+    for activation in ACTIVATION_TYPES:
+        mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
+                            max_iter=150, shuffle=True, random_state=1,
+                            activation=activation)
+        mlp.fit(X_train, y_train)
+        y_predict = mlp.predict(X_test)
+        assert mlp.score(X_train, y_train) > 0.95
+        assert ((y_predict.shape[0], y_predict.dtype.kind) ==
+                expected_shape_dtype)
+
+
+@pytest.mark.parametrize('X,y', regression_datasets)
+def test_lbfgs_regression(X, y):
     # Test lbfgs on the boston dataset, a regression problems.
-    X = Xboston
-    y = yboston
     for activation in ACTIVATION_TYPES:
         mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,
                            max_iter=150, shuffle=True, random_state=1,
@@ -266,6 +266,39 @@ def test_lbfgs_regression():
             assert mlp.score(X, y) > 0.95
 
 
+@pytest.mark.parametrize('X,y', classification_datasets)
+def test_lbfgs_classification_maxfun(X, y):
+    # Test lbfgs parameter max_fun.
+    # It should independently limit the number of iterations for lbfgs.
+    max_fun = 10
+    # classification tests
+    for activation in ACTIVATION_TYPES:
+        mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
+                            max_iter=150, max_fun=max_fun, shuffle=True,
+                            random_state=1, activation=activation)
+        with pytest.warns(ConvergenceWarning):
+            mlp.fit(X, y)
+            assert max_fun >= mlp.n_iter_
+
+
+@pytest.mark.parametrize('X,y', regression_datasets)
+def test_lbfgs_regression_maxfun(X, y):
+    # Test lbfgs parameter max_fun.
+    # It should independently limit the number of iterations for lbfgs.
+    max_fun = 10
+    # regression tests
+    for activation in ACTIVATION_TYPES:
+        mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,
+                           max_iter=150, max_fun=max_fun, shuffle=True,
+                           random_state=1, activation=activation)
+        with pytest.warns(ConvergenceWarning):
+            mlp.fit(X, y)
+            assert max_fun >= mlp.n_iter_
+
+    mlp.max_fun = -1
+    assert_raises(ValueError, mlp.fit, X, y)
+
+
 def test_learning_rate_warmstart():
     # Tests that warm_start reuse past solutions.
     X = [[3, 2], [1, 6], [5, 6], [-2, -4]]

```
