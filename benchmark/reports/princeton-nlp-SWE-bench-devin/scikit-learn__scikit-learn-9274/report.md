# scikit-learn__scikit-learn-9274

| **scikit-learn/scikit-learn** | `faa940608befaeca99db501609c6db796739f30f` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 33477 |
| **Any found context length** | 1876 |
| **Avg pos** | 152.0 |
| **Min pos** | 1 |
| **Max pos** | 59 |
| **Top file pos** | 1 |
| **Missing snippets** | 11 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/neural_network/multilayer_perceptron.py | 54 | 54 | 20 | 1 | 11963
| sklearn/neural_network/multilayer_perceptron.py | 78 | 78 | 20 | 1 | 11963
| sklearn/neural_network/multilayer_perceptron.py | 175 | 175 | 59 | 1 | 33477
| sklearn/neural_network/multilayer_perceptron.py | 384 | 384 | 18 | 1 | 11214
| sklearn/neural_network/multilayer_perceptron.py | 462 | 462 | 14 | 1 | 8906
| sklearn/neural_network/multilayer_perceptron.py | 836 | 836 | 1 | 1 | 1876
| sklearn/neural_network/multilayer_perceptron.py | 901 | 902 | 4 | 1 | 4586
| sklearn/neural_network/multilayer_perceptron.py | 915 | 915 | 4 | 1 | 4586
| sklearn/neural_network/multilayer_perceptron.py | 1219 | 1219 | 2 | 1 | 3719
| sklearn/neural_network/multilayer_perceptron.py | 1282 | 1283 | 5 | 1 | 4902
| sklearn/neural_network/multilayer_perceptron.py | 1296 | 1296 | 5 | 1 | 4902


## Problem Statement

```
Training MLP using l-bfgs limited to default l-bfgs maxiter value
#### Description

Training an MLP regressor (or classifier) using l-bfgs currently cannot run for more than (approx) 15000 iterations.
This artificial limit is caused by the call site to l-bfgs passing the MLP argument value "max_iters" to the argument for "maxfun" (maximum number of function calls), but not for "maxiter" (maximum number of iterations), so that no matter how large a number you pass as "max_iters" to train for MLP, the iterations are capped by the default value for maxiter (15000).

#### Steps/Code to Reproduce
Fit an MLP for a problem that requires > 15000 iterations

Here is an example (tested in python 2.7):
https://gist.github.com/daniel-perry/d9e356a03936673e58e0ce47d5fc70ef

(you will need data.npy from the gist linked to above)

\`\`\``
from __future__ import print_function
import numpy as np
from sklearn.neural_network import MLPRegressor

train = np.load("data.npy").tolist()

max_iter = 18000
clf = MLPRegressor(max_iter=max_iter, activation='relu', solver='lbfgs', verbose=True)

clf.fit(train["train_x"],train["train_y"])

print("score: ", clf.score(train["train_x"],train["train_y"]))
print("iters: ", clf.n_iter_, " / ", max_iter)
\`\`\``

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

\`\`\`py
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                    solver='lbfgs', verbose=10, tol=1e-16, random_state=1,
                    learning_rate_init=.1)
\`\`\`

The result: 393 iterations and 414 function calls.

Not sure whether we nest to test this, and how to test it, suggestions more than welcome!

- [ ] add a whats_new entry once there is agreement

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/neural_network/multilayer_perceptron.py** | 687 | 892| 1876 | 1876 | 11623 | 
| **-> 2 <-** | **1 sklearn/neural_network/multilayer_perceptron.py** | 1071 | 1272| 1843 | 3719 | 11623 | 
| 3 | 2 examples/neural_networks/plot_mlp_training_curves.py | 1 | 53| 555 | 4274 | 12610 | 
| **-> 4 <-** | **2 sklearn/neural_network/multilayer_perceptron.py** | 893 | 915| 312 | 4586 | 12610 | 
| **-> 5 <-** | **2 sklearn/neural_network/multilayer_perceptron.py** | 1273 | 1296| 316 | 4902 | 12610 | 
| 6 | **2 sklearn/neural_network/multilayer_perceptron.py** | 917 | 942| 291 | 5193 | 12610 | 
| 7 | 3 examples/neural_networks/plot_mlp_alpha.py | 1 | 54| 427 | 5620 | 13728 | 
| 8 | 3 examples/neural_networks/plot_mlp_training_curves.py | 56 | 85| 270 | 5890 | 13728 | 
| 9 | 4 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 50 | 116| 672 | 6562 | 14741 | 
| 10 | 5 examples/neural_networks/plot_mnist_filters.py | 1 | 57| 582 | 7144 | 15323 | 
| 11 | **5 sklearn/neural_network/multilayer_perceptron.py** | 310 | 376| 623 | 7767 | 15323 | 
| 12 | **5 sklearn/neural_network/multilayer_perceptron.py** | 507 | 573| 610 | 8377 | 15323 | 
| 13 | 5 examples/neural_networks/plot_mlp_training_curves.py | 88 | 103| 162 | 8539 | 15323 | 
| **-> 14 <-** | **5 sklearn/neural_network/multilayer_perceptron.py** | 429 | 467| 367 | 8906 | 15323 | 
| 15 | 5 examples/neural_networks/plot_mlp_alpha.py | 55 | 114| 671 | 9577 | 15323 | 
| 16 | 6 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 80| 660 | 10237 | 16014 | 
| 17 | 7 benchmarks/bench_mnist.py | 84 | 105| 306 | 10543 | 17725 | 
| **-> 18 <-** | **7 sklearn/neural_network/multilayer_perceptron.py** | 378 | 427| 671 | 11214 | 17725 | 
| 19 | **7 sklearn/neural_network/multilayer_perceptron.py** | 575 | 604| 319 | 11533 | 17725 | 
| **-> 20 <-** | **7 sklearn/neural_network/multilayer_perceptron.py** | 39 | 86| 430 | 11963 | 17725 | 
| 21 | 8 sklearn/neural_network/_stochastic_optimizers.py | 1 | 71| 390 | 12353 | 19656 | 
| 22 | 9 benchmarks/bench_tree.py | 64 | 125| 523 | 12876 | 20526 | 
| 23 | 10 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 59 | 124| 700 | 13576 | 21708 | 
| 24 | 11 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 14073 | 22737 | 
| 25 | 12 benchmarks/bench_glmnet.py | 47 | 129| 796 | 14869 | 23824 | 
| 26 | 13 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 15641 | 24988 | 
| 27 | **13 sklearn/neural_network/multilayer_perceptron.py** | 1 | 36| 253 | 15894 | 24988 | 
| 28 | 14 examples/model_selection/plot_underfitting_overfitting.py | 1 | 72| 631 | 16525 | 25619 | 
| 29 | 14 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 65 | 123| 525 | 17050 | 25619 | 
| 30 | **14 sklearn/neural_network/multilayer_perceptron.py** | 624 | 647| 178 | 17228 | 25619 | 
| 31 | 15 examples/inspection/plot_partial_dependence.py | 72 | 138| 653 | 17881 | 26960 | 
| 32 | **15 sklearn/neural_network/multilayer_perceptron.py** | 469 | 505| 368 | 18249 | 26960 | 
| 33 | 16 examples/text/plot_document_classification_20newsgroups.py | 247 | 323| 658 | 18907 | 29495 | 
| 34 | **16 sklearn/neural_network/multilayer_perceptron.py** | 179 | 254| 722 | 19629 | 29495 | 
| 35 | 17 benchmarks/bench_lasso.py | 60 | 97| 382 | 20011 | 30290 | 
| 36 | 18 benchmarks/bench_hist_gradient_boosting.py | 59 | 132| 728 | 20739 | 32516 | 
| 37 | **18 sklearn/neural_network/multilayer_perceptron.py** | 984 | 1012| 234 | 20973 | 32516 | 
| 38 | 19 sklearn/utils/estimator_checks.py | 2249 | 2282| 319 | 21292 | 54710 | 
| 39 | 20 benchmarks/bench_plot_omp_lars.py | 101 | 123| 306 | 21598 | 55874 | 
| 40 | 21 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 22138 | 57398 | 
| 41 | 22 examples/linear_model/plot_lasso_model_selection.py | 1 | 81| 714 | 22852 | 58768 | 
| 42 | 22 benchmarks/bench_mnist.py | 108 | 180| 695 | 23547 | 58768 | 
| 43 | **22 sklearn/neural_network/multilayer_perceptron.py** | 1014 | 1042| 228 | 23775 | 58768 | 
| 44 | 23 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 947 | 24722 | 63149 | 
| 45 | 24 sklearn/linear_model/coordinate_descent.py | 1399 | 1577| 1649 | 26371 | 83710 | 
| 46 | 25 examples/gaussian_process/plot_gpc.py | 1 | 78| 757 | 27128 | 84742 | 
| 47 | 26 benchmarks/bench_plot_lasso_path.py | 83 | 116| 347 | 27475 | 85716 | 
| 48 | 26 benchmarks/bench_plot_lasso_path.py | 1 | 80| 626 | 28101 | 85716 | 
| 49 | 27 examples/applications/plot_out_of_core_classification.py | 187 | 214| 241 | 28342 | 89021 | 
| 50 | **27 sklearn/neural_network/multilayer_perceptron.py** | 965 | 982| 142 | 28484 | 89021 | 
| 51 | 28 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 140| 520 | 29004 | 90142 | 
| 52 | 29 examples/gaussian_process/plot_gpr_noisy.py | 1 | 67| 743 | 29747 | 91265 | 
| 53 | 30 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 30314 | 91832 | 
| 54 | 31 benchmarks/bench_tsne_mnist.py | 69 | 169| 1011 | 31325 | 93259 | 
| 55 | 32 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 775 | 32100 | 95083 | 
| 56 | 32 benchmarks/bench_lasso.py | 1 | 18| 111 | 32211 | 95083 | 
| 57 | 32 benchmarks/bench_mnist.py | 1 | 57| 499 | 32710 | 95083 | 
| 58 | **32 sklearn/neural_network/multilayer_perceptron.py** | 256 | 293| 320 | 33030 | 95083 | 
| **-> 59 <-** | **32 sklearn/neural_network/multilayer_perceptron.py** | 130 | 177| 447 | 33477 | 95083 | 
| 60 | 33 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 33991 | 98981 | 
| 61 | 33 benchmarks/bench_glmnet.py | 1 | 29| 168 | 34159 | 98981 | 
| 62 | 33 examples/neural_networks/plot_rbm_logistic_classification.py | 1 | 40| 302 | 34461 | 98981 | 
| 63 | 33 benchmarks/bench_plot_nmf.py | 105 | 148| 401 | 34862 | 98981 | 
| 64 | 34 examples/cluster/plot_dict_face_patches.py | 1 | 85| 660 | 35522 | 99641 | 
| 65 | 34 examples/linear_model/plot_lasso_model_selection.py | 95 | 159| 513 | 36035 | 99641 | 
| 66 | **34 sklearn/neural_network/multilayer_perceptron.py** | 295 | 308| 143 | 36178 | 99641 | 
| 67 | 35 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 36845 | 100308 | 
| 68 | 35 benchmarks/bench_plot_omp_lars.py | 1 | 98| 857 | 37702 | 100308 | 
| 69 | 36 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 612 | 38314 | 100945 | 
| 70 | 37 examples/applications/plot_prediction_latency.py | 267 | 311| 348 | 38662 | 103561 | 
| 71 | 38 examples/linear_model/plot_multi_task_lasso_support.py | 1 | 70| 587 | 39249 | 104173 | 
| 72 | 39 examples/linear_model/plot_sgd_early_stopping.py | 1 | 55| 462 | 39711 | 105470 | 
| 73 | 40 examples/linear_model/plot_logistic_path.py | 1 | 76| 503 | 40214 | 105998 | 
| 74 | 41 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 40760 | 106572 | 
| 75 | 42 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 86| 769 | 41529 | 107725 | 
| 76 | 43 sklearn/neural_network/rbm.py | 1 | 104| 790 | 42319 | 110514 | 
| 77 | 43 benchmarks/bench_hist_gradient_boosting.py | 1 | 37| 349 | 42668 | 110514 | 
| 78 | 44 examples/linear_model/plot_theilsen.py | 1 | 88| 785 | 43453 | 111518 | 
| 79 | 45 sklearn/linear_model/least_angle.py | 476 | 633| 1591 | 45044 | 127602 | 
| 80 | 46 examples/linear_model/plot_lasso_coordinate_descent_path.py | 1 | 78| 695 | 45739 | 128456 | 
| 81 | 47 examples/gaussian_process/plot_gpr_noisy_targets.py | 1 | 100| 755 | 46494 | 129428 | 
| 82 | 47 benchmarks/bench_hist_gradient_boosting.py | 158 | 242| 750 | 47244 | 129428 | 
| 83 | 48 examples/svm/plot_svm_scale_c.py | 1 | 102| 763 | 48007 | 130774 | 
| 84 | 49 examples/neighbors/plot_lof_novelty_detection.py | 1 | 67| 753 | 48760 | 131704 | 
| 85 | 50 examples/semi_supervised/plot_label_propagation_digits.py | 1 | 93| 588 | 49348 | 132312 | 
| 86 | 51 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 79| 756 | 50104 | 133359 | 
| 87 | 52 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 358 | 50462 | 147016 | 
| 88 | 52 benchmarks/bench_plot_nmf.py | 332 | 368| 377 | 50839 | 147016 | 
| 89 | 53 examples/linear_model/plot_bayesian_ridge_curvefit.py | 1 | 58| 432 | 51271 | 147827 | 
| 90 | 53 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 1 | 48| 321 | 51592 | 147827 | 
| 91 | 53 sklearn/utils/estimator_checks.py | 319 | 397| 707 | 52299 | 147827 | 
| 92 | 54 examples/exercises/plot_cv_diabetes.py | 1 | 80| 672 | 52971 | 148499 | 
| 93 | 55 benchmarks/bench_covertype.py | 112 | 190| 757 | 53728 | 150390 | 
| 94 | 55 sklearn/linear_model/coordinate_descent.py | 1232 | 1383| 1293 | 55021 | 150390 | 
| 95 | 56 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 55553 | 152335 | 
| 96 | 57 examples/gaussian_process/plot_gpr_co2.py | 1 | 65| 778 | 56331 | 154040 | 
| 97 | 58 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 772 | 57103 | 154961 | 
| 98 | 59 benchmarks/bench_saga.py | 6 | 104| 835 | 57938 | 157425 | 
| 99 | 59 examples/applications/plot_out_of_core_classification.py | 361 | 419| 470 | 58408 | 157425 | 
| 100 | 60 benchmarks/bench_plot_incremental_pca.py | 104 | 151| 468 | 58876 | 158781 | 
| 101 | 61 sklearn/cross_decomposition/pls_.py | 289 | 351| 838 | 59714 | 166975 | 
| 102 | 61 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 60087 | 166975 | 
| 103 | 62 examples/plot_anomaly_comparison.py | 81 | 152| 757 | 60844 | 168493 | 
| 104 | 63 benchmarks/bench_sparsify.py | 1 | 82| 754 | 61598 | 169400 | 
| 105 | 64 examples/linear_model/plot_ard.py | 1 | 101| 769 | 62367 | 170324 | 
| 106 | 64 examples/applications/plot_out_of_core_classification.py | 267 | 360| 807 | 63174 | 170324 | 
| 107 | 65 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 63967 | 171569 | 
| 108 | 66 sklearn/linear_model/logistic.py | 1192 | 1429| 2703 | 66670 | 193437 | 
| 109 | 67 examples/plot_kernel_approximation.py | 1 | 87| 735 | 67405 | 195378 | 
| 110 | 67 sklearn/utils/estimator_checks.py | 1577 | 1649| 655 | 68060 | 195378 | 
| 111 | 68 examples/model_selection/plot_learning_curve.py | 143 | 166| 257 | 68317 | 197070 | 
| 112 | **68 sklearn/neural_network/multilayer_perceptron.py** | 606 | 622| 134 | 68451 | 197070 | 
| 113 | 69 examples/linear_model/plot_lasso_lars.py | 1 | 43| 226 | 68677 | 197340 | 
| 114 | 70 examples/plot_kernel_ridge_regression.py | 80 | 153| 738 | 69415 | 199060 | 
| 115 | 70 benchmarks/bench_plot_nmf.py | 151 | 192| 476 | 69891 | 199060 | 


### Hint

```


```

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


## Code snippets

### 1 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 687, End line: 892

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):
    """Multi-layer Perceptron classifier.

    This model optimizes the log-loss function using LBFGS or stochastic
    gradient descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, optional, default 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate at each
          time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when ``solver='sgd'``.

    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, optional, default 0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    warm_start : bool, optional, default False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default 0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : boolean, default True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least tol for
        ``n_iter_no_change`` consecutive epochs. The split is stratified,
        except in a multilabel setting.
        Only effective when solver='sgd' or 'adam'

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    epsilon : float, optional, default 1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

        .. versionadded:: 0.20

    Attributes
    ----------
    classes_ : array or list of array of shape (n_classes,)
        Class labels for each output.

    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : string
        Name of the output activation function.

    Notes
    -----
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
```
### 2 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 1071, End line: 1272

```python
class MLPRegressor(BaseMultilayerPerceptron, RegressorMixin):
    """Multi-layer Perceptron regressor.

    This model optimizes the squared-loss using LBFGS or stochastic gradient
    descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, optional, default 'auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default 'constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when solver='sgd'.

    learning_rate_init : double, optional, default 0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, optional, default 0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, optional, default 200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    warm_start : bool, optional, default False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default 0.9
        Momentum for gradient descent update.  Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : boolean, default True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    epsilon : float, optional, default 1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

        .. versionadded:: 0.20

    Attributes
    ----------
    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : string
        Name of the output activation function.

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
```
### 3 - examples/neural_networks/plot_mlp_training_curves.py:

Start line: 1, End line: 53

```python
"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.
"""

print(__doc__)

import warnings

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]
```
### 4 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 893, End line: 915

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10):

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss='log_loss', shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)
```
### 5 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 1273, End line: 1296

```python
class MLPRegressor(BaseMultilayerPerceptron, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10):

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss='squared_loss', shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)
```
### 6 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 917, End line: 942

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):

    def _validate_input(self, X, y, incremental):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        if not incremental:
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        elif self.warm_start:
            classes = unique_labels(y)
            if set(classes) != set(self.classes_):
                raise ValueError("warm_start can only be used where `y` has "
                                 "the same classes as in the previous "
                                 "call to fit. Previously got %s, `y` has %s" %
                                 (self.classes_, classes))
        else:
            classes = unique_labels(y)
            if len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError("`y` has classes not in `self.classes_`."
                                 " `self.classes_` has %s. 'y' has %s." %
                                 (self.classes_, classes))

        y = self._label_binarizer.transform(y)
        return X, y
```
### 7 - examples/neural_networks/plot_mlp_alpha.py:

Start line: 1, End line: 54

```python
"""
================================================
Varying regularization in Multi-layer Perceptron
================================================

A comparison of different values for regularization parameter 'alpha' on
synthetic datasets. The plot shows that different alphas yield different
decision functions.

Alpha is a parameter for regularization term, aka penalty term, that combats
overfitting by constraining the size of the weights. Increasing alpha may fix
high variance (a sign of overfitting) by encouraging smaller weights, resulting
in a decision boundary plot that appears with lesser curvatures.
Similarly, decreasing alpha may fix high bias (a sign of underfitting) by
encouraging larger weights, potentially resulting in a more complicated
decision boundary.
"""
print(__doc__)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier

h = .02  # step size in the mesh

alphas = np.logspace(-5, 3, 5)
names = ['alpha ' + str(i) for i in alphas]

classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(solver='lbfgs', alpha=i, random_state=1,
                                     hidden_layer_sizes=[100, 100]))

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure = plt.figure(figsize=(17, 9))
i = 1
# iterate over datasets
```
### 8 - examples/neural_networks/plot_mlp_training_curves.py:

Start line: 56, End line: 85

```python
def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)
```
### 9 - examples/semi_supervised/plot_label_propagation_digits_active_learning.py:

Start line: 50, End line: 116

```python
for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1

    lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=20)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(
        lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.in1d(uncertainty_index, unlabeled_indices)][:5]

    # keep track of indices that we get labels for
    delete_indices = np.array([], dtype=int)

    # for more than 5 iterations, visualize the gain only on the first 5
    if i < 5:
        f.text(.05, (1 - (i + 1) * .183),
               "model %d\n\nfit with\n%d labels" %
               ((i + 1), i * 5 + 10), size=10)
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        # for more than 5 iterations, visualize the gain only on the first 5
        if i < 5:
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
            sub.set_title("predict: %i\ntrue: %i" % (
                lp_model.transduction_[image_index], y[image_index]), size=10)
            sub.axis('off')

        # labeling 5 points, remote from labeled set
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)

f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
           "uncertain labels to learn with the next model.", y=1.15)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                    hspace=0.85)
plt.show()
```
### 10 - examples/neural_networks/plot_mnist_filters.py:

Start line: 1, End line: 57

```python
"""
=====================================
Visualization of MLP weights on MNIST
=====================================

Sometimes looking at the learned coefficients of a neural network can provide
insight into the learning behavior. For example if weights look unstructured,
maybe some were not used at all, or if very large coefficients exist, maybe
regularization was too low or the learning rate too high.

This example shows how to plot some of the first layer weights in a
MLPClassifier trained on the MNIST dataset.

The input data consists of 28x28 pixel handwritten digits, leading to 784
features in the dataset. Therefore the first layer weight matrix have the shape
(784, hidden_layer_sizes[0]).  We can therefore visualize a single column of
the weight matrix as a 28x28 pixel image.

To make the example run faster, we use very few hidden units, and train only
for a very short time. Training longer would result in weights with a much
smoother spatial appearance.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

print(__doc__)

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
```
### 11 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 310, End line: 376

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _fit(self, X, y, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
                                           incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)

        # Run the LBFGS solver
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads,
                            intercept_grads, layer_units)
        return self
```
### 12 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 507, End line: 573

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):
        # ... other code

        try:
            for it in range(self.max_iter):
                if self.shuffle:
                    X, y = shuffle(X, y, random_state=self._random_state)
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    activations[0] = X[batch_slice]
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Validation score did not improve more than "
                               "tol=%f for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))
                    else:
                        msg = ("Training loss did not improve more than tol=%f"
                               " for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts
```
### 14 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 429, End line: 467

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_,
                                  self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        optimal_parameters, self.loss_, d = fmin_l_bfgs_b(
            x0=packed_coef_inter,
            func=self._loss_grad_lbfgs,
            maxfun=self.max_iter,
            iprint=iprint,
            pgtol=self.tol,
            args=(X, y, activations, deltas, coef_grads, intercept_grads))

        self._unpack(optimal_parameters)
```
### 18 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 378, End line: 427

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _validate_hyperparameters(self):
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." %
                             self.shuffle)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if (self.learning_rate in ["constant", "invscaling", "adaptive"] and
                self.learning_rate_init <= 0.0):
            raise ValueError("learning_rate_init must be > 0, got %s." %
                             self.learning_rate)
        if self.momentum > 1 or self.momentum < 0:
            raise ValueError("momentum must be >= 0 and <= 1, got %s" %
                             self.momentum)
        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError("nesterovs_momentum must be either True or False,"
                             " got %s." % self.nesterovs_momentum)
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False,"
                             " got %s." % self.early_stopping)
        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)
        if self.beta_1 < 0 or self.beta_1 >= 1:
            raise ValueError("beta_1 must be >= 0 and < 1, got %s" %
                             self.beta_1)
        if self.beta_2 < 0 or self.beta_2 >= 1:
            raise ValueError("beta_2 must be >= 0 and < 1, got %s" %
                             self.beta_2)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be > 0, got %s."
                             % self.n_iter_no_change)

        # raise ValueError if not registered
        supported_activations = ('identity', 'logistic', 'tanh', 'relu')
        if self.activation not in supported_activations:
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s." % (self.activation,
                                                      supported_activations))
        if self.learning_rate not in ["constant", "invscaling", "adaptive"]:
            raise ValueError("learning rate %s is not supported. " %
                             self.learning_rate)
        supported_solvers = _STOCHASTIC_SOLVERS + ["lbfgs"]
        if self.solver not in supported_solvers:
            raise ValueError("The solver %s is not supported. "
                             " Expected one of: %s" %
                             (self.solver, ", ".join(supported_solvers)))
```
### 19 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 575, End line: 604

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            # compute validation score, use that for stopping
            self.validation_scores_.append(self.score(X_val, y_val))

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            # update best parameters
            # use validation_scores_, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ +
                                   self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy()
                                         for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]
```
### 20 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 39, End line: 86

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):
    """Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.18
    """

    @abstractmethod
    def __init__(self, hidden_layer_sizes, activation, solver,
                 alpha, batch_size, learning_rate, learning_rate_init, power_t,
                 max_iter, loss, shuffle, random_state, tol, verbose,
                 warm_start, momentum, nesterovs_momentum, early_stopping,
                 validation_fraction, beta_1, beta_2, epsilon,
                 n_iter_no_change):
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

    def _unpack(self, packed_parameters):
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]
```
### 27 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 1, End line: 36

```python
"""Multi-layer Perceptron
"""

import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.optimize import fmin_l_bfgs_b
import warnings

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..base import is_classifier
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import gen_batches, check_random_state
from ..utils import shuffle
from ..utils import check_array, check_X_y, column_or_1d
from ..exceptions import ConvergenceWarning
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..utils.multiclass import _check_partial_fit_first_call, unique_labels
from ..utils.multiclass import type_of_target


_STOCHASTIC_SOLVERS = ['sgd', 'adam']


def _pack(coefs_, intercepts_):
    """Pack the parameters into a single vector."""
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])
```
### 30 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 624, End line: 647

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    @property
    def partial_fit(self):
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : returns a trained MLP model.
        """
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError("partial_fit is only available for stochastic"
                                 " optimizers. %s is not stochastic."
                                 % self.solver)
        return self._partial_fit

    def _partial_fit(self, X, y):
        return self._fit(X, y, incremental=True)
```
### 32 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 469, End line: 505

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):

        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X, y, random_state=self._random_state,
                test_size=self.validation_fraction,
                stratify=stratify)
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)
        # ... other code
```
### 34 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 179, End line: 254

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads
```
### 37 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 984, End line: 1012

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):

    @property
    def partial_fit(self):
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target values.

        classes : array, shape (n_classes), default None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : returns a trained MLP model.
        """
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError("partial_fit is only available for stochastic"
                                 " optimizer. %s is not stochastic"
                                 % self.solver)
        return self._partial_fit
```
### 43 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 1014, End line: 1042

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):

    def _partial_fit(self, X, y, classes=None):
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith('multilabel'):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        super()._partial_fit(X, y)

        return self

    def predict_log_proba(self, X):
        """Return the log of probability estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        log_y_prob : array-like, shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to log(predict_proba(X))
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)
```
### 50 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 965, End line: 982

```python
class MLPClassifier(BaseMultilayerPerceptron, ClassifierMixin):

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """
        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))
```
### 58 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 256, End line: 293

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        if not is_classifier(self):
            self.out_activation_ = 'identity'
        # Output for multi class
        elif self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        # Output for binary class and multi-label
        else:
            self.out_activation_ = 'logistic'

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf
```
### 59 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 130, End line: 177

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
                         coef_grads, intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector so it can be used
        in lbfgs

        Parameters
        ----------
        packed_coef_inter : array-like
            A vector comprising the flattened coefficients and intercepts.

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads)
        self.n_iter_ += 1
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad
```
### 66 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 295, End line: 308

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def _init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        return coef_init, intercept_init
```
### 112 - sklearn/neural_network/multilayer_perceptron.py:

Start line: 606, End line: 622

```python
class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """
        return self._fit(X, y, incremental=False)
```
