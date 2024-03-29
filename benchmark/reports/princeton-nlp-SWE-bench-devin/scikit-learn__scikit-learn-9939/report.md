# scikit-learn__scikit-learn-9939

| **scikit-learn/scikit-learn** | `f247ad5adfe86b2ee64a4a3db1b496c8bf1c9dff` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 11761 |
| **Any found context length** | 2500 |
| **Avg pos** | 95.0 |
| **Min pos** | 2 |
| **Max pos** | 61 |
| **Top file pos** | 2 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -1101,14 +1101,18 @@ class LogisticRegression(BaseEstimator, LinearClassifierMixin,
     coef_ : array, shape (1, n_features) or (n_classes, n_features)
         Coefficient of the features in the decision function.
 
-        `coef_` is of shape (1, n_features) when the given problem
-        is binary.
+        `coef_` is of shape (1, n_features) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `coef_` corresponds
+        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
 
     intercept_ : array, shape (1,) or (n_classes,)
         Intercept (a.k.a. bias) added to the decision function.
 
         If `fit_intercept` is set to False, the intercept is set to zero.
-        `intercept_` is of shape(1,) when the problem is binary.
+        `intercept_` is of shape (1,) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `intercept_`
+        corresponds to outcome 1 (True) and `-intercept_` corresponds to
+        outcome 0 (False).
 
     n_iter_ : array, shape (n_classes,) or (1, )
         Actual number of iterations for all classes. If binary or multinomial,
@@ -1332,11 +1336,17 @@ def predict_proba(self, X):
         """
         if not hasattr(self, "coef_"):
             raise NotFittedError("Call fit before prediction")
-        calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
-        if calculate_ovr:
+        if self.multi_class == "ovr":
             return super(LogisticRegression, self)._predict_proba_lr(X)
         else:
-            return softmax(self.decision_function(X), copy=False)
+            decision = self.decision_function(X)
+            if decision.ndim == 1:
+                # Workaround for multi_class="multinomial" and binary outcomes
+                # which requires softmax prediction with only a 1D decision.
+                decision_2d = np.c_[-decision, decision]
+            else:
+                decision_2d = decision
+            return softmax(decision_2d, copy=False)
 
     def predict_log_proba(self, X):
         """Log of probability estimates.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/logistic.py | 1104 | 1111 | 16 | 2 | 11761
| sklearn/linear_model/logistic.py | 1335 | 1339 | 61 | 2 | 36756


## Problem Statement

```
Incorrect predictions when fitting a LogisticRegression model on binary outcomes with `multi_class='multinomial'`.
#### Description
Incorrect predictions when fitting a LogisticRegression model on binary outcomes with `multi_class='multinomial'`.
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->

#### Steps/Code to Reproduce
\`\`\`python
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics
    import numpy as np

    # Set up a logistic regression object
    lr = LogisticRegression(C=1000000, multi_class='multinomial',
                            solver='sag', tol=0.0001, warm_start=False,
                            verbose=0)

    # Set independent variable values
    Z = np.array([
       [ 0.        ,  0.        ],
       [ 1.33448632,  0.        ],
       [ 1.48790105, -0.33289528],
       [-0.47953866, -0.61499779],
       [ 1.55548163,  1.14414766],
       [-0.31476657, -1.29024053],
       [-1.40220786, -0.26316645],
       [ 2.227822  , -0.75403668],
       [-0.78170885, -1.66963585],
       [ 2.24057471, -0.74555021],
       [-1.74809665,  2.25340192],
       [-1.74958841,  2.2566389 ],
       [ 2.25984734, -1.75106702],
       [ 0.50598996, -0.77338402],
       [ 1.21968303,  0.57530831],
       [ 1.65370219, -0.36647173],
       [ 0.66569897,  1.77740068],
       [-0.37088553, -0.92379819],
       [-1.17757946, -0.25393047],
       [-1.624227  ,  0.71525192]])
    
    # Set dependant variable values
    Y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 0, 0, 1, 
                  0, 0, 1, 1], dtype=np.int32)

    lr.fit(Z, Y)
    p = lr.predict_proba(Z)
    print(sklearn.metrics.log_loss(Y, p)) # ...

    print(lr.intercept_)
    print(lr.coef_)
\`\`\`
<!--
Example:
\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = ["Help I have a bug" for i in range(1000)]

vectorizer = CountVectorizer(input=docs, analyzer='word')
lda_features = vectorizer.fit_transform(docs)

lda_model = LatentDirichletAllocation(
    n_topics=10,
    learning_method='online',
    evaluate_every=10,
    n_jobs=4,
)
model = lda_model.fit(lda_features)
\`\`\`
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
If we compare against R or using `multi_class='ovr'`, the log loss (which is approximately proportional to the objective function as the regularisation is set to be negligible through the choice of `C`) is incorrect. We expect the log loss to be roughly `0.5922995`

#### Actual Results
The actual log loss when using `multi_class='multinomial'` is `0.61505641264`.

#### Further Information
See the stack exchange question https://stats.stackexchange.com/questions/306886/confusing-behaviour-of-scikit-learn-logistic-regression-multinomial-optimisation?noredirect=1#comment583412_306886 for more information.

The issue it seems is caused in https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/linear_model/logistic.py#L762. In the `multinomial` case even if `classes.size==2` we cannot reduce to a 1D case by throwing away one of the vectors of coefficients (as we can in normal binary logistic regression). This is essentially a difference between softmax (redundancy allowed) and logistic regression.

This can be fixed by commenting out the lines 762 and 763. I am apprehensive however that this may cause some other unknown issues which is why I am positing as a bug.

#### Versions
<!--
Please run the following snippet and paste the output below.
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->
Linux-4.10.0-33-generic-x86_64-with-Ubuntu-16.04-xenial
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
NumPy 1.13.1
SciPy 0.19.1
Scikit-Learn 0.19.0
<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 59| 462 | 462 | 988 | 
| **-> 2 <-** | **2 sklearn/linear_model/logistic.py** | 964 | 1155| 2038 | 2500 | 17480 | 
| 3 | **2 sklearn/linear_model/logistic.py** | 1675 | 1730| 642 | 3142 | 17480 | 
| 4 | **2 sklearn/linear_model/logistic.py** | 1267 | 1308| 426 | 3568 | 17480 | 
| 5 | 3 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 80| 655 | 4223 | 18166 | 
| 6 | **3 sklearn/linear_model/logistic.py** | 1732 | 1787| 542 | 4765 | 18166 | 
| 7 | 4 examples/linear_model/plot_logistic_multinomial.py | 1 | 72| 669 | 5434 | 18863 | 
| 8 | **4 sklearn/linear_model/logistic.py** | 590 | 654| 740 | 6174 | 18863 | 
| 9 | 4 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 61 | 119| 519 | 6693 | 18863 | 
| 10 | **4 sklearn/linear_model/logistic.py** | 704 | 770| 772 | 7465 | 18863 | 
| 11 | **4 sklearn/linear_model/logistic.py** | 656 | 703| 639 | 8104 | 18863 | 
| 12 | **4 sklearn/linear_model/logistic.py** | 243 | 297| 430 | 8534 | 18863 | 
| 13 | 5 sklearn/linear_model/randomized_l1.py | 358 | 387| 296 | 8830 | 24634 | 
| 14 | **5 sklearn/linear_model/logistic.py** | 1 | 37| 215 | 9045 | 24634 | 
| 15 | **5 sklearn/linear_model/logistic.py** | 1362 | 1556| 2114 | 11159 | 24634 | 
| **-> 16 <-** | **5 sklearn/linear_model/logistic.py** | 903 | 1357| 602 | 11761 | 24634 | 
| 17 | **5 sklearn/linear_model/logistic.py** | 1157 | 1175| 202 | 11963 | 24634 | 
| 18 | **5 sklearn/linear_model/logistic.py** | 426 | 447| 247 | 12210 | 24634 | 
| 19 | **5 sklearn/linear_model/logistic.py** | 1558 | 1578| 221 | 12431 | 24634 | 
| 20 | 6 examples/linear_model/plot_logistic.py | 1 | 67| 440 | 12871 | 25082 | 
| 21 | 7 sklearn/multiclass.py | 1 | 63| 489 | 13360 | 31570 | 
| 22 | 7 sklearn/multiclass.py | 375 | 410| 300 | 13660 | 31570 | 
| 23 | 8 examples/linear_model/plot_iris_logistic.py | 1 | 60| 453 | 14113 | 32031 | 
| 24 | 9 examples/linear_model/plot_logistic_path.py | 1 | 56| 286 | 14399 | 32342 | 
| 25 | 10 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 1 | 80| 698 | 15097 | 33094 | 
| 26 | **10 sklearn/linear_model/logistic.py** | 300 | 350| 432 | 15529 | 33094 | 
| 27 | 11 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 142| 488 | 16017 | 34185 | 
| 28 | 12 sklearn/utils/estimator_checks.py | 1200 | 1307| 1123 | 17140 | 51632 | 
| 29 | **12 sklearn/linear_model/logistic.py** | 1341 | 1786| 151 | 17291 | 51632 | 
| 30 | 12 sklearn/linear_model/randomized_l1.py | 390 | 506| 1055 | 18346 | 51632 | 
| 31 | 12 sklearn/linear_model/randomized_l1.py | 507 | 538| 332 | 18678 | 51632 | 
| 32 | 13 sklearn/linear_model/stochastic_gradient.py | 849 | 888| 296 | 18974 | 63773 | 
| 33 | 14 sklearn/naive_bayes.py | 631 | 710| 716 | 19690 | 72126 | 
| 34 | 15 examples/calibration/plot_calibration_multiclass.py | 1 | 80| 764 | 20454 | 74259 | 
| 35 | 16 benchmarks/bench_glmnet.py | 47 | 129| 796 | 21250 | 75346 | 
| 36 | 17 sklearn/ensemble/gradient_boosting.py | 528 | 544| 124 | 21374 | 93599 | 
| 37 | 18 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 80| 760 | 22134 | 94622 | 
| 38 | 19 examples/text/plot_document_classification_20newsgroups.py | 250 | 328| 673 | 22807 | 97179 | 
| 39 | 19 sklearn/multiclass.py | 633 | 705| 713 | 23520 | 97179 | 
| 40 | 19 examples/calibration/plot_calibration_multiclass.py | 122 | 169| 589 | 24109 | 97179 | 
| 41 | 20 examples/ensemble/plot_random_forest_regression_multioutput.py | 1 | 77| 633 | 24742 | 97833 | 
| 42 | 21 examples/classification/plot_lda.py | 41 | 72| 325 | 25067 | 98424 | 
| 43 | 21 sklearn/utils/estimator_checks.py | 1523 | 1565| 449 | 25516 | 98424 | 
| 44 | 22 sklearn/metrics/classification.py | 1737 | 1771| 361 | 25877 | 116794 | 
| 45 | **22 sklearn/linear_model/logistic.py** | 1177 | 1266| 807 | 26684 | 116794 | 
| 46 | 23 sklearn/neural_network/multilayer_perceptron.py | 1019 | 1047| 234 | 26918 | 128437 | 
| 47 | 24 examples/ensemble/plot_adaboost_multiclass.py | 1 | 91| 757 | 27675 | 129460 | 
| 48 | 24 examples/neural_networks/plot_rbm_logistic_classification.py | 1 | 42| 304 | 27979 | 129460 | 
| 49 | 24 sklearn/naive_bayes.py | 840 | 909| 705 | 28684 | 129460 | 
| 50 | 25 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 29224 | 130984 | 
| 51 | 25 sklearn/neural_network/multilayer_perceptron.py | 692 | 897| 1853 | 31077 | 130984 | 
| 52 | 26 examples/linear_model/plot_lasso_model_selection.py | 92 | 156| 503 | 31580 | 132313 | 
| 53 | 26 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 312 | 31892 | 132313 | 
| 54 | 27 sklearn/discriminant_analysis.py | 130 | 248| 1044 | 32936 | 138944 | 
| 55 | **27 sklearn/linear_model/logistic.py** | 402 | 423| 246 | 33182 | 138944 | 
| 56 | 28 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 33555 | 140892 | 
| 57 | 29 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 34122 | 141459 | 
| 58 | 30 examples/linear_model/plot_multi_task_lasso_support.py | 1 | 70| 587 | 34709 | 142071 | 
| 59 | 30 sklearn/multiclass.py | 110 | 130| 137 | 34846 | 142071 | 
| 60 | 31 sklearn/svm/classes.py | 15 | 172| 1635 | 36481 | 152483 | 
| **-> 61 <-** | **31 sklearn/linear_model/logistic.py** | 1310 | 1339| 275 | 36756 | 152483 | 
| 62 | **31 sklearn/linear_model/logistic.py** | 450 | 589| 1370 | 38126 | 152483 | 
| 63 | 32 sklearn/linear_model/__init__.py | 1 | 87| 709 | 38835 | 153192 | 
| 64 | 33 sklearn/svm/base.py | 845 | 907| 648 | 39483 | 161032 | 
| 65 | 33 benchmarks/bench_rcv1_logreg_convergence.py | 6 | 31| 202 | 39685 | 161032 | 
| 66 | 33 examples/linear_model/plot_lasso_model_selection.py | 1 | 78| 691 | 40376 | 161032 | 
| 67 | 33 sklearn/utils/estimator_checks.py | 1393 | 1427| 336 | 40712 | 161032 | 
| 68 | 34 examples/calibration/plot_compare_calibration.py | 1 | 78| 759 | 41471 | 162213 | 
| 69 | 34 sklearn/neural_network/multilayer_perceptron.py | 1076 | 1278| 1831 | 43302 | 162213 | 
| 70 | 35 examples/plot_multilabel.py | 97 | 115| 185 | 43487 | 163317 | 
| 71 | 36 examples/classification/plot_classification_probability.py | 1 | 86| 703 | 44190 | 164045 | 
| 72 | 36 examples/calibration/plot_calibration_multiclass.py | 81 | 121| 753 | 44943 | 164045 | 
| 73 | 37 benchmarks/bench_mnist.py | 85 | 106| 314 | 45257 | 165776 | 
| 74 | 38 sklearn/linear_model/base.py | 331 | 348| 167 | 45424 | 170744 | 
| 75 | 39 examples/applications/plot_out_of_core_classification.py | 188 | 215| 234 | 45658 | 174053 | 
| 76 | **39 sklearn/linear_model/logistic.py** | 1580 | 1674| 786 | 46444 | 174053 | 
| 77 | 39 sklearn/utils/estimator_checks.py | 1458 | 1496| 442 | 46886 | 174053 | 
| 78 | 40 examples/gaussian_process/plot_gpc.py | 1 | 77| 751 | 47637 | 175089 | 
| 79 | 40 sklearn/utils/estimator_checks.py | 1 | 72| 625 | 48262 | 175089 | 
| 80 | 41 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 49021 | 176572 | 
| 81 | 41 sklearn/ensemble/gradient_boosting.py | 624 | 649| 293 | 49314 | 176572 | 
| 82 | 41 sklearn/linear_model/randomized_l1.py | 313 | 335| 239 | 49553 | 176572 | 
| 83 | 41 sklearn/linear_model/randomized_l1.py | 337 | 355| 223 | 49776 | 176572 | 
| 84 | 41 sklearn/utils/estimator_checks.py | 1166 | 1197| 258 | 50034 | 176572 | 
| 85 | 42 sklearn/neural_network/_base.py | 225 | 253| 215 | 50249 | 178179 | 
| 86 | 42 sklearn/linear_model/stochastic_gradient.py | 789 | 808| 281 | 50530 | 178179 | 
| 87 | **42 sklearn/linear_model/logistic.py** | 773 | 902| 1258 | 51788 | 178179 | 
| 88 | 43 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 771 | 52559 | 179099 | 


### Hint

```
Yes, just taking the coef for one class indeed seems incorrect. Is there
any way to adjust the coef of one class (and the intercept) given the other
to get the right probabilities?

> This is essentially a difference between softmax (redundancy allowed) and logistic regression.

Indeed, there is a difference in the way we want to compute `predict_proba`:
1. In OVR-LR, you want a sigmoid: `exp(D(x)) / (exp(D(x)) + 1)`, where `D(x)` is the decision function.
2. In multinomial-LR with `n_classes=2`, you want a softmax `exp(D(x)) / (exp(D(x)) + exp(-D(x))`.

So we **do** expect different results between (1) and (2). 
However, there is indeed a bug and your fix is correct:

In the current code, we incorrectly use the sigmoid on case (2). Removing lines 762 and 763 does solve this problem, but change the API of `self.coef_`, which specifically states `coef_ is of shape (1, n_features) when the given problem is binary`.

Another way to fix it is to change directly `predict_proba`, adding the case binary+multinomial:
\`\`\`py
        if self.multi_class == "ovr":
            return super(LogisticRegression, self)._predict_proba_lr(X)
        elif self.coef_.shape[0] == 1:
            decision = self.decision_function(X)
            return softmax(np.c_[-decision, decision], copy=False)
        else:
            return softmax(self.decision_function(X), copy=False)
\`\`\`
It will also break current behavior of `predict_proba`, but we don't need deprecation if we consider it is a bug.


If it didn't cause too many further issues, I think the first solution would be better i.e. changing the API of `self.coef_` for the `multinomial` case. This is because, say we fit a logistic regression `lr`, then upon inspecting the `lr.coef_ ` and `lr.intercept_` objects, it is clear what model is being used. 

I also believe anyone using `multinomial` for a binary case (as I was) is doing it as part of some more general functionality and will also be fitting non-binary models depending on their data. If they want to access the parameters of the models (as I was) `.intercept_` and `.coef_` their generalisation will be broken in the binary case if only `predict_proba` is changed.
We already break the generalisation elsewhere, as in `decision_function` and in the `coef_` shape for other multiclass methods. I think maintaining internal consistency here might be more important than some abstract concern that "generalisation will be broken". I think we should choose modifying `predict_proba`. This also makes it clear that the multinomial case does not suddenly introduce more free parameters.
I agree it would make more sense to have `coef_.shape = (n_classes, n_features)` even when `n_classes = 2`, to have more consistency and avoid special cases.

However, it is a valid argument also for the OVR case (actually, it is nice to have the same `coef_` API for both multinomial and OVR cases). Does that mean we should change the `coef_`  API in all cases? It is an important API change which will break a lot of user code, and which might not be consistent with the rest of scikit-learn...
Another option could be to always use the `ovr` method in the binary case even when `multiclass` is set to `multinomial`. This would avoid the case of models having exactly the same coefficients but predicting different values due to have different `multiclass` parameters. As previously mentioned, if `predict_proba` gets changed, the `multinomial` prediction would be particularly confusing if someone just looks at the 1D coefficients `coef_` (I think the `ovr` case is the intuitive one).

I believe by doing this, the only code that would get broken would be anyone who already knew about the bug and had coded their own workaround.

Note: If we do this, it is not returning the actual correct parameters with regards to the regularisation, despite the fact the solutions will be identical in terms of prediction. This may make it a no go.
Changing the binary coef_ in the general case is just not going to happen.
If you want to fix a bug, fix a bug...

On 10 October 2017 at 00:05, Tom Dupré la Tour <notifications@github.com>
wrote:

> I agree it would make more sense to have coef_.shape = (n_classes,
> n_features) even when n_classes = 2, to have more consistency and avoid
> special cases.
>
> However, it is a valid argument also for the OVR case (actually, it is
> nice to have the same coef_ API for both multinomial and OVR cases). Does
> that mean we should change the coef_ API in all cases? It is an important
> API changes which will break a lot of user code, and which might not be
> consistent with the rest of scikit-learn.
>
> —
> You are receiving this because you commented.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/9889#issuecomment-335150971>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz6-kpIPulfN1z6KQYbqEQP2bRd3pPks5sqhkPgaJpZM4PyMNd>
> .
>

Are the learnt probabilities then equivalent if it changes to ovr for 2
classes? Seems a reasonable idea to me.

Thinking about it, it works with no regularisation but when regularisation is involved we should expect slightly different results for `ovr` and `multinomial`.

Maybe then just change `predict_proba` as suggested and a warning message when fitting a binary model with `multinomial`.
why the warning? what would it say?

The issue is that the values of `coef_` do not intuitively describe the model in the binary case using `multinomial`. If someone fits a binary logistic regression and receives back a 1D vector of coefficients (`W` say for convenience), I would assume that they will think the predicted probability of a new observation `X`, is given by

    exp(dot(W,X)) / (1 + exp(dot(W,X)))

This is true in the `ovr` case only. In the `multinomial` case, it is actually given by

    exp(dot(W,X)) / (exp(dot(-W,X)) + exp(dot(W,X)))

I believe this would surprise and cause errors for many people upon receiving a 1D vector of coefficients `W` so I think they should be warned about it. In fact I wouldn't be surprised if people currently using the logistic regression coefficients in the `multinomial`, binary outcome case, have bugs in their code.

I would suggest a warning message when `.fit` is called with `multinomial`, when binary outcomes are detected. Something along the lines of (this can probably be made more concise):

    Fitting a binary model with multi_class=multinomial. The returned `coef_` and `intercept_` values form the coefficients for outcome 1 (True), use `-coef_` and `-intercept` to form the coefficients for outcome 0 (False).
I think it would be excessive noise to warn such upon fit. why not just
amend the coef_ description? most users will not be manually making
probabilistic interpretations of coef_ in any case, and we can't in general
stop users misinterpreting things on the basis of assumption rather than
reading the docs...

Fair enough. My only argument to the contrary would be that using `multinomial` for binary classification is a fairly uncommon thing to do, so the warning would be infrequent.

However I agree in this case that if a user is only receiving a 1D vector of coefficients (i.e. it is not in the general form as for dimensions > 2), then they should be checking the documentation for exactly what this means, so amending the `coef_` description should suffice.
So to sum-up, we need to:
 - update `predict_proba` as described in https://github.com/scikit-learn/scikit-learn/issues/9889#issuecomment-335129554
- update `coef_`'s docstring
- add a test and a bugfix entry in `whats_new`

Do you want to do it @rwolst ?
Sure, I'll do it.
Thanks
```

## Patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -1101,14 +1101,18 @@ class LogisticRegression(BaseEstimator, LinearClassifierMixin,
     coef_ : array, shape (1, n_features) or (n_classes, n_features)
         Coefficient of the features in the decision function.
 
-        `coef_` is of shape (1, n_features) when the given problem
-        is binary.
+        `coef_` is of shape (1, n_features) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `coef_` corresponds
+        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
 
     intercept_ : array, shape (1,) or (n_classes,)
         Intercept (a.k.a. bias) added to the decision function.
 
         If `fit_intercept` is set to False, the intercept is set to zero.
-        `intercept_` is of shape(1,) when the problem is binary.
+        `intercept_` is of shape (1,) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `intercept_`
+        corresponds to outcome 1 (True) and `-intercept_` corresponds to
+        outcome 0 (False).
 
     n_iter_ : array, shape (n_classes,) or (1, )
         Actual number of iterations for all classes. If binary or multinomial,
@@ -1332,11 +1336,17 @@ def predict_proba(self, X):
         """
         if not hasattr(self, "coef_"):
             raise NotFittedError("Call fit before prediction")
-        calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
-        if calculate_ovr:
+        if self.multi_class == "ovr":
             return super(LogisticRegression, self)._predict_proba_lr(X)
         else:
-            return softmax(self.decision_function(X), copy=False)
+            decision = self.decision_function(X)
+            if decision.ndim == 1:
+                # Workaround for multi_class="multinomial" and binary outcomes
+                # which requires softmax prediction with only a 1D decision.
+                decision_2d = np.c_[-decision, decision]
+            else:
+                decision_2d = decision
+            return softmax(decision_2d, copy=False)
 
     def predict_log_proba(self, X):
         """Log of probability estimates.

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_logistic.py b/sklearn/linear_model/tests/test_logistic.py
--- a/sklearn/linear_model/tests/test_logistic.py
+++ b/sklearn/linear_model/tests/test_logistic.py
@@ -198,6 +198,23 @@ def test_multinomial_binary():
         assert_greater(np.mean(pred == target), .9)
 
 
+def test_multinomial_binary_probabilities():
+    # Test multinomial LR gives expected probabilities based on the
+    # decision function, for a binary problem.
+    X, y = make_classification()
+    clf = LogisticRegression(multi_class='multinomial', solver='saga')
+    clf.fit(X, y)
+
+    decision = clf.decision_function(X)
+    proba = clf.predict_proba(X)
+
+    expected_proba_class_1 = (np.exp(decision) /
+                              (np.exp(decision) + np.exp(-decision)))
+    expected_proba = np.c_[1-expected_proba_class_1, expected_proba_class_1]
+
+    assert_almost_equal(proba, expected_proba)
+
+
 def test_sparsify():
     # Test sparsify and densify members.
     n_samples, n_features = iris.data.shape

```


## Code snippets

### 1 - examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py:

Start line: 1, End line: 59

```python
"""
=====================================================
Multiclass sparse logisitic regression on newgroups20
=====================================================

Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression
to classify documents from the newgroups20 dataset. Multinomial logistic
regression yields more accurate results and is faster to train on the larger
scale dataset.

Here we use the l1 sparsity that trims the weights of not informative
features to zero. This is good if the goal is to extract the strongly
discriminative vocabulary of each class. If the goal is to get the best
predictive accuracy, it is better to use the non sparsity-inducing l2 penalty
instead.

A more traditional (and possibly better) way to predict on a sparse subset of
input features would be to use univariate feature selection followed by a
traditional (l2-penalised) logistic regression model.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print(__doc__)

t0 = time.clock()

# We use SAGA solver
solver = 'saga'

# Turn down for faster run time
n_samples = 10000

# Memorized fetch_rcv1 for faster access
dataset = fetch_20newsgroups_vectorized('all')
X = dataset.data
y = dataset.target
X = X[:n_samples]
y = y[:n_samples]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, n_features, n_classes))

models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 3]},
          'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}}
```
### 2 - sklearn/linear_model/logistic.py:

Start line: 964, End line: 1155

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag' and 'lbfgs' solvers. It can handle
    both dense and sparse input. Use C-ordered arrays or CSR matrices
    containing 64-bit floats for optimal performance; any other input format
    will be converted (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation. The 'liblinear' solver supports both L1 and L2
    regularization, with a dual formulation only for the L2 penalty.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'liblinear'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
            'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
            handle multinomial loss; 'liblinear' is limited to one-versus-rest
            schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
            'liblinear' and 'saga' handle L1 penalty.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for liblinear
        solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    Attributes
    ----------

    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem
        is binary.

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.

    n_iter_ : array, shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

    See also
    --------
    SGDClassifier : incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    sklearn.svm.LinearSVC : learns SVM models using the same algorithm.
    LogisticRegressionCV : Logistic regression with built-in cross validation

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
        SAGA: A Fast Incremental Gradient Method With Support
        for Non-Strongly Convex Composite Objectives
        https://arxiv.org/abs/1407.0202

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    """
```
### 3 - sklearn/linear_model/logistic.py:

Start line: 1675, End line: 1730

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, train, test, pos_class=label, Cs=self.Cs,
                      fit_intercept=self.fit_intercept, penalty=self.penalty,
                      dual=self.dual, solver=self.solver, tol=self.tol,
                      max_iter=self.max_iter, verbose=self.verbose,
                      class_weight=class_weight, scoring=self.scoring,
                      multi_class=self.multi_class,
                      intercept_scaling=self.intercept_scaling,
                      random_state=self.random_state,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight
                      )
            for label in iter_encoded_labels
            for train, test in folds)

        if self.multi_class == 'multinomial':
            multi_coefs_paths, Cs, multi_scores, n_iter_ = zip(*fold_coefs_)
            multi_coefs_paths = np.asarray(multi_coefs_paths)
            multi_scores = np.asarray(multi_scores)

            # This is just to maintain API similarity between the ovr and
            # multinomial option.
            # Coefs_paths in now n_folds X len(Cs) X n_classes X n_features
            # we need it to be n_classes X len(Cs) X n_folds X n_features
            # to be similar to "ovr".
            coefs_paths = np.rollaxis(multi_coefs_paths, 2, 0)

            # Multinomial has a true score across all labels. Hence the
            # shape is n_folds X len(Cs). We need to repeat this score
            # across all labels for API similarity.
            scores = np.tile(multi_scores, (n_classes, 1, 1))
            self.Cs_ = Cs[0]
            self.n_iter_ = np.reshape(n_iter_, (1, len(folds),
                                                len(self.Cs_)))

        else:
            coefs_paths, Cs, scores, n_iter_ = zip(*fold_coefs_)
            self.Cs_ = Cs[0]
            coefs_paths = np.reshape(coefs_paths, (n_classes, len(folds),
                                                   len(self.Cs_), -1))
            self.n_iter_ = np.reshape(n_iter_, (n_classes, len(folds),
                                                len(self.Cs_)))

        self.coefs_paths_ = dict(zip(classes, coefs_paths))
        scores = np.reshape(scores, (n_classes, len(folds), -1))
        self.scores_ = dict(zip(classes, scores))

        self.C_ = list()
        self.coef_ = np.empty((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)

        # hack to iterate only once for multinomial case.
        if self.multi_class == 'multinomial':
            scores = multi_scores
            coefs_paths = multi_coefs_paths
        # ... other code
```
### 4 - sklearn/linear_model/logistic.py:

Start line: 1267, End line: 1308

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code
        if self.multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ['sag', 'saga']:
            backend = 'threading'
        else:
            backend = 'multiprocessing'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol,
                      verbose=self.verbose, solver=self.solver,
                      multi_class=self.multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      penalty=self.penalty,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if self.multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self
```
### 5 - examples/linear_model/plot_sparse_logistic_regression_mnist.py:

Start line: 1, End line: 80

```python
"""
=====================================================
MNIST classfification using multinomial logistic + L1
=====================================================

Here we fit a multinomial logistic regression with L1 penalty on a subset of
the MNIST digits classification task. We use the SAGA algorithm for this
purpose: this a solver that is fast when the number of samples is significantly
larger than the number of features and is able to finely optimize non-smooth
objective functions which is the case with the l1-penalty. Test accuracy
reaches > 0.8, while weight vectors remains *sparse* and therefore more easily
*interpretable*.

Note that this accuracy of this l1-penalized linear model is significantly
below what can be reached by an l2-penalized linear model or a non-linear
multi-layer perceptron model on this dataset.

"""
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

print(__doc__)
t0 = time.time()
train_samples = 5000

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```
### 6 - sklearn/linear_model/logistic.py:

Start line: 1732, End line: 1787

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code

        for index, (cls, encoded_label) in enumerate(
                zip(iter_classes, iter_encoded_labels)):

            if self.multi_class == 'ovr':
                # The scores_ / coefs_paths_ dict have unencoded class
                # labels as their keys
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]

            if self.refit:
                best_index = scores.sum(axis=0).argmax()

                C_ = self.Cs_[best_index]
                self.C_.append(C_)
                if self.multi_class == 'multinomial':
                    coef_init = np.mean(coefs_paths[:, best_index, :, :],
                                        axis=0)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)

                # Note that y is label encoded and hence pos_class must be
                # the encoded label / None (for 'multinomial')
                w, _, _ = logistic_regression_path(
                    X, y, pos_class=encoded_label, Cs=[C_], solver=self.solver,
                    fit_intercept=self.fit_intercept, coef=coef_init,
                    max_iter=self.max_iter, tol=self.tol,
                    penalty=self.penalty,
                    class_weight=class_weight,
                    multi_class=self.multi_class,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    check_input=False, max_squared_sum=max_squared_sum,
                    sample_weight=sample_weight)
                w = w[0]

            else:
                # Take the best scores across every fold and the average of all
                # coefficients corresponding to the best scores.
                best_indices = np.argmax(scores, axis=1)
                w = np.mean([coefs_paths[i][best_indices[i]]
                             for i in range(len(folds))], axis=0)
                self.C_.append(np.mean(self.Cs_[best_indices]))

            if self.multi_class == 'multinomial':
                self.C_ = np.tile(self.C_, n_classes)
                self.coef_ = w[:, :X.shape[1]]
                if self.fit_intercept:
                    self.intercept_ = w[:, -1]
            else:
                self.coef_[index] = w[: X.shape[1]]
                if self.fit_intercept:
                    self.intercept_[index] = w[-1]

        self.C_ = np.asarray(self.C_)
        return self
```
### 7 - examples/linear_model/plot_logistic_multinomial.py:

Start line: 1, End line: 72

```python
"""
====================================================
Plot multinomial and One-vs-Rest Logistic Regression
====================================================

Plot decision surface of multinomial and One-vs-Rest Logistic Regression.
The hyperplanes corresponding to the three One-vs-Rest (OVR) classifiers
are represented by the dashed lines.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# make 3-class dataset for classification
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)

for multi_class in ('multinomial', 'ovr'):
    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class=multi_class).fit(X, y)

    # print the training scores
    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    plt.axis('tight')

    # Plot also the training points
    colors = "bry"
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
                    edgecolor='black', s=20)

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                 ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)

plt.show()
```
### 8 - sklearn/linear_model/logistic.py:

Start line: 590, End line: 654

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    _check_solver_option(solver, multi_class, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)

    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == 'multinomial':
        class_weight_ = compute_class_weight(class_weight, classes, y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver not in ['sag', 'saga']:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      order='F', dtype=X.dtype)
    # ... other code
```
### 9 - examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py:

Start line: 61, End line: 119

```python
for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params['iters']:
        print('[model=%s, solver=%s] Number of epochs: %s' %
              (model_params['name'], solver, this_max_iter))
        lr = LogisticRegression(solver=solver,
                                multi_class=model,
                                C=1,
                                penalty='l1',
                                fit_intercept=True,
                                max_iter=this_max_iter,
                                random_state=42,
                                )
        t1 = time.clock()
        lr.fit(X_train, y_train)
        train_time = time.clock() - t1

        y_pred = lr.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        density = np.mean(lr.coef_ != 0, axis=1) * 100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]['times'] = times
    models[model]['densities'] = densities
    models[model]['accuracies'] = accuracies
    print('Test accuracy for model %s: %.4f' % (model, accuracies[-1]))
    print('%% non-zero coefficients for model %s, '
          'per class:\n %s' % (model, densities[-1]))
    print('Run time (%i epochs) for model %s:'
          '%.2f' % (model_params['iters'][-1], model, times[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]['name']
    times = models[model]['times']
    accuracies = models[model]['accuracies']
    ax.plot(times, accuracies, marker='o',
            label='Model: %s' % name)
    ax.set_xlabel('Train time (s)')
    ax.set_ylabel('Test accuracy')
ax.legend()
fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
             'Dataset %s' % '20newsgroups')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = time.clock() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```
### 10 - sklearn/linear_model/logistic.py:

Start line: 704, End line: 770

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    # ... other code
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            try:
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            except TypeError:
                # old scipy doesn't have maxiter
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, target, 1. / C, sample_weight),
                    iprint=(verbose > 0) - 1, pgtol=tol)
            if info["warnflag"] == 1 and verbose > 0:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.")
            try:
                n_iter_i = info['nit'] - 1
            except:
                n_iter_i = info['funcalls'] - 1
        elif solver == 'newton-cg':
            args = (X, target, 1. / C, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(np.float64)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            else:
                alpha = 1. / C
                beta = 0.
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            multi_w0 = np.reshape(w0, (classes.size, -1))
            if classes.size == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0)
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter
```
### 11 - sklearn/linear_model/logistic.py:

Start line: 656, End line: 703

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    # ... other code

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))
            w0[:, :coef.shape[1]] = coef

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            w0 = w0.ravel()
        target = Y_multi
        if solver == 'lbfgs':
            func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            func = lambda x, *args: _multinomial_loss(x, *args)[0]
            grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
            hess = _multinomial_grad_hess
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            func = _logistic_loss_and_grad
        elif solver == 'newton-cg':
            func = _logistic_loss
            grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
            hess = _logistic_grad_hess
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    # ... other code
```
### 12 - sklearn/linear_model/logistic.py:

Start line: 243, End line: 297

```python
def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w
```
### 14 - sklearn/linear_model/logistic.py:

Start line: 1, End line: 37

```python
"""
Logistic Regression
"""

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit

from .base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from .sag import sag_solver
from ..preprocessing import LabelEncoder, LabelBinarizer
from ..svm.base import _fit_liblinear
from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_random_state
from ..utils.extmath import (log_logistic, safe_sparse_dot, softmax,
                             squared_norm)
from ..utils.extmath import row_norms
from ..utils.fixes import logsumexp
from ..utils.optimize import newton_cg
from ..utils.validation import check_X_y
from ..exceptions import NotFittedError
from ..utils.multiclass import check_classification_targets
from ..externals.joblib import Parallel, delayed
from ..model_selection import check_cv
from ..externals import six
from ..metrics import get_scorer
```
### 15 - sklearn/linear_model/logistic.py:

Start line: 1362, End line: 1556

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):
    """Logistic Regression CV (aka logit, MaxEnt) classifier.

    This class implements logistic regression using liblinear, newton-cg, sag
    of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
    regularization with primal formulation. The liblinear solver supports both
    L1 and L2 regularization, with a dual formulation only for the L2 penalty.

    For the grid of Cs values (that are set by default to be ten values in
    a logarithmic scale between 1e-4 and 1e4), the best hyperparameter is
    selected by the cross-validator StratifiedKFold, but it can be changed
    using the cv parameter. In the case of newton-cg and lbfgs solvers,
    we warm start along the path i.e guess the initial coefficients of the
    present fit to be the coefficients got after convergence in the previous
    fit, so it is supposed to be faster for high-dimensional dense data.

    For a multiclass problem, the hyperparameters for each class are computed
    using the best scores got by doing a one-vs-rest in parallel across all
    folds and classes. Hence this is not the true multinomial loss.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    Cs : list of floats | int
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    cv : integer or cross-validation generator
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    scoring : string, callable, or None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is 'accuracy'.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'lbfgs'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
            'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
            handle multinomial loss; 'liblinear' is limited to one-versus-rest
            schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
            'liblinear' and 'saga' handle L1 penalty.
        - 'liblinear' might be slower in LogisticRegressionCV because it does
            not handle warm-starting.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can preprocess the data
        with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    tol : float, optional
        Tolerance for stopping criteria.

    max_iter : int, optional
        Maximum number of iterations of the optimization algorithm.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           class_weight == 'balanced'

    n_jobs : int, optional
        Number of CPU cores used during the cross-validation loop. If given
        a value of -1, all cores are used.

    verbose : int
        For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
        positive number for verbosity.

    refit : bool
        If set to True, the scores are averaged across all folds, and the
        coefs and the C that corresponds to the best score is taken, and a
        final refit is done using these parameters.
        Otherwise the coefs, intercepts and C that correspond to the
        best scores across folds are averaged.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'newton-cg',
        'sag', 'saga' and 'lbfgs' solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem
        is binary.

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.

    Cs_ : array
        Array of C i.e. inverse of regularization parameter values used
        for cross-validation.

    coefs_paths_ : array, shape ``(n_folds, len(Cs_), n_features)`` or \
                   ``(n_folds, len(Cs_), n_features + 1)``
        dict with classes as the keys, and the path of coefficients obtained
        during cross-validating across each fold and then across each Cs
        after doing an OvR for the corresponding class as values.
        If the 'multi_class' option is set to 'multinomial', then
        the coefs_paths are the coefficients corresponding to each class.
        Each dict value has shape ``(n_folds, len(Cs_), n_features)`` or
        ``(n_folds, len(Cs_), n_features + 1)`` depending on whether the
        intercept is fit or not.

    scores_ : dict
        dict with classes as the keys, and the values as the
        grid of scores obtained during cross-validating each fold, after doing
        an OvR for the corresponding class. If the 'multi_class' option
        given is 'multinomial' then the same scores are repeated across
        all classes, since this is the multinomial class.
        Each dict value has shape (n_folds, len(Cs))

    C_ : array, shape (n_classes,) or (n_classes - 1,)
        Array of C that maps to the best scores across every class. If refit is
        set to False, then for each class, the best C is the average of the
        C's that correspond to the best scores for each fold.
        `C_` is of shape(n_classes,) when the problem is binary.

    n_iter_ : array, shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
        Actual number of iterations for all classes, folds and Cs.
        In the binary or multinomial cases, the first dimension is equal to 1.

    See also
    --------
    LogisticRegression

    """
```
### 16 - sklearn/linear_model/logistic.py:

Start line: 903, End line: 1357

```python
def _log_reg_scoring_path(X, y, train, test, pos_class=None, Cs=10,
                          scoring=None, fit_intercept=False,
                          max_iter=100, tol=1e-4, class_weight=None,
                          verbose=0, solver='lbfgs', penalty='l2',
                          dual=False, intercept_scaling=1.,
                          multi_class='ovr', random_state=None,
                          max_squared_sum=None, sample_weight=None):
    _check_solver_option(solver, multi_class, penalty, dual)

    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y, sample_weight)

        sample_weight = sample_weight[train]

    coefs, Cs, n_iter = logistic_regression_path(
        X_train, y_train, Cs=Cs, fit_intercept=fit_intercept,
        solver=solver, max_iter=max_iter, class_weight=class_weight,
        pos_class=pos_class, multi_class=multi_class,
        tol=tol, verbose=verbose, dual=dual, penalty=penalty,
        intercept_scaling=intercept_scaling, random_state=random_state,
        check_input=False, max_squared_sum=max_squared_sum,
        sample_weight=sample_weight)

    log_reg = LogisticRegression(fit_intercept=fit_intercept)

    # The score method of Logistic Regression has a classes_ attribute.
    if multi_class == 'ovr':
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == 'multinomial':
        log_reg.classes_ = np.unique(y_train)
    else:
        raise ValueError("multi_class should be either multinomial or ovr, "
                         "got %d" % multi_class)

    if pos_class is not None:
        mask = (y_test == pos_class)
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.

    scores = list()

    if isinstance(scoring, six.string_types):
        scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == 'ovr':
            w = w[np.newaxis, :]
        if fit_intercept:
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            log_reg.coef_ = w
            log_reg.intercept_ = 0.

        if scoring is None:
            scores.append(log_reg.score(X_test, y_test))
        else:
            scores.append(scoring(log_reg, X_test, y_test))
    return coefs, Cs, np.array(scores), n_iter


class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
```
### 17 - sklearn/linear_model/logistic.py:

Start line: 1157, End line: 1175

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
```
### 18 - sklearn/linear_model/logistic.py:

Start line: 426, End line: 447

```python
def _check_solver_option(solver, multi_class, penalty, dual):
    if solver not in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
        raise ValueError("Logistic Regression supports only liblinear, "
                         "newton-cg, lbfgs, sag and saga solvers, got %s"
                         % solver)

    if multi_class not in ['multinomial', 'ovr']:
        raise ValueError("multi_class should be either multinomial or "
                         "ovr, got %s" % multi_class)

    if multi_class == 'multinomial' and solver == 'liblinear':
        raise ValueError("Solver %s does not support "
                         "a multinomial backend." % solver)

    if solver not in ['liblinear', 'saga']:
        if penalty != 'l2':
            raise ValueError("Solver %s supports only l2 penalties, "
                             "got %s penalty." % (solver, penalty))
    if solver != 'liblinear':
        if dual:
            raise ValueError("Solver %s supports only "
                             "dual=False, got dual=%s" % (solver, dual))
```
### 19 - sklearn/linear_model/logistic.py:

Start line: 1558, End line: 1578

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False,
                 penalty='l2', scoring=None, solver='lbfgs', tol=1e-4,
                 max_iter=100, class_weight=None, n_jobs=1, verbose=0,
                 refit=True, intercept_scaling=1., multi_class='ovr',
                 random_state=None):
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
```
### 26 - sklearn/linear_model/logistic.py:

Start line: 300, End line: 350

```python
def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p
```
### 29 - sklearn/linear_model/logistic.py:

Start line: 1341, End line: 1786

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def predict_log_proba(self, X):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):
```
### 45 - sklearn/linear_model/logistic.py:

Start line: 1177, End line: 1266

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        if self.solver in ['newton-cg']:
            _dtype = [np.float64, np.float32]
        else:
            _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype,
                         order="C")
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        _check_solver_option(self.solver, self.multi_class, self.penalty,
                             self.dual)

        if self.solver == 'liblinear':
            if self.n_jobs != 1:
                warnings.warn("'n_jobs' > 1 does not have any effect when"
                              " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                              " = {}.".format(self.n_jobs))
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol, self.random_state,
                sample_weight=sample_weight)
            self.n_iter_ = np.array([n_iter_])
            return self

        if self.solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        # ... other code
```
### 55 - sklearn/linear_model/logistic.py:

Start line: 402, End line: 423

```python
def _multinomial_grad_hess(w, X, Y, alpha, sample_weight):
    # ... other code
    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        r_yhat *= sample_weight
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()

    return grad, hessp
```
### 61 - sklearn/linear_model/logistic.py:

Start line: 1310, End line: 1339

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")
        calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
        if calculate_ovr:
            return super(LogisticRegression, self)._predict_proba_lr(X)
        else:
            return softmax(self.decision_function(X), copy=False)
```
### 62 - sklearn/linear_model/logistic.py:

Start line: 450, End line: 589

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Input data, target values.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Numerical solver to use.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'lbfgs' and
        'newton-cg' solvers.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    # ... other code
```
### 76 - sklearn/linear_model/logistic.py:

Start line: 1580, End line: 1674

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """
        _check_solver_option(self.solver, self.multi_class, self.penalty,
                             self.dual)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        check_classification_targets(y)

        class_weight = self.class_weight

        # Encode for string labels
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        if isinstance(class_weight, dict):
            class_weight = dict((label_encoder.transform([cls])[0], v)
                                for cls, v in class_weight.items())

        # The original class labels
        classes = self.classes_ = label_encoder.classes_
        encoded_labels = label_encoder.transform(label_encoder.classes_)

        if self.solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        # init cross-validation generator
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))

        # Use the label encoded classes
        n_classes = len(encoded_labels)

        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes[0])

        if n_classes == 2:
            # OvR in case of binary problems is as good as fitting
            # the higher label
            n_classes = 1
            encoded_labels = encoded_labels[1:]
            classes = classes[1:]

        # We need this hack to iterate only once over labels, in the case of
        # multi_class = multinomial, without changing the value of the labels.
        if self.multi_class == 'multinomial':
            iter_encoded_labels = iter_classes = [None]
        else:
            iter_encoded_labels = encoded_labels
            iter_classes = classes

        # compute the class weights for the entire dataset y
        if class_weight == "balanced":
            class_weight = compute_class_weight(class_weight,
                                                np.arange(len(self.classes_)),
                                                y)
            class_weight = dict(enumerate(class_weight))

        path_func = delayed(_log_reg_scoring_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ['sag', 'saga']:
            backend = 'threading'
        else:
            backend = 'multiprocessing'
        # ... other code
```
### 87 - sklearn/linear_model/logistic.py:

Start line: 773, End line: 902

```python
# helper function for LogisticCV
def _log_reg_scoring_path(X, y, train, test, pos_class=None, Cs=10,
                          scoring=None, fit_intercept=False,
                          max_iter=100, tol=1e-4, class_weight=None,
                          verbose=0, solver='lbfgs', penalty='l2',
                          dual=False, intercept_scaling=1.,
                          multi_class='ovr', random_state=None,
                          max_squared_sum=None, sample_weight=None):
    """Computes scores across logistic_regression_path

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target labels.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : list of floats | int
        Each of the values in Cs describes the inverse of
        regularization strength. If Cs is as an int, then a grid of Cs
        values are chosen in a logarithmic scale between 1e-4 and 1e4.
        If not provided, then a fixed set of values for Cs are used.

    scoring : callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is accuracy_score.

    fit_intercept : bool
        If False, then the bias term is set to zero. Else the last
        term of each coef_ gives us the intercept.

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Tolerance for stopping criteria.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
        Decides which solver to use.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for
        liblinear solver.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' and
        'liblinear'.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    scores : ndarray, shape (n_cs,)
        Scores obtained for each Cs.

    n_iter : array, shape(n_cs,)
        Actual number of iteration for each Cs.
    """
    # ... other code
```
