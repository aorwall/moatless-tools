# scikit-learn__scikit-learn-25363

| **scikit-learn/scikit-learn** | `cfd428afc5b6e25bbbe4bc92067f857fa9658442` |
| ---- | ---- |
| **No of patches** | 31 |
| **All found context length** | 2027 |
| **Any found context length** | 811 |
| **Avg pos** | 5.935483870967742 |
| **Min pos** | 2 |
| **Max pos** | 33 |
| **Top file pos** | 2 |
| **Missing snippets** | 55 |
| **Missing patch files** | 23 |


## Expected patch

```diff
diff --git a/benchmarks/bench_saga.py b/benchmarks/bench_saga.py
--- a/benchmarks/bench_saga.py
+++ b/benchmarks/bench_saga.py
@@ -7,8 +7,7 @@
 import time
 import os
 
-from joblib import Parallel
-from sklearn.utils.fixes import delayed
+from sklearn.utils.parallel import delayed, Parallel
 import matplotlib.pyplot as plt
 import numpy as np
 
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -14,7 +14,6 @@
 
 from math import log
 import numpy as np
-from joblib import Parallel
 
 from scipy.special import expit
 from scipy.special import xlogy
@@ -36,7 +35,7 @@
 )
 
 from .utils.multiclass import check_classification_targets
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .utils._param_validation import StrOptions, HasMethods, Hidden
 from .utils.validation import (
     _check_fit_params,
diff --git a/sklearn/cluster/_mean_shift.py b/sklearn/cluster/_mean_shift.py
--- a/sklearn/cluster/_mean_shift.py
+++ b/sklearn/cluster/_mean_shift.py
@@ -16,13 +16,12 @@
 
 import numpy as np
 import warnings
-from joblib import Parallel
 from numbers import Integral, Real
 
 from collections import defaultdict
 from ..utils._param_validation import Interval, validate_params
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils import check_random_state, gen_batches, check_array
 from ..base import BaseEstimator, ClusterMixin
 from ..neighbors import NearestNeighbors
diff --git a/sklearn/compose/_column_transformer.py b/sklearn/compose/_column_transformer.py
--- a/sklearn/compose/_column_transformer.py
+++ b/sklearn/compose/_column_transformer.py
@@ -12,7 +12,6 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel
 
 from ..base import clone, TransformerMixin
 from ..utils._estimator_html_repr import _VisualBlock
@@ -26,7 +25,7 @@
 from ..utils import check_pandas_support
 from ..utils.metaestimators import _BaseComposition
 from ..utils.validation import check_array, check_is_fitted, _check_feature_names_in
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 __all__ = ["ColumnTransformer", "make_column_transformer", "make_column_selector"]
diff --git a/sklearn/covariance/_graph_lasso.py b/sklearn/covariance/_graph_lasso.py
--- a/sklearn/covariance/_graph_lasso.py
+++ b/sklearn/covariance/_graph_lasso.py
@@ -13,7 +13,6 @@
 from numbers import Integral, Real
 import numpy as np
 from scipy import linalg
-from joblib import Parallel
 
 from . import empirical_covariance, EmpiricalCovariance, log_likelihood
 
@@ -23,7 +22,7 @@
     check_random_state,
     check_scalar,
 )
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Interval, StrOptions
 
 # mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
diff --git a/sklearn/decomposition/_dict_learning.py b/sklearn/decomposition/_dict_learning.py
--- a/sklearn/decomposition/_dict_learning.py
+++ b/sklearn/decomposition/_dict_learning.py
@@ -13,7 +13,7 @@
 
 import numpy as np
 from scipy import linalg
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
 from ..utils import check_array, check_random_state, gen_even_slices, gen_batches
@@ -21,7 +21,7 @@
 from ..utils._param_validation import validate_params
 from ..utils.extmath import randomized_svd, row_norms, svd_flip
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars
 
 
diff --git a/sklearn/decomposition/_lda.py b/sklearn/decomposition/_lda.py
--- a/sklearn/decomposition/_lda.py
+++ b/sklearn/decomposition/_lda.py
@@ -15,13 +15,13 @@
 import numpy as np
 import scipy.sparse as sp
 from scipy.special import gammaln, logsumexp
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
 from ..utils import check_random_state, gen_batches, gen_even_slices
 from ..utils.validation import check_non_negative
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Interval, StrOptions
 
 from ._online_lda_fast import (
diff --git a/sklearn/ensemble/_bagging.py b/sklearn/ensemble/_bagging.py
--- a/sklearn/ensemble/_bagging.py
+++ b/sklearn/ensemble/_bagging.py
@@ -12,8 +12,6 @@
 from warnings import warn
 from functools import partial
 
-from joblib import Parallel
-
 from ._base import BaseEnsemble, _partition_estimators
 from ..base import ClassifierMixin, RegressorMixin
 from ..metrics import r2_score, accuracy_score
@@ -25,7 +23,7 @@
 from ..utils.random import sample_without_replacement
 from ..utils._param_validation import Interval, HasMethods, StrOptions
 from ..utils.validation import has_fit_parameter, check_is_fitted, _check_sample_weight
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 __all__ = ["BaggingClassifier", "BaggingRegressor"]
diff --git a/sklearn/ensemble/_forest.py b/sklearn/ensemble/_forest.py
--- a/sklearn/ensemble/_forest.py
+++ b/sklearn/ensemble/_forest.py
@@ -48,7 +48,6 @@ class calls the ``fit`` method of each sub-estimator on random samples
 import numpy as np
 from scipy.sparse import issparse
 from scipy.sparse import hstack as sparse_hstack
-from joblib import Parallel
 
 from ..base import is_classifier
 from ..base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin
@@ -66,7 +65,7 @@ class calls the ``fit`` method of each sub-estimator on random samples
 from ..utils import check_random_state, compute_sample_weight
 from ..exceptions import DataConversionWarning
 from ._base import BaseEnsemble, _partition_estimators
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.multiclass import check_classification_targets, type_of_target
 from ..utils.validation import (
     check_is_fitted,
diff --git a/sklearn/ensemble/_stacking.py b/sklearn/ensemble/_stacking.py
--- a/sklearn/ensemble/_stacking.py
+++ b/sklearn/ensemble/_stacking.py
@@ -8,7 +8,6 @@
 from numbers import Integral
 
 import numpy as np
-from joblib import Parallel
 import scipy.sparse as sparse
 
 from ..base import clone
@@ -33,7 +32,7 @@
 from ..utils.metaestimators import available_if
 from ..utils.validation import check_is_fitted
 from ..utils.validation import column_or_1d
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import HasMethods, StrOptions
 from ..utils.validation import _check_feature_names_in
 
diff --git a/sklearn/ensemble/_voting.py b/sklearn/ensemble/_voting.py
--- a/sklearn/ensemble/_voting.py
+++ b/sklearn/ensemble/_voting.py
@@ -18,8 +18,6 @@
 
 import numpy as np
 
-from joblib import Parallel
-
 from ..base import ClassifierMixin
 from ..base import RegressorMixin
 from ..base import TransformerMixin
@@ -36,7 +34,7 @@
 from ..utils._param_validation import StrOptions
 from ..exceptions import NotFittedError
 from ..utils._estimator_html_repr import _VisualBlock
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
diff --git a/sklearn/feature_selection/_rfe.py b/sklearn/feature_selection/_rfe.py
--- a/sklearn/feature_selection/_rfe.py
+++ b/sklearn/feature_selection/_rfe.py
@@ -8,7 +8,7 @@
 
 import numpy as np
 from numbers import Integral, Real
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 
 from ..utils.metaestimators import available_if
@@ -16,7 +16,7 @@
 from ..utils._param_validation import HasMethods, Interval
 from ..utils._tags import _safe_tags
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..base import BaseEstimator
 from ..base import MetaEstimatorMixin
 from ..base import clone
diff --git a/sklearn/inspection/_permutation_importance.py b/sklearn/inspection/_permutation_importance.py
--- a/sklearn/inspection/_permutation_importance.py
+++ b/sklearn/inspection/_permutation_importance.py
@@ -1,7 +1,6 @@
 """Permutation importance for estimators."""
 import numbers
 import numpy as np
-from joblib import Parallel
 
 from ..ensemble._bagging import _generate_indices
 from ..metrics import check_scoring
@@ -10,7 +9,7 @@
 from ..utils import Bunch, _safe_indexing
 from ..utils import check_random_state
 from ..utils import check_array
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 def _weights_scorer(scorer, estimator, X, y, sample_weight):
diff --git a/sklearn/inspection/_plot/partial_dependence.py b/sklearn/inspection/_plot/partial_dependence.py
--- a/sklearn/inspection/_plot/partial_dependence.py
+++ b/sklearn/inspection/_plot/partial_dependence.py
@@ -6,7 +6,6 @@
 import numpy as np
 from scipy import sparse
 from scipy.stats.mstats import mquantiles
-from joblib import Parallel
 
 from .. import partial_dependence
 from .._pd_utils import _check_feature_names, _get_feature_index
@@ -16,7 +15,7 @@
 from ...utils import check_matplotlib_support  # noqa
 from ...utils import check_random_state
 from ...utils import _safe_indexing
-from ...utils.fixes import delayed
+from ...utils.parallel import delayed, Parallel
 from ...utils._encode import _unique
 
 
diff --git a/sklearn/linear_model/_base.py b/sklearn/linear_model/_base.py
--- a/sklearn/linear_model/_base.py
+++ b/sklearn/linear_model/_base.py
@@ -25,7 +25,6 @@
 from scipy import sparse
 from scipy.sparse.linalg import lsqr
 from scipy.special import expit
-from joblib import Parallel
 from numbers import Integral
 
 from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
@@ -40,7 +39,7 @@
 from ..utils._seq_dataset import ArrayDataset32, CSRDataset32
 from ..utils._seq_dataset import ArrayDataset64, CSRDataset64
 from ..utils.validation import check_is_fitted, _check_sample_weight
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 # TODO: bayesian_ridge_regression and bayesian_regression_ard
 # should be squashed into its respective objects.
diff --git a/sklearn/linear_model/_coordinate_descent.py b/sklearn/linear_model/_coordinate_descent.py
--- a/sklearn/linear_model/_coordinate_descent.py
+++ b/sklearn/linear_model/_coordinate_descent.py
@@ -14,7 +14,7 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._base import LinearModel, _pre_fit
 from ..base import RegressorMixin, MultiOutputMixin
@@ -30,7 +30,7 @@
     check_is_fitted,
     column_or_1d,
 )
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 # mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
 from . import _cd_fast as cd_fast  # type: ignore
diff --git a/sklearn/linear_model/_least_angle.py b/sklearn/linear_model/_least_angle.py
--- a/sklearn/linear_model/_least_angle.py
+++ b/sklearn/linear_model/_least_angle.py
@@ -16,7 +16,6 @@
 import numpy as np
 from scipy import linalg, interpolate
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel
 
 from ._base import LinearModel, LinearRegression
 from ._base import _deprecate_normalize, _preprocess_data
@@ -28,7 +27,7 @@
 from ..utils._param_validation import Hidden, Interval, StrOptions
 from ..model_selection import check_cv
 from ..exceptions import ConvergenceWarning
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 SOLVE_TRIANGULAR_ARGS = {"check_finite": False}
 
diff --git a/sklearn/linear_model/_logistic.py b/sklearn/linear_model/_logistic.py
--- a/sklearn/linear_model/_logistic.py
+++ b/sklearn/linear_model/_logistic.py
@@ -16,7 +16,7 @@
 
 import numpy as np
 from scipy import optimize
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from sklearn.metrics import get_scorer_names
 
@@ -34,7 +34,7 @@
 from ..utils.optimize import _newton_cg, _check_optimize_result
 from ..utils.validation import check_is_fitted, _check_sample_weight
 from ..utils.multiclass import check_classification_targets
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import StrOptions, Interval
 from ..model_selection import check_cv
 from ..metrics import get_scorer
diff --git a/sklearn/linear_model/_omp.py b/sklearn/linear_model/_omp.py
--- a/sklearn/linear_model/_omp.py
+++ b/sklearn/linear_model/_omp.py
@@ -12,12 +12,11 @@
 import numpy as np
 from scipy import linalg
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel
 
 from ._base import LinearModel, _pre_fit, _deprecate_normalize
 from ..base import RegressorMixin, MultiOutputMixin
 from ..utils import as_float_array, check_array
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Hidden, Interval, StrOptions
 from ..model_selection import check_cv
 
diff --git a/sklearn/linear_model/_stochastic_gradient.py b/sklearn/linear_model/_stochastic_gradient.py
--- a/sklearn/linear_model/_stochastic_gradient.py
+++ b/sklearn/linear_model/_stochastic_gradient.py
@@ -12,8 +12,6 @@
 from abc import ABCMeta, abstractmethod
 from numbers import Integral, Real
 
-from joblib import Parallel
-
 from ..base import clone, is_classifier
 from ._base import LinearClassifierMixin, SparseCoefMixin
 from ._base import make_dataset
@@ -26,7 +24,7 @@
 from ..utils._param_validation import Interval
 from ..utils._param_validation import StrOptions
 from ..utils._param_validation import Hidden
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..exceptions import ConvergenceWarning
 from ..model_selection import StratifiedShuffleSplit, ShuffleSplit
 
diff --git a/sklearn/linear_model/_theil_sen.py b/sklearn/linear_model/_theil_sen.py
--- a/sklearn/linear_model/_theil_sen.py
+++ b/sklearn/linear_model/_theil_sen.py
@@ -15,13 +15,13 @@
 from scipy import linalg
 from scipy.special import binom
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._base import LinearModel
 from ..base import RegressorMixin
 from ..utils import check_random_state
 from ..utils._param_validation import Interval
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..exceptions import ConvergenceWarning
 
 _EPSILON = np.finfo(np.double).eps
diff --git a/sklearn/manifold/_mds.py b/sklearn/manifold/_mds.py
--- a/sklearn/manifold/_mds.py
+++ b/sklearn/manifold/_mds.py
@@ -8,7 +8,7 @@
 from numbers import Integral, Real
 
 import numpy as np
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 import warnings
 
@@ -17,7 +17,7 @@
 from ..utils import check_random_state, check_array, check_symmetric
 from ..isotonic import IsotonicRegression
 from ..utils._param_validation import Interval, StrOptions, Hidden
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 def _smacof_single(
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -15,7 +15,7 @@
 from scipy.spatial import distance
 from scipy.sparse import csr_matrix
 from scipy.sparse import issparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from .. import config_context
 from ..utils.validation import _num_samples
@@ -27,7 +27,7 @@
 from ..utils.extmath import row_norms, safe_sparse_dot
 from ..preprocessing import normalize
 from ..utils._mask import _get_mask
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.fixes import sp_version, parse_version
 
 from ._pairwise_distances_reduction import ArgKmin
diff --git a/sklearn/model_selection/_search.py b/sklearn/model_selection/_search.py
--- a/sklearn/model_selection/_search.py
+++ b/sklearn/model_selection/_search.py
@@ -33,14 +33,13 @@
 from ._validation import _normalize_score_results
 from ._validation import _warn_or_raise_about_fit_failures
 from ..exceptions import NotFittedError
-from joblib import Parallel
 from ..utils import check_random_state
 from ..utils.random import sample_without_replacement
 from ..utils._param_validation import HasMethods, Interval, StrOptions
 from ..utils._tags import _safe_tags
 from ..utils.validation import indexable, check_is_fitted, _check_fit_params
 from ..utils.metaestimators import available_if
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..metrics._scorer import _check_multimetric_scoring, get_scorer_names
 from ..metrics import check_scoring
 
diff --git a/sklearn/model_selection/_validation.py b/sklearn/model_selection/_validation.py
--- a/sklearn/model_selection/_validation.py
+++ b/sklearn/model_selection/_validation.py
@@ -21,13 +21,13 @@
 
 import numpy as np
 import scipy.sparse as sp
-from joblib import Parallel, logger
+from joblib import logger
 
 from ..base import is_classifier, clone
 from ..utils import indexable, check_random_state, _safe_indexing
 from ..utils.validation import _check_fit_params
 from ..utils.validation import _num_samples
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.metaestimators import _safe_split
 from ..metrics import check_scoring
 from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
diff --git a/sklearn/multiclass.py b/sklearn/multiclass.py
--- a/sklearn/multiclass.py
+++ b/sklearn/multiclass.py
@@ -56,9 +56,7 @@
     _ovr_decision_function,
 )
 from .utils.metaestimators import _safe_split, available_if
-from .utils.fixes import delayed
-
-from joblib import Parallel
+from .utils.parallel import delayed, Parallel
 
 __all__ = [
     "OneVsRestClassifier",
diff --git a/sklearn/multioutput.py b/sklearn/multioutput.py
--- a/sklearn/multioutput.py
+++ b/sklearn/multioutput.py
@@ -17,7 +17,6 @@
 
 import numpy as np
 import scipy.sparse as sp
-from joblib import Parallel
 
 from abc import ABCMeta, abstractmethod
 from .base import BaseEstimator, clone, MetaEstimatorMixin
@@ -31,7 +30,7 @@
     has_fit_parameter,
     _check_fit_params,
 )
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .utils._param_validation import HasMethods, StrOptions
 
 __all__ = [
diff --git a/sklearn/neighbors/_base.py b/sklearn/neighbors/_base.py
--- a/sklearn/neighbors/_base.py
+++ b/sklearn/neighbors/_base.py
@@ -16,7 +16,7 @@
 
 import numpy as np
 from scipy.sparse import csr_matrix, issparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._ball_tree import BallTree
 from ._kd_tree import KDTree
@@ -37,8 +37,8 @@
 from ..utils.validation import check_is_fitted
 from ..utils.validation import check_non_negative
 from ..utils._param_validation import Interval, StrOptions
-from ..utils.fixes import delayed, sp_version
-from ..utils.fixes import parse_version
+from ..utils.parallel import delayed, Parallel
+from ..utils.fixes import parse_version, sp_version
 from ..exceptions import DataConversionWarning, EfficiencyWarning
 
 VALID_METRICS = dict(
diff --git a/sklearn/pipeline.py b/sklearn/pipeline.py
--- a/sklearn/pipeline.py
+++ b/sklearn/pipeline.py
@@ -14,7 +14,6 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel
 
 from .base import clone, TransformerMixin
 from .preprocessing import FunctionTransformer
@@ -30,7 +29,7 @@
 from .utils import check_pandas_support
 from .utils._param_validation import HasMethods, Hidden
 from .utils._set_output import _safe_set_output, _get_output_config
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .exceptions import NotFittedError
 
 from .utils.metaestimators import _BaseComposition
diff --git a/sklearn/utils/fixes.py b/sklearn/utils/fixes.py
--- a/sklearn/utils/fixes.py
+++ b/sklearn/utils/fixes.py
@@ -10,9 +10,7 @@
 #
 # License: BSD 3 clause
 
-from functools import update_wrapper
 from importlib import resources
-import functools
 import sys
 
 import sklearn
@@ -20,7 +18,8 @@
 import scipy
 import scipy.stats
 import threadpoolctl
-from .._config import config_context, get_config
+
+from .deprecation import deprecated
 from ..externals._packaging.version import parse as parse_version
 
 
@@ -106,30 +105,6 @@ def _eigh(*args, **kwargs):
         return scipy.linalg.eigh(*args, eigvals=eigvals, **kwargs)
 
 
-# remove when https://github.com/joblib/joblib/issues/1071 is fixed
-def delayed(function):
-    """Decorator used to capture the arguments of a function."""
-
-    @functools.wraps(function)
-    def delayed_function(*args, **kwargs):
-        return _FuncWrapper(function), args, kwargs
-
-    return delayed_function
-
-
-class _FuncWrapper:
-    """ "Load the global configuration before calling the function."""
-
-    def __init__(self, function):
-        self.function = function
-        self.config = get_config()
-        update_wrapper(self, self.function)
-
-    def __call__(self, *args, **kwargs):
-        with config_context(**self.config):
-            return self.function(*args, **kwargs)
-
-
 # Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
 # `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
 def _percentile(a, q, *, method="linear", **kwargs):
@@ -178,6 +153,16 @@ def threadpool_info():
 threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__
 
 
+@deprecated(
+    "The function `delayed` has been moved from `sklearn.utils.fixes` to "
+    "`sklearn.utils.parallel`. This import path will be removed in 1.5."
+)
+def delayed(function):
+    from sklearn.utils.parallel import delayed
+
+    return delayed(function)
+
+
 # TODO: Remove when SciPy 1.11 is the minimum supported version
 def _mode(a, axis=0):
     if sp_version >= parse_version("1.9.0"):
diff --git a/sklearn/utils/parallel.py b/sklearn/utils/parallel.py
new file mode 100644
--- /dev/null
+++ b/sklearn/utils/parallel.py
@@ -0,0 +1,123 @@
+"""Module that customize joblib tools for scikit-learn usage."""
+
+import functools
+import warnings
+from functools import update_wrapper
+
+import joblib
+
+from .._config import config_context, get_config
+
+
+def _with_config(delayed_func, config):
+    """Helper function that intends to attach a config to a delayed function."""
+    if hasattr(delayed_func, "with_config"):
+        return delayed_func.with_config(config)
+    else:
+        warnings.warn(
+            "`sklearn.utils.parallel.Parallel` needs to be used in "
+            "conjunction with `sklearn.utils.parallel.delayed` instead of "
+            "`joblib.delayed` to correctly propagate the scikit-learn "
+            "configuration to the joblib workers.",
+            UserWarning,
+        )
+        return delayed_func
+
+
+class Parallel(joblib.Parallel):
+    """Tweak of :class:`joblib.Parallel` that propagates the scikit-learn configuration.
+
+    This subclass of :class:`joblib.Parallel` ensures that the active configuration
+    (thread-local) of scikit-learn is propagated to the parallel workers for the
+    duration of the execution of the parallel tasks.
+
+    The API does not change and you can refer to :class:`joblib.Parallel`
+    documentation for more details.
+
+    .. versionadded:: 1.3
+    """
+
+    def __call__(self, iterable):
+        """Dispatch the tasks and return the results.
+
+        Parameters
+        ----------
+        iterable : iterable
+            Iterable containing tuples of (delayed_function, args, kwargs) that should
+            be consumed.
+
+        Returns
+        -------
+        results : list
+            List of results of the tasks.
+        """
+        # Capture the thread-local scikit-learn configuration at the time
+        # Parallel.__call__ is issued since the tasks can be dispatched
+        # in a different thread depending on the backend and on the value of
+        # pre_dispatch and n_jobs.
+        config = get_config()
+        iterable_with_config = (
+            (_with_config(delayed_func, config), args, kwargs)
+            for delayed_func, args, kwargs in iterable
+        )
+        return super().__call__(iterable_with_config)
+
+
+# remove when https://github.com/joblib/joblib/issues/1071 is fixed
+def delayed(function):
+    """Decorator used to capture the arguments of a function.
+
+    This alternative to `joblib.delayed` is meant to be used in conjunction
+    with `sklearn.utils.parallel.Parallel`. The latter captures the the scikit-
+    learn configuration by calling `sklearn.get_config()` in the current
+    thread, prior to dispatching the first task. The captured configuration is
+    then propagated and enabled for the duration of the execution of the
+    delayed function in the joblib workers.
+
+    .. versionchanged:: 1.3
+       `delayed` was moved from `sklearn.utils.fixes` to `sklearn.utils.parallel`
+       in scikit-learn 1.3.
+
+    Parameters
+    ----------
+    function : callable
+        The function to be delayed.
+
+    Returns
+    -------
+    output: tuple
+        Tuple containing the delayed function, the positional arguments, and the
+        keyword arguments.
+    """
+
+    @functools.wraps(function)
+    def delayed_function(*args, **kwargs):
+        return _FuncWrapper(function), args, kwargs
+
+    return delayed_function
+
+
+class _FuncWrapper:
+    """Load the global configuration before calling the function."""
+
+    def __init__(self, function):
+        self.function = function
+        update_wrapper(self, self.function)
+
+    def with_config(self, config):
+        self.config = config
+        return self
+
+    def __call__(self, *args, **kwargs):
+        config = getattr(self, "config", None)
+        if config is None:
+            warnings.warn(
+                "`sklearn.utils.parallel.delayed` should be used with "
+                "`sklearn.utils.parallel.Parallel` to make it possible to propagate "
+                "the scikit-learn configuration of the current thread to the "
+                "joblib workers.",
+                UserWarning,
+            )
+            config = {}
+        with config_context(**config):
+            return self.function(*args, **kwargs)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| benchmarks/bench_saga.py | 10 | 11 | - | - | -
| sklearn/calibration.py | 17 | 17 | - | - | -
| sklearn/calibration.py | 39 | 39 | - | - | -
| sklearn/cluster/_mean_shift.py | 19 | 25 | - | - | -
| sklearn/compose/_column_transformer.py | 15 | 15 | - | - | -
| sklearn/compose/_column_transformer.py | 29 | 29 | - | - | -
| sklearn/covariance/_graph_lasso.py | 16 | 16 | - | - | -
| sklearn/covariance/_graph_lasso.py | 26 | 26 | - | - | -
| sklearn/decomposition/_dict_learning.py | 16 | 16 | - | - | -
| sklearn/decomposition/_dict_learning.py | 24 | 24 | - | - | -
| sklearn/decomposition/_lda.py | 18 | 24 | - | - | -
| sklearn/ensemble/_bagging.py | 15 | 16 | - | 5 | -
| sklearn/ensemble/_bagging.py | 28 | 28 | - | 5 | -
| sklearn/ensemble/_forest.py | 51 | 51 | - | 21 | -
| sklearn/ensemble/_forest.py | 69 | 69 | - | 21 | -
| sklearn/ensemble/_stacking.py | 11 | 11 | - | - | -
| sklearn/ensemble/_stacking.py | 36 | 36 | - | - | -
| sklearn/ensemble/_voting.py | 21 | 22 | - | - | -
| sklearn/ensemble/_voting.py | 39 | 39 | - | - | -
| sklearn/feature_selection/_rfe.py | 11 | 11 | - | - | -
| sklearn/feature_selection/_rfe.py | 19 | 19 | - | - | -
| sklearn/inspection/_permutation_importance.py | 4 | 4 | - | - | -
| sklearn/inspection/_permutation_importance.py | 13 | 13 | - | - | -
| sklearn/inspection/_plot/partial_dependence.py | 9 | 9 | - | - | -
| sklearn/inspection/_plot/partial_dependence.py | 19 | 19 | - | - | -
| sklearn/linear_model/_base.py | 28 | 28 | - | - | -
| sklearn/linear_model/_base.py | 43 | 43 | - | - | -
| sklearn/linear_model/_coordinate_descent.py | 17 | 17 | 4 | 4 | 2027
| sklearn/linear_model/_coordinate_descent.py | 33 | 33 | 4 | 4 | 2027
| sklearn/linear_model/_least_angle.py | 19 | 19 | - | - | -
| sklearn/linear_model/_least_angle.py | 31 | 31 | - | - | -
| sklearn/linear_model/_logistic.py | 19 | 19 | - | - | -
| sklearn/linear_model/_logistic.py | 37 | 37 | - | - | -
| sklearn/linear_model/_omp.py | 15 | 20 | 20 | 15 | 7315
| sklearn/linear_model/_stochastic_gradient.py | 15 | 16 | 19 | 14 | 7154
| sklearn/linear_model/_stochastic_gradient.py | 29 | 29 | 19 | 14 | 7154
| sklearn/linear_model/_theil_sen.py | 18 | 24 | - | - | -
| sklearn/manifold/_mds.py | 11 | 11 | - | - | -
| sklearn/manifold/_mds.py | 20 | 20 | - | - | -
| sklearn/metrics/pairwise.py | 18 | 18 | 25 | 18 | 8764
| sklearn/metrics/pairwise.py | 30 | 30 | 25 | 18 | 8764
| sklearn/model_selection/_search.py | 36 | 43 | - | - | -
| sklearn/model_selection/_validation.py | 24 | 30 | - | - | -
| sklearn/multiclass.py | 59 | 61 | - | - | -
| sklearn/multioutput.py | 20 | 20 | - | - | -
| sklearn/multioutput.py | 34 | 34 | - | - | -
| sklearn/neighbors/_base.py | 19 | 19 | - | - | -
| sklearn/neighbors/_base.py | 40 | 41 | - | - | -
| sklearn/pipeline.py | 17 | 17 | - | 20 | -
| sklearn/pipeline.py | 33 | 33 | - | 20 | -
| sklearn/utils/fixes.py | 13 | 15 | 33 | 2 | 10944
| sklearn/utils/fixes.py | 23 | 23 | 33 | 2 | 10944
| sklearn/utils/fixes.py | 109 | 132 | 2 | 2 | 811
| sklearn/utils/fixes.py | 181 | 181 | - | 2 | -
| sklearn/utils/parallel.py | 0 | 0 | - | - | -


## Problem Statement

```
FIX pass explicit configuration to delayed
Working alternative to #25242
closes #25242 
closes #25239 

This is an alternative to #25242 that does not work if the thread import scikit-learn is different from the thread making the call to `Parallel`.

Here, we have an alternative where we pass explicitly the configuration that is obtained by the thread that makes the `Parallel` code.

We raise a warning if this is not the case. It makes sure that it will turn into an error if we forget to pass the config to `delayed`. The code will still be working if `joblib` decides to provide a way to provide a `context` and a `config`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/utils/_joblib.py | 1 | 32| 177 | 177 | 177 | 
| **-> 2 <-** | **2 sklearn/utils/fixes.py** | 98 | 178| 634 | 811 | 2008 | 
| 3 | 3 sklearn/_config.py | 162 | 295| 1017 | 1828 | 4285 | 
| **-> 4 <-** | **4 sklearn/linear_model/_coordinate_descent.py** | 8 | 36| 199 | 2027 | 29301 | 
| 5 | 4 sklearn/_config.py | 1 | 27| 214 | 2241 | 29301 | 
| 6 | **5 sklearn/ensemble/_bagging.py** | 206 | 219| 108 | 2349 | 38662 | 
| 7 | 5 sklearn/_config.py | 48 | 142| 755 | 3104 | 38662 | 
| 8 | 6 sklearn/utils/_testing.py | 391 | 450| 539 | 3643 | 46484 | 
| 9 | 6 sklearn/_config.py | 144 | 159| 235 | 3878 | 46484 | 
| 10 | 7 sklearn/exceptions.py | 41 | 64| 174 | 4052 | 47635 | 
| 11 | 8 setup.py | 169 | 211| 344 | 4396 | 53578 | 
| 12 | 9 sklearn/conftest.py | 80 | 184| 794 | 5190 | 55393 | 
| 13 | 10 sklearn/utils/__init__.py | 1 | 80| 513 | 5703 | 64466 | 
| 14 | 10 sklearn/exceptions.py | 81 | 118| 257 | 5960 | 64466 | 
| 15 | 11 sklearn/cluster/_kmeans.py | 893 | 918| 304 | 6264 | 82507 | 
| 16 | 12 sklearn/experimental/enable_hist_gradient_boosting.py | 1 | 22| 171 | 6435 | 82678 | 
| 17 | 13 sklearn/utils/deprecation.py | 1 | 35| 195 | 6630 | 83385 | 
| 18 | 13 sklearn/exceptions.py | 155 | 166| 121 | 6751 | 83385 | 
| **-> 19 <-** | **14 sklearn/linear_model/_stochastic_gradient.py** | 9 | 58| 403 | 7154 | 102982 | 
| **-> 20 <-** | **15 sklearn/linear_model/_omp.py** | 1 | 28| 161 | 7315 | 112281 | 
| 21 | **15 sklearn/ensemble/_bagging.py** | 154 | 178| 178 | 7493 | 112281 | 
| 22 | **15 sklearn/ensemble/_bagging.py** | 73 | 151| 540 | 8033 | 112281 | 
| 23 | 16 doc/conf.py | 553 | 562| 123 | 8156 | 117755 | 
| 24 | 17 sklearn/utils/validation.py | 39 | 93| 417 | 8573 | 134698 | 
| **-> 25 <-** | **18 sklearn/metrics/pairwise.py** | 10 | 35| 191 | 8764 | 153412 | 
| 26 | 18 sklearn/exceptions.py | 133 | 153| 130 | 8894 | 153412 | 
| 27 | 18 sklearn/utils/_testing.py | 161 | 215| 401 | 9295 | 153412 | 
| 28 | 19 sklearn/linear_model/_passive_aggressive.py | 177 | 221| 285 | 9580 | 157762 | 
| 29 | **20 sklearn/pipeline.py** | 1230 | 1253| 193 | 9773 | 168156 | 
| 30 | **21 sklearn/ensemble/_forest.py** | 654 | 708| 316 | 10089 | 192182 | 
| 31 | 22 sklearn/__init__.py | 68 | 129| 430 | 10519 | 193298 | 
| 32 | 22 sklearn/cluster/_kmeans.py | 1912 | 1921| 120 | 10639 | 193298 | 
| **-> 33 <-** | **22 sklearn/utils/fixes.py** | 1 | 46| 305 | 10944 | 193298 | 
| 34 | **22 sklearn/ensemble/_bagging.py** | 181 | 203| 203 | 11147 | 193298 | 
| 35 | 22 sklearn/utils/deprecation.py | 58 | 75| 123 | 11270 | 193298 | 
| 36 | 23 examples/applications/plot_prediction_latency.py | 177 | 197| 229 | 11499 | 195937 | 
| 37 | 24 benchmarks/bench_hist_gradient_boosting_threading.py | 241 | 289| 367 | 11866 | 198675 | 


## Missing Patch Files

 * 1: benchmarks/bench_saga.py
 * 2: sklearn/calibration.py
 * 3: sklearn/cluster/_mean_shift.py
 * 4: sklearn/compose/_column_transformer.py
 * 5: sklearn/covariance/_graph_lasso.py
 * 6: sklearn/decomposition/_dict_learning.py
 * 7: sklearn/decomposition/_lda.py
 * 8: sklearn/ensemble/_bagging.py
 * 9: sklearn/ensemble/_forest.py
 * 10: sklearn/ensemble/_stacking.py
 * 11: sklearn/ensemble/_voting.py
 * 12: sklearn/feature_selection/_rfe.py
 * 13: sklearn/inspection/_permutation_importance.py
 * 14: sklearn/inspection/_plot/partial_dependence.py
 * 15: sklearn/linear_model/_base.py
 * 16: sklearn/linear_model/_coordinate_descent.py
 * 17: sklearn/linear_model/_least_angle.py
 * 18: sklearn/linear_model/_logistic.py
 * 19: sklearn/linear_model/_omp.py
 * 20: sklearn/linear_model/_stochastic_gradient.py
 * 21: sklearn/linear_model/_theil_sen.py
 * 22: sklearn/manifold/_mds.py
 * 23: sklearn/metrics/pairwise.py
 * 24: sklearn/model_selection/_search.py
 * 25: sklearn/model_selection/_validation.py
 * 26: sklearn/multiclass.py
 * 27: sklearn/multioutput.py
 * 28: sklearn/neighbors/_base.py
 * 29: sklearn/pipeline.py
 * 30: sklearn/utils/fixes.py
 * 31: sklearn/utils/parallel.py

### Hint

```
Thinking more about it, we could also make this more automatic by subclassing `joblib.Parallel` as `sklearn.fixes.Parallel` to overried the `Parallel.__call__` method to automatically call `sklearn.get_config` there and then rewrap the generator args of `Parallel.__call__` to call `delayed_object.set_config(config)` on each task.

That would mandate using the `sklearn.fixes.Parallel` subclass everywhere though.

And indeed, maybe we should consider those tools (`Parallel` and `delayed`) semi-public with proper docstrings to explain how they extend the joblib equivalent to propagate scikit-learn specific configuration to worker threads and processes.
```

## Patch

```diff
diff --git a/benchmarks/bench_saga.py b/benchmarks/bench_saga.py
--- a/benchmarks/bench_saga.py
+++ b/benchmarks/bench_saga.py
@@ -7,8 +7,7 @@
 import time
 import os
 
-from joblib import Parallel
-from sklearn.utils.fixes import delayed
+from sklearn.utils.parallel import delayed, Parallel
 import matplotlib.pyplot as plt
 import numpy as np
 
diff --git a/sklearn/calibration.py b/sklearn/calibration.py
--- a/sklearn/calibration.py
+++ b/sklearn/calibration.py
@@ -14,7 +14,6 @@
 
 from math import log
 import numpy as np
-from joblib import Parallel
 
 from scipy.special import expit
 from scipy.special import xlogy
@@ -36,7 +35,7 @@
 )
 
 from .utils.multiclass import check_classification_targets
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .utils._param_validation import StrOptions, HasMethods, Hidden
 from .utils.validation import (
     _check_fit_params,
diff --git a/sklearn/cluster/_mean_shift.py b/sklearn/cluster/_mean_shift.py
--- a/sklearn/cluster/_mean_shift.py
+++ b/sklearn/cluster/_mean_shift.py
@@ -16,13 +16,12 @@
 
 import numpy as np
 import warnings
-from joblib import Parallel
 from numbers import Integral, Real
 
 from collections import defaultdict
 from ..utils._param_validation import Interval, validate_params
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils import check_random_state, gen_batches, check_array
 from ..base import BaseEstimator, ClusterMixin
 from ..neighbors import NearestNeighbors
diff --git a/sklearn/compose/_column_transformer.py b/sklearn/compose/_column_transformer.py
--- a/sklearn/compose/_column_transformer.py
+++ b/sklearn/compose/_column_transformer.py
@@ -12,7 +12,6 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel
 
 from ..base import clone, TransformerMixin
 from ..utils._estimator_html_repr import _VisualBlock
@@ -26,7 +25,7 @@
 from ..utils import check_pandas_support
 from ..utils.metaestimators import _BaseComposition
 from ..utils.validation import check_array, check_is_fitted, _check_feature_names_in
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 __all__ = ["ColumnTransformer", "make_column_transformer", "make_column_selector"]
diff --git a/sklearn/covariance/_graph_lasso.py b/sklearn/covariance/_graph_lasso.py
--- a/sklearn/covariance/_graph_lasso.py
+++ b/sklearn/covariance/_graph_lasso.py
@@ -13,7 +13,6 @@
 from numbers import Integral, Real
 import numpy as np
 from scipy import linalg
-from joblib import Parallel
 
 from . import empirical_covariance, EmpiricalCovariance, log_likelihood
 
@@ -23,7 +22,7 @@
     check_random_state,
     check_scalar,
 )
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Interval, StrOptions
 
 # mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
diff --git a/sklearn/decomposition/_dict_learning.py b/sklearn/decomposition/_dict_learning.py
--- a/sklearn/decomposition/_dict_learning.py
+++ b/sklearn/decomposition/_dict_learning.py
@@ -13,7 +13,7 @@
 
 import numpy as np
 from scipy import linalg
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
 from ..utils import check_array, check_random_state, gen_even_slices, gen_batches
@@ -21,7 +21,7 @@
 from ..utils._param_validation import validate_params
 from ..utils.extmath import randomized_svd, row_norms, svd_flip
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars
 
 
diff --git a/sklearn/decomposition/_lda.py b/sklearn/decomposition/_lda.py
--- a/sklearn/decomposition/_lda.py
+++ b/sklearn/decomposition/_lda.py
@@ -15,13 +15,13 @@
 import numpy as np
 import scipy.sparse as sp
 from scipy.special import gammaln, logsumexp
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ..base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
 from ..utils import check_random_state, gen_batches, gen_even_slices
 from ..utils.validation import check_non_negative
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Interval, StrOptions
 
 from ._online_lda_fast import (
diff --git a/sklearn/ensemble/_bagging.py b/sklearn/ensemble/_bagging.py
--- a/sklearn/ensemble/_bagging.py
+++ b/sklearn/ensemble/_bagging.py
@@ -12,8 +12,6 @@
 from warnings import warn
 from functools import partial
 
-from joblib import Parallel
-
 from ._base import BaseEnsemble, _partition_estimators
 from ..base import ClassifierMixin, RegressorMixin
 from ..metrics import r2_score, accuracy_score
@@ -25,7 +23,7 @@
 from ..utils.random import sample_without_replacement
 from ..utils._param_validation import Interval, HasMethods, StrOptions
 from ..utils.validation import has_fit_parameter, check_is_fitted, _check_sample_weight
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 __all__ = ["BaggingClassifier", "BaggingRegressor"]
diff --git a/sklearn/ensemble/_forest.py b/sklearn/ensemble/_forest.py
--- a/sklearn/ensemble/_forest.py
+++ b/sklearn/ensemble/_forest.py
@@ -48,7 +48,6 @@ class calls the ``fit`` method of each sub-estimator on random samples
 import numpy as np
 from scipy.sparse import issparse
 from scipy.sparse import hstack as sparse_hstack
-from joblib import Parallel
 
 from ..base import is_classifier
 from ..base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin
@@ -66,7 +65,7 @@ class calls the ``fit`` method of each sub-estimator on random samples
 from ..utils import check_random_state, compute_sample_weight
 from ..exceptions import DataConversionWarning
 from ._base import BaseEnsemble, _partition_estimators
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.multiclass import check_classification_targets, type_of_target
 from ..utils.validation import (
     check_is_fitted,
diff --git a/sklearn/ensemble/_stacking.py b/sklearn/ensemble/_stacking.py
--- a/sklearn/ensemble/_stacking.py
+++ b/sklearn/ensemble/_stacking.py
@@ -8,7 +8,6 @@
 from numbers import Integral
 
 import numpy as np
-from joblib import Parallel
 import scipy.sparse as sparse
 
 from ..base import clone
@@ -33,7 +32,7 @@
 from ..utils.metaestimators import available_if
 from ..utils.validation import check_is_fitted
 from ..utils.validation import column_or_1d
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import HasMethods, StrOptions
 from ..utils.validation import _check_feature_names_in
 
diff --git a/sklearn/ensemble/_voting.py b/sklearn/ensemble/_voting.py
--- a/sklearn/ensemble/_voting.py
+++ b/sklearn/ensemble/_voting.py
@@ -18,8 +18,6 @@
 
 import numpy as np
 
-from joblib import Parallel
-
 from ..base import ClassifierMixin
 from ..base import RegressorMixin
 from ..base import TransformerMixin
@@ -36,7 +34,7 @@
 from ..utils._param_validation import StrOptions
 from ..exceptions import NotFittedError
 from ..utils._estimator_html_repr import _VisualBlock
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
diff --git a/sklearn/feature_selection/_rfe.py b/sklearn/feature_selection/_rfe.py
--- a/sklearn/feature_selection/_rfe.py
+++ b/sklearn/feature_selection/_rfe.py
@@ -8,7 +8,7 @@
 
 import numpy as np
 from numbers import Integral, Real
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 
 from ..utils.metaestimators import available_if
@@ -16,7 +16,7 @@
 from ..utils._param_validation import HasMethods, Interval
 from ..utils._tags import _safe_tags
 from ..utils.validation import check_is_fitted
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..base import BaseEstimator
 from ..base import MetaEstimatorMixin
 from ..base import clone
diff --git a/sklearn/inspection/_permutation_importance.py b/sklearn/inspection/_permutation_importance.py
--- a/sklearn/inspection/_permutation_importance.py
+++ b/sklearn/inspection/_permutation_importance.py
@@ -1,7 +1,6 @@
 """Permutation importance for estimators."""
 import numbers
 import numpy as np
-from joblib import Parallel
 
 from ..ensemble._bagging import _generate_indices
 from ..metrics import check_scoring
@@ -10,7 +9,7 @@
 from ..utils import Bunch, _safe_indexing
 from ..utils import check_random_state
 from ..utils import check_array
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 def _weights_scorer(scorer, estimator, X, y, sample_weight):
diff --git a/sklearn/inspection/_plot/partial_dependence.py b/sklearn/inspection/_plot/partial_dependence.py
--- a/sklearn/inspection/_plot/partial_dependence.py
+++ b/sklearn/inspection/_plot/partial_dependence.py
@@ -6,7 +6,6 @@
 import numpy as np
 from scipy import sparse
 from scipy.stats.mstats import mquantiles
-from joblib import Parallel
 
 from .. import partial_dependence
 from .._pd_utils import _check_feature_names, _get_feature_index
@@ -16,7 +15,7 @@
 from ...utils import check_matplotlib_support  # noqa
 from ...utils import check_random_state
 from ...utils import _safe_indexing
-from ...utils.fixes import delayed
+from ...utils.parallel import delayed, Parallel
 from ...utils._encode import _unique
 
 
diff --git a/sklearn/linear_model/_base.py b/sklearn/linear_model/_base.py
--- a/sklearn/linear_model/_base.py
+++ b/sklearn/linear_model/_base.py
@@ -25,7 +25,6 @@
 from scipy import sparse
 from scipy.sparse.linalg import lsqr
 from scipy.special import expit
-from joblib import Parallel
 from numbers import Integral
 
 from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
@@ -40,7 +39,7 @@
 from ..utils._seq_dataset import ArrayDataset32, CSRDataset32
 from ..utils._seq_dataset import ArrayDataset64, CSRDataset64
 from ..utils.validation import check_is_fitted, _check_sample_weight
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 # TODO: bayesian_ridge_regression and bayesian_regression_ard
 # should be squashed into its respective objects.
diff --git a/sklearn/linear_model/_coordinate_descent.py b/sklearn/linear_model/_coordinate_descent.py
--- a/sklearn/linear_model/_coordinate_descent.py
+++ b/sklearn/linear_model/_coordinate_descent.py
@@ -14,7 +14,7 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._base import LinearModel, _pre_fit
 from ..base import RegressorMixin, MultiOutputMixin
@@ -30,7 +30,7 @@
     check_is_fitted,
     column_or_1d,
 )
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 # mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
 from . import _cd_fast as cd_fast  # type: ignore
diff --git a/sklearn/linear_model/_least_angle.py b/sklearn/linear_model/_least_angle.py
--- a/sklearn/linear_model/_least_angle.py
+++ b/sklearn/linear_model/_least_angle.py
@@ -16,7 +16,6 @@
 import numpy as np
 from scipy import linalg, interpolate
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel
 
 from ._base import LinearModel, LinearRegression
 from ._base import _deprecate_normalize, _preprocess_data
@@ -28,7 +27,7 @@
 from ..utils._param_validation import Hidden, Interval, StrOptions
 from ..model_selection import check_cv
 from ..exceptions import ConvergenceWarning
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 SOLVE_TRIANGULAR_ARGS = {"check_finite": False}
 
diff --git a/sklearn/linear_model/_logistic.py b/sklearn/linear_model/_logistic.py
--- a/sklearn/linear_model/_logistic.py
+++ b/sklearn/linear_model/_logistic.py
@@ -16,7 +16,7 @@
 
 import numpy as np
 from scipy import optimize
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from sklearn.metrics import get_scorer_names
 
@@ -34,7 +34,7 @@
 from ..utils.optimize import _newton_cg, _check_optimize_result
 from ..utils.validation import check_is_fitted, _check_sample_weight
 from ..utils.multiclass import check_classification_targets
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import StrOptions, Interval
 from ..model_selection import check_cv
 from ..metrics import get_scorer
diff --git a/sklearn/linear_model/_omp.py b/sklearn/linear_model/_omp.py
--- a/sklearn/linear_model/_omp.py
+++ b/sklearn/linear_model/_omp.py
@@ -12,12 +12,11 @@
 import numpy as np
 from scipy import linalg
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel
 
 from ._base import LinearModel, _pre_fit, _deprecate_normalize
 from ..base import RegressorMixin, MultiOutputMixin
 from ..utils import as_float_array, check_array
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils._param_validation import Hidden, Interval, StrOptions
 from ..model_selection import check_cv
 
diff --git a/sklearn/linear_model/_stochastic_gradient.py b/sklearn/linear_model/_stochastic_gradient.py
--- a/sklearn/linear_model/_stochastic_gradient.py
+++ b/sklearn/linear_model/_stochastic_gradient.py
@@ -12,8 +12,6 @@
 from abc import ABCMeta, abstractmethod
 from numbers import Integral, Real
 
-from joblib import Parallel
-
 from ..base import clone, is_classifier
 from ._base import LinearClassifierMixin, SparseCoefMixin
 from ._base import make_dataset
@@ -26,7 +24,7 @@
 from ..utils._param_validation import Interval
 from ..utils._param_validation import StrOptions
 from ..utils._param_validation import Hidden
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..exceptions import ConvergenceWarning
 from ..model_selection import StratifiedShuffleSplit, ShuffleSplit
 
diff --git a/sklearn/linear_model/_theil_sen.py b/sklearn/linear_model/_theil_sen.py
--- a/sklearn/linear_model/_theil_sen.py
+++ b/sklearn/linear_model/_theil_sen.py
@@ -15,13 +15,13 @@
 from scipy import linalg
 from scipy.special import binom
 from scipy.linalg.lapack import get_lapack_funcs
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._base import LinearModel
 from ..base import RegressorMixin
 from ..utils import check_random_state
 from ..utils._param_validation import Interval
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..exceptions import ConvergenceWarning
 
 _EPSILON = np.finfo(np.double).eps
diff --git a/sklearn/manifold/_mds.py b/sklearn/manifold/_mds.py
--- a/sklearn/manifold/_mds.py
+++ b/sklearn/manifold/_mds.py
@@ -8,7 +8,7 @@
 from numbers import Integral, Real
 
 import numpy as np
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 import warnings
 
@@ -17,7 +17,7 @@
 from ..utils import check_random_state, check_array, check_symmetric
 from ..isotonic import IsotonicRegression
 from ..utils._param_validation import Interval, StrOptions, Hidden
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 
 
 def _smacof_single(
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -15,7 +15,7 @@
 from scipy.spatial import distance
 from scipy.sparse import csr_matrix
 from scipy.sparse import issparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from .. import config_context
 from ..utils.validation import _num_samples
@@ -27,7 +27,7 @@
 from ..utils.extmath import row_norms, safe_sparse_dot
 from ..preprocessing import normalize
 from ..utils._mask import _get_mask
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.fixes import sp_version, parse_version
 
 from ._pairwise_distances_reduction import ArgKmin
diff --git a/sklearn/model_selection/_search.py b/sklearn/model_selection/_search.py
--- a/sklearn/model_selection/_search.py
+++ b/sklearn/model_selection/_search.py
@@ -33,14 +33,13 @@
 from ._validation import _normalize_score_results
 from ._validation import _warn_or_raise_about_fit_failures
 from ..exceptions import NotFittedError
-from joblib import Parallel
 from ..utils import check_random_state
 from ..utils.random import sample_without_replacement
 from ..utils._param_validation import HasMethods, Interval, StrOptions
 from ..utils._tags import _safe_tags
 from ..utils.validation import indexable, check_is_fitted, _check_fit_params
 from ..utils.metaestimators import available_if
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..metrics._scorer import _check_multimetric_scoring, get_scorer_names
 from ..metrics import check_scoring
 
diff --git a/sklearn/model_selection/_validation.py b/sklearn/model_selection/_validation.py
--- a/sklearn/model_selection/_validation.py
+++ b/sklearn/model_selection/_validation.py
@@ -21,13 +21,13 @@
 
 import numpy as np
 import scipy.sparse as sp
-from joblib import Parallel, logger
+from joblib import logger
 
 from ..base import is_classifier, clone
 from ..utils import indexable, check_random_state, _safe_indexing
 from ..utils.validation import _check_fit_params
 from ..utils.validation import _num_samples
-from ..utils.fixes import delayed
+from ..utils.parallel import delayed, Parallel
 from ..utils.metaestimators import _safe_split
 from ..metrics import check_scoring
 from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
diff --git a/sklearn/multiclass.py b/sklearn/multiclass.py
--- a/sklearn/multiclass.py
+++ b/sklearn/multiclass.py
@@ -56,9 +56,7 @@
     _ovr_decision_function,
 )
 from .utils.metaestimators import _safe_split, available_if
-from .utils.fixes import delayed
-
-from joblib import Parallel
+from .utils.parallel import delayed, Parallel
 
 __all__ = [
     "OneVsRestClassifier",
diff --git a/sklearn/multioutput.py b/sklearn/multioutput.py
--- a/sklearn/multioutput.py
+++ b/sklearn/multioutput.py
@@ -17,7 +17,6 @@
 
 import numpy as np
 import scipy.sparse as sp
-from joblib import Parallel
 
 from abc import ABCMeta, abstractmethod
 from .base import BaseEstimator, clone, MetaEstimatorMixin
@@ -31,7 +30,7 @@
     has_fit_parameter,
     _check_fit_params,
 )
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .utils._param_validation import HasMethods, StrOptions
 
 __all__ = [
diff --git a/sklearn/neighbors/_base.py b/sklearn/neighbors/_base.py
--- a/sklearn/neighbors/_base.py
+++ b/sklearn/neighbors/_base.py
@@ -16,7 +16,7 @@
 
 import numpy as np
 from scipy.sparse import csr_matrix, issparse
-from joblib import Parallel, effective_n_jobs
+from joblib import effective_n_jobs
 
 from ._ball_tree import BallTree
 from ._kd_tree import KDTree
@@ -37,8 +37,8 @@
 from ..utils.validation import check_is_fitted
 from ..utils.validation import check_non_negative
 from ..utils._param_validation import Interval, StrOptions
-from ..utils.fixes import delayed, sp_version
-from ..utils.fixes import parse_version
+from ..utils.parallel import delayed, Parallel
+from ..utils.fixes import parse_version, sp_version
 from ..exceptions import DataConversionWarning, EfficiencyWarning
 
 VALID_METRICS = dict(
diff --git a/sklearn/pipeline.py b/sklearn/pipeline.py
--- a/sklearn/pipeline.py
+++ b/sklearn/pipeline.py
@@ -14,7 +14,6 @@
 
 import numpy as np
 from scipy import sparse
-from joblib import Parallel
 
 from .base import clone, TransformerMixin
 from .preprocessing import FunctionTransformer
@@ -30,7 +29,7 @@
 from .utils import check_pandas_support
 from .utils._param_validation import HasMethods, Hidden
 from .utils._set_output import _safe_set_output, _get_output_config
-from .utils.fixes import delayed
+from .utils.parallel import delayed, Parallel
 from .exceptions import NotFittedError
 
 from .utils.metaestimators import _BaseComposition
diff --git a/sklearn/utils/fixes.py b/sklearn/utils/fixes.py
--- a/sklearn/utils/fixes.py
+++ b/sklearn/utils/fixes.py
@@ -10,9 +10,7 @@
 #
 # License: BSD 3 clause
 
-from functools import update_wrapper
 from importlib import resources
-import functools
 import sys
 
 import sklearn
@@ -20,7 +18,8 @@
 import scipy
 import scipy.stats
 import threadpoolctl
-from .._config import config_context, get_config
+
+from .deprecation import deprecated
 from ..externals._packaging.version import parse as parse_version
 
 
@@ -106,30 +105,6 @@ def _eigh(*args, **kwargs):
         return scipy.linalg.eigh(*args, eigvals=eigvals, **kwargs)
 
 
-# remove when https://github.com/joblib/joblib/issues/1071 is fixed
-def delayed(function):
-    """Decorator used to capture the arguments of a function."""
-
-    @functools.wraps(function)
-    def delayed_function(*args, **kwargs):
-        return _FuncWrapper(function), args, kwargs
-
-    return delayed_function
-
-
-class _FuncWrapper:
-    """ "Load the global configuration before calling the function."""
-
-    def __init__(self, function):
-        self.function = function
-        self.config = get_config()
-        update_wrapper(self, self.function)
-
-    def __call__(self, *args, **kwargs):
-        with config_context(**self.config):
-            return self.function(*args, **kwargs)
-
-
 # Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
 # `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
 def _percentile(a, q, *, method="linear", **kwargs):
@@ -178,6 +153,16 @@ def threadpool_info():
 threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__
 
 
+@deprecated(
+    "The function `delayed` has been moved from `sklearn.utils.fixes` to "
+    "`sklearn.utils.parallel`. This import path will be removed in 1.5."
+)
+def delayed(function):
+    from sklearn.utils.parallel import delayed
+
+    return delayed(function)
+
+
 # TODO: Remove when SciPy 1.11 is the minimum supported version
 def _mode(a, axis=0):
     if sp_version >= parse_version("1.9.0"):
diff --git a/sklearn/utils/parallel.py b/sklearn/utils/parallel.py
new file mode 100644
--- /dev/null
+++ b/sklearn/utils/parallel.py
@@ -0,0 +1,123 @@
+"""Module that customize joblib tools for scikit-learn usage."""
+
+import functools
+import warnings
+from functools import update_wrapper
+
+import joblib
+
+from .._config import config_context, get_config
+
+
+def _with_config(delayed_func, config):
+    """Helper function that intends to attach a config to a delayed function."""
+    if hasattr(delayed_func, "with_config"):
+        return delayed_func.with_config(config)
+    else:
+        warnings.warn(
+            "`sklearn.utils.parallel.Parallel` needs to be used in "
+            "conjunction with `sklearn.utils.parallel.delayed` instead of "
+            "`joblib.delayed` to correctly propagate the scikit-learn "
+            "configuration to the joblib workers.",
+            UserWarning,
+        )
+        return delayed_func
+
+
+class Parallel(joblib.Parallel):
+    """Tweak of :class:`joblib.Parallel` that propagates the scikit-learn configuration.
+
+    This subclass of :class:`joblib.Parallel` ensures that the active configuration
+    (thread-local) of scikit-learn is propagated to the parallel workers for the
+    duration of the execution of the parallel tasks.
+
+    The API does not change and you can refer to :class:`joblib.Parallel`
+    documentation for more details.
+
+    .. versionadded:: 1.3
+    """
+
+    def __call__(self, iterable):
+        """Dispatch the tasks and return the results.
+
+        Parameters
+        ----------
+        iterable : iterable
+            Iterable containing tuples of (delayed_function, args, kwargs) that should
+            be consumed.
+
+        Returns
+        -------
+        results : list
+            List of results of the tasks.
+        """
+        # Capture the thread-local scikit-learn configuration at the time
+        # Parallel.__call__ is issued since the tasks can be dispatched
+        # in a different thread depending on the backend and on the value of
+        # pre_dispatch and n_jobs.
+        config = get_config()
+        iterable_with_config = (
+            (_with_config(delayed_func, config), args, kwargs)
+            for delayed_func, args, kwargs in iterable
+        )
+        return super().__call__(iterable_with_config)
+
+
+# remove when https://github.com/joblib/joblib/issues/1071 is fixed
+def delayed(function):
+    """Decorator used to capture the arguments of a function.
+
+    This alternative to `joblib.delayed` is meant to be used in conjunction
+    with `sklearn.utils.parallel.Parallel`. The latter captures the the scikit-
+    learn configuration by calling `sklearn.get_config()` in the current
+    thread, prior to dispatching the first task. The captured configuration is
+    then propagated and enabled for the duration of the execution of the
+    delayed function in the joblib workers.
+
+    .. versionchanged:: 1.3
+       `delayed` was moved from `sklearn.utils.fixes` to `sklearn.utils.parallel`
+       in scikit-learn 1.3.
+
+    Parameters
+    ----------
+    function : callable
+        The function to be delayed.
+
+    Returns
+    -------
+    output: tuple
+        Tuple containing the delayed function, the positional arguments, and the
+        keyword arguments.
+    """
+
+    @functools.wraps(function)
+    def delayed_function(*args, **kwargs):
+        return _FuncWrapper(function), args, kwargs
+
+    return delayed_function
+
+
+class _FuncWrapper:
+    """Load the global configuration before calling the function."""
+
+    def __init__(self, function):
+        self.function = function
+        update_wrapper(self, self.function)
+
+    def with_config(self, config):
+        self.config = config
+        return self
+
+    def __call__(self, *args, **kwargs):
+        config = getattr(self, "config", None)
+        if config is None:
+            warnings.warn(
+                "`sklearn.utils.parallel.delayed` should be used with "
+                "`sklearn.utils.parallel.Parallel` to make it possible to propagate "
+                "the scikit-learn configuration of the current thread to the "
+                "joblib workers.",
+                UserWarning,
+            )
+            config = {}
+        with config_context(**config):
+            return self.function(*args, **kwargs)

```

## Test Patch

```diff
diff --git a/sklearn/decomposition/tests/test_dict_learning.py b/sklearn/decomposition/tests/test_dict_learning.py
--- a/sklearn/decomposition/tests/test_dict_learning.py
+++ b/sklearn/decomposition/tests/test_dict_learning.py
@@ -5,8 +5,6 @@
 from functools import partial
 import itertools
 
-from joblib import Parallel
-
 import sklearn
 
 from sklearn.base import clone
@@ -14,6 +12,7 @@
 from sklearn.exceptions import ConvergenceWarning
 
 from sklearn.utils import check_array
+from sklearn.utils.parallel import Parallel
 
 from sklearn.utils._testing import assert_allclose
 from sklearn.utils._testing import assert_array_almost_equal
diff --git a/sklearn/ensemble/tests/test_forest.py b/sklearn/ensemble/tests/test_forest.py
--- a/sklearn/ensemble/tests/test_forest.py
+++ b/sklearn/ensemble/tests/test_forest.py
@@ -18,16 +18,15 @@
 from typing import Dict, Any
 
 import numpy as np
-from joblib import Parallel
 from scipy.sparse import csr_matrix
 from scipy.sparse import csc_matrix
 from scipy.sparse import coo_matrix
 from scipy.special import comb
 
-import pytest
-
 import joblib
 
+import pytest
+
 import sklearn
 from sklearn.dummy import DummyRegressor
 from sklearn.metrics import mean_poisson_deviance
@@ -52,6 +51,7 @@
 from sklearn.model_selection import train_test_split, cross_val_score
 from sklearn.model_selection import GridSearchCV
 from sklearn.svm import LinearSVC
+from sklearn.utils.parallel import Parallel
 from sklearn.utils.validation import check_random_state
 
 from sklearn.metrics import mean_squared_error
diff --git a/sklearn/neighbors/tests/test_kd_tree.py b/sklearn/neighbors/tests/test_kd_tree.py
--- a/sklearn/neighbors/tests/test_kd_tree.py
+++ b/sklearn/neighbors/tests/test_kd_tree.py
@@ -1,7 +1,6 @@
 import numpy as np
 import pytest
-from joblib import Parallel
-from sklearn.utils.fixes import delayed
+from sklearn.utils.parallel import delayed, Parallel
 
 from sklearn.neighbors._kd_tree import KDTree
 
diff --git a/sklearn/tests/test_config.py b/sklearn/tests/test_config.py
--- a/sklearn/tests/test_config.py
+++ b/sklearn/tests/test_config.py
@@ -1,11 +1,10 @@
 import time
 from concurrent.futures import ThreadPoolExecutor
 
-from joblib import Parallel
 import pytest
 
 from sklearn import get_config, set_config, config_context
-from sklearn.utils.fixes import delayed
+from sklearn.utils.parallel import delayed, Parallel
 
 
 def test_config_context():
@@ -120,15 +119,15 @@ def test_config_threadsafe_joblib(backend):
     should be the same as the value passed to the function. In other words,
     it is not influenced by the other job setting assume_finite to True.
     """
-    assume_finites = [False, True]
-    sleep_durations = [0.1, 0.2]
+    assume_finites = [False, True, False, True]
+    sleep_durations = [0.1, 0.2, 0.1, 0.2]
 
     items = Parallel(backend=backend, n_jobs=2)(
         delayed(set_assume_finite)(assume_finite, sleep_dur)
         for assume_finite, sleep_dur in zip(assume_finites, sleep_durations)
     )
 
-    assert items == [False, True]
+    assert items == [False, True, False, True]
 
 
 def test_config_threadsafe():
@@ -136,8 +135,8 @@ def test_config_threadsafe():
     between threads. Same test as `test_config_threadsafe_joblib` but with
     `ThreadPoolExecutor`."""
 
-    assume_finites = [False, True]
-    sleep_durations = [0.1, 0.2]
+    assume_finites = [False, True, False, True]
+    sleep_durations = [0.1, 0.2, 0.1, 0.2]
 
     with ThreadPoolExecutor(max_workers=2) as e:
         items = [
@@ -145,4 +144,4 @@ def test_config_threadsafe():
             for output in e.map(set_assume_finite, assume_finites, sleep_durations)
         ]
 
-    assert items == [False, True]
+    assert items == [False, True, False, True]
diff --git a/sklearn/utils/tests/test_fixes.py b/sklearn/utils/tests/test_fixes.py
--- a/sklearn/utils/tests/test_fixes.py
+++ b/sklearn/utils/tests/test_fixes.py
@@ -11,8 +11,7 @@
 
 from sklearn.utils._testing import assert_array_equal
 
-from sklearn.utils.fixes import _object_dtype_isnan
-from sklearn.utils.fixes import loguniform
+from sklearn.utils.fixes import _object_dtype_isnan, delayed, loguniform
 
 
 @pytest.mark.parametrize("dtype, val", ([object, 1], [object, "a"], [float, 1]))
@@ -46,3 +45,14 @@ def test_loguniform(low, high, base):
     assert loguniform(base**low, base**high).rvs(random_state=0) == loguniform(
         base**low, base**high
     ).rvs(random_state=0)
+
+
+def test_delayed_deprecation():
+    """Check that we issue the FutureWarning regarding the deprecation of delayed."""
+
+    def func(x):
+        return x
+
+    warn_msg = "The function `delayed` has been moved from `sklearn.utils.fixes`"
+    with pytest.warns(FutureWarning, match=warn_msg):
+        delayed(func)
diff --git a/sklearn/utils/tests/test_parallel.py b/sklearn/utils/tests/test_parallel.py
--- a/sklearn/utils/tests/test_parallel.py
+++ b/sklearn/utils/tests/test_parallel.py
@@ -1,10 +1,19 @@
-import pytest
-from joblib import Parallel
+import time
 
+import joblib
+import numpy as np
+import pytest
 from numpy.testing import assert_array_equal
 
-from sklearn._config import config_context, get_config
-from sklearn.utils.fixes import delayed
+from sklearn import config_context, get_config
+from sklearn.compose import make_column_transformer
+from sklearn.datasets import load_iris
+from sklearn.ensemble import RandomForestClassifier
+from sklearn.model_selection import GridSearchCV
+from sklearn.pipeline import make_pipeline
+from sklearn.preprocessing import StandardScaler
+
+from sklearn.utils.parallel import delayed, Parallel
 
 
 def get_working_memory():
@@ -22,3 +31,71 @@ def test_configuration_passes_through_to_joblib(n_jobs, backend):
         )
 
     assert_array_equal(results, [123] * 2)
+
+
+def test_parallel_delayed_warnings():
+    """Informative warnings should be raised when mixing sklearn and joblib API"""
+    # We should issue a warning when one wants to use sklearn.utils.fixes.Parallel
+    # with joblib.delayed. The config will not be propagated to the workers.
+    warn_msg = "`sklearn.utils.parallel.Parallel` needs to be used in conjunction"
+    with pytest.warns(UserWarning, match=warn_msg) as records:
+        Parallel()(joblib.delayed(time.sleep)(0) for _ in range(10))
+    assert len(records) == 10
+
+    # We should issue a warning if one wants to use sklearn.utils.fixes.delayed with
+    # joblib.Parallel
+    warn_msg = (
+        "`sklearn.utils.parallel.delayed` should be used with "
+        "`sklearn.utils.parallel.Parallel` to make it possible to propagate"
+    )
+    with pytest.warns(UserWarning, match=warn_msg) as records:
+        joblib.Parallel()(delayed(time.sleep)(0) for _ in range(10))
+    assert len(records) == 10
+
+
+@pytest.mark.parametrize("n_jobs", [1, 2])
+def test_dispatch_config_parallel(n_jobs):
+    """Check that we properly dispatch the configuration in parallel processing.
+
+    Non-regression test for:
+    https://github.com/scikit-learn/scikit-learn/issues/25239
+    """
+    pd = pytest.importorskip("pandas")
+    iris = load_iris(as_frame=True)
+
+    class TransformerRequiredDataFrame(StandardScaler):
+        def fit(self, X, y=None):
+            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
+            return super().fit(X, y)
+
+        def transform(self, X, y=None):
+            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
+            return super().transform(X, y)
+
+    dropper = make_column_transformer(
+        ("drop", [0]),
+        remainder="passthrough",
+        n_jobs=n_jobs,
+    )
+    param_grid = {"randomforestclassifier__max_depth": [1, 2, 3]}
+    search_cv = GridSearchCV(
+        make_pipeline(
+            dropper,
+            TransformerRequiredDataFrame(),
+            RandomForestClassifier(n_estimators=5, n_jobs=n_jobs),
+        ),
+        param_grid,
+        cv=5,
+        n_jobs=n_jobs,
+        error_score="raise",  # this search should not fail
+    )
+
+    # make sure that `fit` would fail in case we don't request dataframe
+    with pytest.raises(AssertionError, match="X should be a DataFrame"):
+        search_cv.fit(iris.data, iris.target)
+
+    with config_context(transform_output="pandas"):
+        # we expect each intermediate steps to output a DataFrame
+        search_cv.fit(iris.data, iris.target)
+
+    assert not np.isnan(search_cv.cv_results_["mean_test_score"]).any()

```


## Code snippets

### 1 - sklearn/utils/_joblib.py:

Start line: 1, End line: 32

```python
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib imports may raise DeprecationWarning on certain Python
    # versions
    import joblib
    from joblib import logger
    from joblib import dump, load
    from joblib import __version__
    from joblib import effective_n_jobs
    from joblib import hash
    from joblib import cpu_count, Parallel, Memory, delayed
    from joblib import parallel_backend, register_parallel_backend


__all__ = [
    "parallel_backend",
    "register_parallel_backend",
    "cpu_count",
    "Parallel",
    "Memory",
    "delayed",
    "effective_n_jobs",
    "hash",
    "logger",
    "dump",
    "load",
    "joblib",
    "__version__",
]
```
### 2 - sklearn/utils/fixes.py:

Start line: 98, End line: 178

```python
# TODO: remove when the minimum scipy version is >= 1.5
if sp_version >= parse_version("1.5"):
    from scipy.linalg import eigh as _eigh  # noqa
else:

    def _eigh(*args, **kwargs):
        """Wrapper for `scipy.linalg.eigh` that handles the deprecation of `eigvals`."""
        eigvals = kwargs.pop("subset_by_index", None)
        return scipy.linalg.eigh(*args, eigvals=eigvals, **kwargs)


# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs

    return delayed_function


class _FuncWrapper:
    """ "Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)


# Rename the `method` kwarg to `interpolation` for NumPy < 1.22, because
# `interpolation` kwarg was deprecated in favor of `method` in NumPy >= 1.22.
def _percentile(a, q, *, method="linear", **kwargs):
    return np.percentile(a, q, interpolation=method, **kwargs)


if np_version < parse_version("1.22"):
    percentile = _percentile
else:  # >= 1.22
    from numpy import percentile  # type: ignore  # noqa


# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(sklearn, "_sklearn_threadpool_controller"):
        sklearn._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return sklearn._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__
```
### 3 - sklearn/_config.py:

Start line: 162, End line: 295

```python
@contextmanager
def config_context(
    *,
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
    transform_output=None,
):
    """Context manager for global scikit-learn configuration.

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error. If None, the existing value won't change.
        The default value is False.

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. If None, the existing value won't change.
        The default value is 1024.

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()', but would print
        'SVC(C=1.0, cache_size=200, ...)' with all the non-changed parameters
        when False. If None, the existing value won't change.
        The default value is True.

        .. versionchanged:: 0.23
           Default changed from False to True.

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. If None, the existing value won't change.
        The default value is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 1.2

    transform_output : str, default=None
        Configure output of `transform` and `fit_transform`.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        - `"default"`: Default output format of a transformer
        - `"pandas"`: DataFrame output
        - `None`: Transform configuration is unchanged

        .. versionadded:: 1.2

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN...
    """
    old_config = get_config()
    set_config(
        assume_finite=assume_finite,
        working_memory=working_memory,
        print_changed_only=print_changed_only,
        display=display,
        pairwise_dist_chunk_size=pairwise_dist_chunk_size,
        enable_cython_pairwise_dist=enable_cython_pairwise_dist,
        array_api_dispatch=array_api_dispatch,
        transform_output=transform_output,
    )

    try:
        yield
    finally:
        set_config(**old_config)
```
### 4 - sklearn/linear_model/_coordinate_descent.py:

Start line: 8, End line: 36

```python
import sys
import warnings
import numbers
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from joblib import Parallel, effective_n_jobs

from ._base import LinearModel, _pre_fit
from ..base import RegressorMixin, MultiOutputMixin
from ._base import _preprocess_data
from ..utils import check_array, check_scalar
from ..utils.validation import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..model_selection import check_cv
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import (
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)
from ..utils.fixes import delayed

# mypy error: Module 'sklearn.linear_model' has no attribute '_cd_fast'
from . import _cd_fast as cd_fast
```
### 5 - sklearn/_config.py:

Start line: 1, End line: 27

```python
"""Global configuration state and functions for management
"""
import os
from contextlib import contextmanager as contextmanager
import threading

_global_config = {
    "assume_finite": bool(os.environ.get("SKLEARN_ASSUME_FINITE", False)),
    "working_memory": int(os.environ.get("SKLEARN_WORKING_MEMORY", 1024)),
    "print_changed_only": True,
    "display": "diagram",
    "pairwise_dist_chunk_size": int(
        os.environ.get("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", 256)
    ),
    "enable_cython_pairwise_dist": True,
    "array_api_dispatch": False,
    "transform_output": "default",
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config
```
### 6 - sklearn/ensemble/_bagging.py:

Start line: 206, End line: 219

```python
def _parallel_decision_function(estimators, estimators_features, X):
    """Private function used to compute decisions within a job."""
    return sum(
        estimator.decision_function(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


def _parallel_predict_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )
```
### 7 - sklearn/_config.py:

Start line: 48, End line: 142

```python
def set_config(
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
    transform_output=None,
):
    """Set global scikit-learn configuration

    .. versionadded:: 0.19

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.19

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.20

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()' while the default
        behaviour would be to print 'SVC(C=1.0, cache_size=200, ...)' with
        all the non-changed parameters.

        .. versionadded:: 0.21

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. Default is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 1.2

    transform_output : str, default=None
        Configure output of `transform` and `fit_transform`.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        - `"default"`: Default output format of a transformer
        - `"pandas"`: DataFrame output
        - `None`: Transform configuration is unchanged

        .. versionadded:: 1.2

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.
    """
    local_config = _get_threadlocal_config()
    # ... other code
```
### 8 - sklearn/utils/_testing.py:

Start line: 391, End line: 450

```python
try:
    import pytest

    skip_if_32bit = pytest.mark.skipif(_IS_32BIT, reason="skipped on 32bit platforms")
    skip_travis = pytest.mark.skipif(
        os.environ.get("TRAVIS") == "true", reason="skip on travis"
    )
    fails_if_pypy = pytest.mark.xfail(IS_PYPY, reason="not compatible with PyPy")
    fails_if_unstable_openblas = pytest.mark.xfail(
        _in_unstable_openblas_configuration(),
        reason="OpenBLAS is unstable for this configuration",
    )
    skip_if_no_parallel = pytest.mark.skipif(
        not joblib.parallel.mp, reason="joblib is in serial mode"
    )

    #  Decorator for tests involving both BLAS calls and multiprocessing.
    #
    #  Under POSIX (e.g. Linux or OSX), using multiprocessing in conjunction
    #  with some implementation of BLAS (or other libraries that manage an
    #  internal posix thread pool) can cause a crash or a freeze of the Python
    #  process.
    #
    #  In practice all known packaged distributions (from Linux distros or
    #  Anaconda) of BLAS under Linux seems to be safe. So we this problem seems
    #  to only impact OSX users.
    #
    #  This wrapper makes it possible to skip tests that can possibly cause
    #  this crash under OS X with.
    #
    #  Under Python 3.4+ it is possible to use the `forkserver` start method
    #  for multiprocessing to avoid this issue. However it can cause pickling
    #  errors on interactively defined functions. It therefore not enabled by
    #  default.

    if_safe_multiprocessing_with_blas = pytest.mark.skipif(
        sys.platform == "darwin", reason="Possible multi-process bug with some BLAS"
    )
except ImportError:
    pass


def check_skip_network():
    if int(os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", 0)):
        raise SkipTest("Text tutorial requires large dataset download")


def _delete_folder(folder_path, warn=False):
    """Utility function to cleanup a temporary folder if still existing.

    Copy from joblib.pool (for independence).
    """
    try:
        if os.path.exists(folder_path):
            # This can fail under windows,
            #  but will succeed when called by atexit
            shutil.rmtree(folder_path)
    except WindowsError:
        if warn:
            warnings.warn("Could not delete temporary folder %s" % folder_path)
```
### 9 - sklearn/_config.py:

Start line: 144, End line: 159

```python
def set_config(
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
    transform_output=None,
):
    # ... other code

    if assume_finite is not None:
        local_config["assume_finite"] = assume_finite
    if working_memory is not None:
        local_config["working_memory"] = working_memory
    if print_changed_only is not None:
        local_config["print_changed_only"] = print_changed_only
    if display is not None:
        local_config["display"] = display
    if pairwise_dist_chunk_size is not None:
        local_config["pairwise_dist_chunk_size"] = pairwise_dist_chunk_size
    if enable_cython_pairwise_dist is not None:
        local_config["enable_cython_pairwise_dist"] = enable_cython_pairwise_dist
    if array_api_dispatch is not None:
        local_config["array_api_dispatch"] = array_api_dispatch
    if transform_output is not None:
        local_config["transform_output"] = transform_output
```
### 10 - sklearn/exceptions.py:

Start line: 41, End line: 64

```python
class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """
```
### 19 - sklearn/linear_model/_stochastic_gradient.py:

Start line: 9, End line: 58

```python
import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

from joblib import Parallel

from ..base import clone, is_classifier
from ._base import LinearClassifierMixin, SparseCoefMixin
from ._base import make_dataset
from ..base import BaseEstimator, RegressorMixin, OutlierMixin
from ..utils import check_random_state
from ..utils.metaestimators import available_if
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils._param_validation import Interval
from ..utils._param_validation import StrOptions
from ..utils._param_validation import Hidden
from ..utils.fixes import delayed
from ..exceptions import ConvergenceWarning
from ..model_selection import StratifiedShuffleSplit, ShuffleSplit

from ._sgd_fast import _plain_sgd
from ..utils import compute_class_weight
from ._sgd_fast import Hinge
from ._sgd_fast import SquaredHinge
from ._sgd_fast import Log
from ._sgd_fast import ModifiedHuber
from ._sgd_fast import SquaredLoss
from ._sgd_fast import Huber
from ._sgd_fast import EpsilonInsensitive
from ._sgd_fast import SquaredEpsilonInsensitive

LEARNING_RATE_TYPES = {
    "constant": 1,
    "optimal": 2,
    "invscaling": 3,
    "adaptive": 4,
    "pa1": 5,
    "pa2": 6,
}

PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

DEFAULT_EPSILON = 0.1
# Default value of ``epsilon`` parameter.

MAX_INT = np.iinfo(np.int32).max
```
### 20 - sklearn/linear_model/_omp.py:

Start line: 1, End line: 28

```python
"""Orthogonal matching pursuit algorithms
"""

import warnings
from math import sqrt

from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from joblib import Parallel

from ._base import LinearModel, _pre_fit, _deprecate_normalize
from ..base import RegressorMixin, MultiOutputMixin
from ..utils import as_float_array, check_array
from ..utils.fixes import delayed
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..model_selection import check_cv

premature = (
    "Orthogonal matching pursuit ended prematurely due to linear"
    " dependence in the dictionary. The requested precision might"
    " not have been met."
)
```
### 21 - sklearn/ensemble/_bagging.py:

Start line: 154, End line: 178

```python
def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += proba_estimator[
                    :, range(len(estimator.classes_))
                ]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba
```
### 22 - sklearn/ensemble/_bagging.py:

Start line: 73, End line: 151

```python
def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    sample_weight,
    seeds,
    total_n_estimators,
    verbose,
    check_input,
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")
    has_check_input = has_fit_parameter(ensemble.estimator_, "check_input")
    requires_feature_indexing = bootstrap_features or max_features != n_features

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            X_ = X[:, features] if requires_feature_indexing else X
            estimator_fit(X_, y, sample_weight=curr_sample_weight)
        else:
            X_ = X[indices][:, features] if requires_feature_indexing else X[indices]
            estimator_fit(X_, y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features
```
### 25 - sklearn/metrics/pairwise.py:

Start line: 10, End line: 35

```python
import itertools
from functools import partial
import warnings

import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from joblib import Parallel, effective_n_jobs

from .. import config_context
from ..utils.validation import _num_samples
from ..utils.validation import check_non_negative
from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches, get_chunk_n_rows
from ..utils import is_scalar_nan
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..utils._mask import _get_mask
from ..utils.fixes import delayed
from ..utils.fixes import sp_version, parse_version

from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
from ..exceptions import DataConversionWarning
```
### 29 - sklearn/pipeline.py:

Start line: 1230, End line: 1253

```python
class FeatureUnion(TransformerMixin, _BaseComposition):

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return "(step %d of %d) Processing %s" % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        self._validate_transformer_weights()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(
                transformer,
                X,
                y,
                weight,
                message_clsname="FeatureUnion",
                message=self._log_message(name, idx, len(transformers)),
                **fit_params,
            )
            for idx, (name, transformer, weight) in enumerate(transformers, 1)
        )
```
### 30 - sklearn/ensemble/_forest.py:

Start line: 654, End line: 708

```python
def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
            base_estimator=base_estimator,
        )
```
### 33 - sklearn/utils/fixes.py:

Start line: 1, End line: 46

```python
"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""

from functools import update_wrapper
from importlib import resources
import functools
import sys

import sklearn
import numpy as np
import scipy
import scipy.stats
import threadpoolctl
from .._config import config_context, get_config
from ..externals._packaging.version import parse as parse_version


np_version = parse_version(np.__version__)
sp_version = parse_version(scipy.__version__)


if sp_version >= parse_version("1.4"):
    from scipy.sparse.linalg import lobpcg
else:
    # Backport of lobpcg functionality from scipy 1.4.0, can be removed
    # once support for sp_version < parse_version('1.4') is dropped
    # mypy error: Name 'lobpcg' already defined (possibly by an import)
    from ..externals._lobpcg import lobpcg  # type: ignore  # noqa

try:
    from scipy.optimize._linesearch import line_search_wolfe2, line_search_wolfe1
except ImportError:  # SciPy < 1.8
    from scipy.optimize.linesearch import line_search_wolfe2, line_search_wolfe1  # type: ignore  # noqa


def _object_dtype_isnan(X):
    return X != X
```
### 34 - sklearn/ensemble/_bagging.py:

Start line: 181, End line: 203

```python
def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute log probabilities within a job."""
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    all_classes = np.arange(n_classes, dtype=int)

    for estimator, features in zip(estimators, estimators_features):
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))],
            )

            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing], -np.inf)

    return log_proba
```
