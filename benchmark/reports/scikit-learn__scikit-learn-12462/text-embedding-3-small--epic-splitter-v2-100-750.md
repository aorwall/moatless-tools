0 - /tmp/repos/scikit-learn/sklearn/metrics/regression.py
```python
y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)
# ...
```
1 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'calinski_harabasz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'davies_bouldin_score',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'max_error',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_distances_chunked',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
2 - /tmp/repos/scikit-learn/sklearn/metrics/regression.py
```python
check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)
# ...
```
3 - /tmp/repos/scikit-learn/sklearn/metrics/regression.py
```python
check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)
```
4 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'calinski_harabasz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'davies_bouldin_score',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'max_error',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'multilabel_confusion_matrix',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_chunked',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
5 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'davies_bouldin_score',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'max_error',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_distances_chunked',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
6 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'davies_bouldin_score',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_distances_chunked',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
7 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
8 - /tmp/repos/scikit-learn/sklearn/metrics/__init__.py
```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]

```
**9 - /tmp/repos/scikit-learn/sklearn/utils/validation.py**:
```python
if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    
```
