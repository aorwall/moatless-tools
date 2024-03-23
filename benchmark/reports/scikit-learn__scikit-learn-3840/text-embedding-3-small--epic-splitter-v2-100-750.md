**0 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py**:
```python
"""Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default='deprecated')
        Whether to sort x before computing. If False, assume that x must be
        either monotonic increasing or monotonic decreasing. If True, y is
        used to break ties when sorting x. Make sure that y has a monotonic
        relation to x when setting reorder to True.

        .. deprecated:: 0.20
           Parameter ``reorder`` has been deprecated in version 0.20 and will
           be removed in 0.22. It's introduced for roc_auc_score (not for
           general use) and is no longer used there. What's more, the result
           from auc will be significantly influenced if x is sorted
           unexpectedly due to slight floating point error (See issue #9786).
           Future (and default) behavior is equivalent to ``reorder=False``.

    Returns
    -------
    auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve
    average_precision_score : Compute average precision from prediction scores
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    check_consistent_length(x, y)
    x = column_or_1d(x)
    y = column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    if reorder != 'deprecated':
        warnings.warn("The 'reorder' parameter has been deprecated in "
                      "version 0.20 and will be removed in 0.22. It is "
                      "recommended not to set 'reorder' and ensure that x "
                      "is monotonic increasing or monotonic decreasing.",
                      DeprecationWarning)

    direction = 1
    if reorder is True:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("x is neither increasing nor decreasing "
                                 ": {}.".format(x))

    area = direction * np.trapz(y, x)
    
```
