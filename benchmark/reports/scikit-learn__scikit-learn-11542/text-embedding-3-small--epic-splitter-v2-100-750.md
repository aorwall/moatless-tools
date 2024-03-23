**0 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestRegressor(ForestRegressor):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        `None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
    >>> regr.fit(X, y)
    RandomForestRegressor(max_depth=2, random_state=0)
    >>> print(regr.feature_importances_)
    [0.18146984 0.81473937 0.00145312 0.00233767]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-8.32987858]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor
    
```
**1 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.


    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or 
```
**2 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestRegressor(ForestRegressor):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
    >>> regr.fit(X, y)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(regr.feature_importances_)
    [ 0.17339552  0.81594114  0.          0.01066333]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-2.50699856]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor
    
```
**3 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestRegressor(ForestRegressor):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomForestRegressor(max_depth=2, random_state=0)
    >>> regr.fit(X, y)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(regr.feature_importances_)
    [ 0.17339552  0.81594114  0.          0.01066333]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-2.50699856]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor
    
```
**4 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(clf.feature_importances_)
    [ 0.17287856  0.80608704  0.01884792  0.00218648]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    
```
**5 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(clf.feature_importances_)
    [ 0.17287856  0.80608704  0.01884792  0.00218648]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    
```
**6 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomTreesEmbedding(BaseForest):
    """An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        Number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    max_depth : integer, optional (default=5)
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    sparse_output : bool, optional (default=True)
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    
```
**7 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomTreesEmbedding(BaseForest):
    """An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        Number of trees in the forest.

    max_depth : integer, optional (default=5)
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    sparse_output : bool, optional (default=True)
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    
```
**8 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class RandomTreesEmbedding(BaseForest):
    """An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        Number of trees in the forest.

    max_depth : integer, optional (default=5)
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    sparse_output : bool, optional (default=True)
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007

    
```
**9 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
\
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    >>> print(clf.feature_importances_)
    [0.17287856 0.80608704 0.01884792 0.00218648]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    
```
**10 - /tmp/repos/scikit-learn/examples/ensemble/plot_ensemble_oob.py**:
```python
"""
=============================
OOB Errors for Random Forests
=============================

The ``RandomForestClassifier`` is trained using *bootstrap aggregation*, where
each new tree is fit from a bootstrap sample of the training observations
:math:`z_i = (x_i, y_i)`. The *out-of-bag* (OOB) error is the average error for
each :math:`z_i` calculated using predictions from the trees that do not
contain :math:`z_i` in their respective bootstrap sample. This allows the
``RandomForestClassifier`` to be fit and validated whilst being trained [1]_.

The example below demonstrates how the OOB error can be measured at the
addition of each new tree during training. The resulting plot allows a
practitioner to approximate a suitable value of ``n_estimators`` at which the
error stabilizes.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
       Learning Ed. 2", p592-593, Springer, 2009.

"""
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

print(__doc__)

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 175

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
```
**11 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**12 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**13 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
\
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification

    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)
    RandomForestClassifier(max_depth=2, random_state=0)
    >>> print(clf.feature_importances_)
    [0.14205973 0.76664038 0.0282433  0.06305659]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    
```
**14 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or 
```
**15 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**16 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**17 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesRegressor(ForestRegressor):
    """An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.
    
```
**18 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesRegressor(ForestRegressor):
    """An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.
    
```
**19 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesRegressor(ForestRegressor):
    """An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.
    
```
20 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="random")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.


    See also
    --------
    ExtraTreeClassifier, sklearn.ensemble.ExtraTreesClassifier,
    sklearn.ensemble.ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
**21 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    
```
**22 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**23 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...


class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    
```
24 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="random")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.


    See also
    --------
    ExtraTreeClassifier, sklearn.ensemble.ExtraTreesClassifier,
    sklearn.ensemble.ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
**25 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""Forest of trees-based ensemble methods

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.

"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# License: BSD 3 clause

from __future__ import division

import warnings
from warnings import warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack


from ..base import ClassifierMixin, RegressorMixin
from ..externals.joblib import Parallel, delayed
from ..externals import six
from ..metrics import r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from ..tree._tree import DTYPE, DOUBLE
from ..utils import check_random_state, check_array, compute_sample_weight
from ..exceptions import DataConversionWarning, NotFittedError
from .base import BaseEnsemble, _partition_estimators
from ..utils.fixes import parallel_helper
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted

__all__ = ["RandomForestClassifier",
           "RandomForestRegressor",
           "ExtraTreesClassifier",
           "ExtraTreesRegressor",
           "RandomTreesEmbedding"]

MAX_INT = np.iinfo(np.int32).max
```
26 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.


    See also
    --------
    ExtraTreeClassifier, ExtraTreesClassifier, ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
**27 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
28 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.


    See also
    --------
    ExtraTreeClassifier, ExtraTreesClassifier, ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
**29 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**30 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**31 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
def set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    if ("n_iter" in params and estimator.__class__.__name__ != "TSNE"
            and not isinstance(estimator, BaseSGD)):
        estimator.set_params(n_iter=5)
    if "max_iter" in params:
        warnings.simplefilter("ignore", ConvergenceWarning)
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        # LinearSVR, LinearSVC
        if estimator.__class__.__name__ in ['LinearSVR', 'LinearSVC']:
            estimator.set_params(max_iter=20)
        # NMF
        if estimator.__class__.__name__ == 'NMF':
            estimator.set_params(max_iter=100)
        # MLP
        if estimator.__class__.__name__ in ['MLPClassifier', 'MLPRegressor']:
            estimator.set_params(max_iter=100)
    if "n_resampling" in params:
        # randomized lasso
        estimator.set_params(n_resampling=5)
    if "n_estimators" in params:
        # especially gradient boosting with default 100
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if "max_trials" in params:
        # RANSAC
        estimator.set_params(max_trials=10)
    if "n_init" in params:
        # K-Means
        estimator.set_params(n_init=2)
    if "decision_function_shape" in params:
        # SVC
        estimator.set_params(decision_function_shape='ovo')

    if estimator.__class__.__name__ == "SelectFdr":
        # be tolerant of noisy datasets (not actually speed)
        estimator.set_params(alpha=.5)

    if estimator.__class__.__name__ == "TheilSenRegressor":
        estimator.max_subpopulation = 100

    if isinstance(estimator, BaseRandomProjection):
        # Due to the jl lemma and often very few samples, the number
        # of components of the random matrix projection will be probably
        # greater than the number of features.
        # So we impose a smaller number (avoid "auto" mode)
        estimator.set_params(n_components=2)

    if isinstance(estimator, SelectKBest):
        # SelectKBest has a default of k=10
        # which is more feature than we have in most case.
        estimator.set_params(k=1)
```
32 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
"""

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="legacy",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        if contamination == "legacy":
            warnings.warn('default contamination parameter 0.1 will change '
                          'in version 0.22 to "auto". This will change the '
                          'predict method behavior.',
                          DeprecationWarning)
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    
```
33 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
"""

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="legacy",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        if contamination == "legacy":
            warnings.warn('default contamination parameter 0.1 will change '
                          'in version 0.22 to "auto". This will change the '
                          'predict method behavior.',
                          DeprecationWarning)
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    
```
34 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
"""

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="legacy",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        if contamination == "legacy":
            warnings.warn('default contamination parameter 0.1 will change '
                          'in version 0.22 to "auto". This will change the '
                          'predict method behavior.',
                          DeprecationWarning)
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    
```
35 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="random")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    See also
    --------
    ExtraTreeRegressor, sklearn.ensemble.ExtraTreesClassifier,
    sklearn.ensemble.ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
36 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    See also
    --------
    ExtraTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
**37 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesRegressor, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**38 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesRegressor, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
39 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    See also
    --------
    ExtraTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
40 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="random")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    See also
    --------
    ExtraTreeRegressor, sklearn.ensemble.ExtraTreesClassifier,
    sklearn.ensemble.ExtraTreesRegressor

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    
```
41 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
"""

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="auto",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=None,
                 behaviour='deprecated',
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.behaviour = behaviour
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies.
        return _joblib_parallel_args(prefer='threads')

    
```
42 - /tmp/repos/scikit-learn/benchmarks/bench_covertype.py
```python
# ..."""
===========================
Covertype dataset benchmark
===========================

Benchmark stochastic gradient descent (SGD), Liblinear, and Naive Bayes, CART
(decision tree), RandomForest and Extra-Trees on the forest covertype dataset
of Blackard, Jock, and Dean [1]. The dataset comprises 581,012 samples. It is
low dimensional with 54 features and a sparsity of approx. 23%. Here, we
consider the task of predicting class 1 (spruce/fir). The classification
performance of SGD is competitive with Liblinear while being two orders of
magnitude faster to train::

    [..]
    Classification performance:
    ===========================
    Classifier   train-time test-time error-rate
    --------------------------------------------
    liblinear     15.9744s    0.0705s     0.2305
    GaussianNB    3.0666s     0.3884s     0.4841
    SGD           1.0558s     0.1152s     0.2300
    CART          79.4296s    0.0523s     0.0469
    RandomForest  1190.1620s  0.5881s     0.0243
    ExtraTrees    640.3194s   0.6495s     0.0198

The same task has been used in a number of papers including:

 * `"SVM Optimization: Inverse Dependence on Training Set Size"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.139.2112>`_
   S. Shalev-Shwartz, N. Srebro - In Proceedings of ICML '08.

 * `"Pegasos: Primal estimated sub-gradient solver for svm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513>`_
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.

 * `"Training Linear SVMs in Linear Time"
   <www.cs.cornell.edu/People/tj/publications/joachims_06a.pdf>`_
   T. Joachims - In SIGKDD '06

[1] http://archive.ics.uci.edu/ml/datasets/Covertype


```
43 - /tmp/repos/scikit-learn/benchmarks/bench_20newsgroups.py
```python
# ...from __future__ import print_function, division
from time import time
import argparse
import numpy as np

from sklearn.dummy import DummyClassifier

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ESTIMATORS = {
    "dummy": DummyClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100,
                                            max_features="sqrt",
                                            min_samples_split=10),
    "extra_trees": ExtraTreesClassifier(n_estimators=100,
                                        max_features="sqrt",
                                        min_samples_split=10),
    "logistic_regression": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "adaboost": AdaBoostClassifier(n_estimators=10),
}


###############################################################################
# Data


```
**44 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
45 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           optional parameter *presort*.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20


    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : BaseEstimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
46 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python
# ...from time import time
import argparse

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.ensemble._hist_gradient_boosting.utils import (
    get_equivalent_estimator)


parser = argparse.ArgumentParser()
parser.add_argument('--n-leaf-nodes', type=int, default=31)
parser.add_argument('--n-trees', type=int, default=10)
parser.add_argument('--lightgbm', action="store_true", default=False,
                    help='also plot lightgbm')
parser.add_argument('--xgboost', action="store_true", default=False,
                    help='also plot xgboost')
parser.add_argument('--catboost', action="store_true", default=False,
                    help='also plot catboost')
parser.add_argument('--learning-rate', type=float, default=.1)
parser.add_argument('--problem', type=str, default='classification',
                    choices=['classification', 'regression'])
parser.add_argument('--n-classes', type=int, default=2)
parser.add_argument('--n-samples-max', type=int, default=int(1e6))
parser.add_argument('--n-features', type=int, default=20)
parser.add_argument('--max-bins', type=int, default=255)
args = parser.parse_args()

n_leaf_nodes = args.n_leaf_nodes
n_trees = args.n_trees
lr = args.learning_rate
max_bins = args.max_bins


def get_estimator_and_data():
    if args.problem == 'classification':
        X, y = make_classification(args.n_samples_max * 2,
                                   n_features=args.n_features,
                                   n_classes=args.n_classes,
                                   n_clusters_per_class=1,
                                   random_state=0)
        return X, y, HistGradientBoostingClassifier
    elif args.problem == 'regression':
        X, y = make_regression(args.n_samples_max * 2,
                               n_features=args.n_features, random_state=0)
        return X, y, HistGradientBoostingRegressor


X, y, Estimator = get_estimator_and_data()
X_train_, X_test_, y_train_, y_test_ = train_test_split(
    X, y, test_size=0.5, random_state=0)


def one_run(n_samples):
    
```
47 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    init : estimator, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           optional parameter *presort*.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20


    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, 1]
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
**48 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):
    """Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ForestRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
```
**49 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
\
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    """
    
```
**50 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
\
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    """
    
```
51 - /tmp/repos/scikit-learn/examples/model_selection/plot_randomized_search.py
```python
"""
=========================================================================
Comparing randomized search and grid search for hyperparameter estimation
=========================================================================

Compare randomized search and grid search for optimizing hyperparameters of a
random forest.
All parameters that influence the learning are searched simultaneously
(except for the number of estimators, which poses a time / quality tradeoff).

The randomized search and the grid search explore exactly the same space of
parameters. The result in parameter settings is quite similar, while the run
time for randomized search is drastically lower.

The performance is slightly worse for the randomized search, though this
is most likely a noise effect and would not carry over to a held-out test set.

Note that in practice, one would not search over this many different parameters
simultaneously using grid search, but pick only the ones deemed most important.
"""
print(__doc__)

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)



# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
```
52 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide `fit` and `predict`. If 'zero', the initial
        raw predictions are set to zero. By default a ``DummyEstimator`` is
        used, predicting either the average target value (for loss='ls'), or
        a quantile for the other losses.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           optional parameter *presort*.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20


    Attributes
    ----------
    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : array of DecisionTreeRegressor, shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.ensemble.HistGradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
**53 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
def _generate_unsampled_indices(random_state, n_samples):
    """Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices
```
**54 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
def check_parameters_default_constructible(name, Estimator):
    # this check works on classes, not instances
    classifier = LinearDiscriminantAnalysis()
    # test default-constructibility
    # get rid of deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        if name in META_ESTIMATORS:
            estimator = Estimator(classifier)
        else:
            estimator = Estimator()
        # test cloning
        clone(estimator)
        # test __repr__
        repr(estimator)
        # test that set_params returns self
        assert_true(estimator.set_params() is estimator)

        # test if init does nothing but set parameters
        # this is important for grid_search etc.
        # We get the default parameters from init and then
        # compare these against the actual values of the attributes.

        # this comes from getattr. Gets rid of deprecation decorator.
        init = getattr(estimator.__init__, 'deprecated_original',
                       estimator.__init__)

        try:
            def param_filter(p):
                """Identify hyper parameters of an estimator"""
                return (p.name != 'self' and
                        p.kind != p.VAR_KEYWORD and
                        p.kind != p.VAR_POSITIONAL)

            init_params = [p for p in signature(init).parameters.values()
                           if param_filter(p)]
        except (TypeError, ValueError):
            # init is not a python function.
            # true for mixins
            return
        params = estimator.get_params()
        if name in META_ESTIMATORS:
            # they can need a non-default argument
            init_params = init_params[1:]

        for init_param in init_params:
            assert_not_equal(init_param.default, init_param.empty,
                             "parameter %s for %s has no default value"
                             % (init_param.name, type(estimator).__name__))
            assert_in(type(init_param.default),
                      [str, int, float, bool, tuple, type(None),
                       np.float64, types.FunctionType, Memory])
            if init_param.name not in params.keys():
                # deprecated parameter, not in get_params
                assert_true(init_param.default is None)
                continue

            if (issubclass(Estimator, BaseSGD) and
                    init_param.name in ['tol', 'max_iter']):
                # To remove in 0.21, when they get their future default values
                continue

            param_value = params[init_param.name]
            if isinstance(param_value, np.ndarray):
                assert_array_equal(param_value, init_param.default)
            else:
                assert_equal(param_value, init_param.default, init_param.name)
```
55 - /tmp/repos/scikit-learn/benchmarks/bench_tree.py
```python
"""
To run this, you'll need to have installed.

  * scikit-learn

Does two benchmarks

First, we fix a training set, increase the number of
samples to classify and plot number of classified samples as a
function of time.

In the second benchmark, we increase the number of dimensions of the
training set, classify a sample and plot the time taken as a function
of the number of dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10 ** 6



if __name__ == '__main__':

    print('============================================')
    print('Warning: this is going to take a looong time')
    print('============================================')

    n = 10
    step = 10000
    n_samples = 10000
    dim = 10
    n_classes = 10
    for i in range(n):
        print('============================================')
        print('Entering iteration %s of %s' % (i, n))
        print('============================================')
        n_samples += step
        X = np.random.randn(n_samples, dim)
        Y = np.random.randint(0, n_classes, (n_samples,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(n_samples)
        bench_scikit_tree_regressor(X, Y)

    xx = range(0, n * step, step)
    plt.figure('scikit-learn tree benchmark results')
    plt.subplot(211)
    plt.title('Learning with varying number of samples')
    plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
    plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
    plt.legend(loc='upper left')
    plt.xlabel('number of samples')
    plt.ylabel('Time (s)')

    scikit_classifier_results = []
    scikit_regressor_results = []
    n = 10
    step = 500
    start_dim = 500
    n_classes = 10

    dim = start_dim
    for i in range(0, n):
        print('============================================')
        print('Entering iteration %s of %s' % (i, n))
        print('============================================')
        dim += step
        X = np.random.randn(100, dim)
        Y = np.random.randint(0, n_classes, (100,))
        bench_scikit_tree_classifier(X, Y)
        Y = np.random.randn(100)
        bench_scikit_tree_regressor(X, Y)

    xx = np.arange(start_dim, start_dim + n * step, step)
    plt.subplot(212)
    plt.title('Learning in high dimensional spaces')
    plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
    plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
    plt.legend(loc='upper left')
    plt.xlabel('number of dimensions')
    plt.ylabel('Time (s)')
    plt.axis('tight')
    plt.show()
```
56 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
# ...


class HistGradientBoostingRegressor(BaseHistGradientBoosting, RegressorMixin):
    """Histogram-based Gradient Boosting Regression Tree.

    This estimator is much faster than
    :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
    for big datasets (n_samples >= 10 000). The input data ``X`` is pre-binned
    into integer-valued bins, which considerably reduces the number of
    splitting points to consider, and allows the algorithm to leverage
    integer-based data structures. For small sample sizes,
    :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
    might be preferred since binning may lead to split points that are too
    approximate in this setting.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_hist_gradient_boosting``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        >>> # now you can import normally from ensemble
        >>> from sklearn.ensemble import HistGradientBoostingClassifier


    Parameters
    ----------
    loss : {'least_squares'}, optional (default='least_squares')
        The loss function to use in the boosting process. Note that the
        "least squares" loss actually implements an "half least squares loss"
        to simplify the computation of the gradient.
    learning_rate : float, optional (default=0.1)
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, optional (default=100)
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int or None, optional (default=31)
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, optional (default=None)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf. Must be strictly greater
        than 1. Depth isn't constrained by default.
    min_samples_leaf : int, optional (default=20)
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, optional (default=0)
        The L2 regularization parameter. Use ``0`` for no regularization
        (default).
    max_bins : int, optional (default=256)
        The maximum number of bins to use. Before training, each feature of
        the input array ``X`` is binned into at most ``max_bins`` bins, which
        allows for a much faster training stage. Features with a small
        number of unique values may use less than ``max_bins`` bins. Must be no
        larger than 256.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    scoring : str or callable or None, optional (default=None)
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used. If
        ``scoring='loss'``, early stopping is checked w.r.t the loss value.
        Only used if ``n_iter_no_change`` is not None.
    validation_fraction : int or float or None, optional (default=0.1)
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if ``n_iter_no_change`` is not None.
    n_iter_no_change : int or None, optional (default=None)
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. If None or 0, no early-stopping is done.
    tol : float or None, optional (default=1e-7)
        The absolute tolerance to use when comparing scores during early
        stopping. The higher the tolerance, the more likely we are to early
        stop: higher tolerance means that it will be harder for subsequent
        iterations to be considered an improvement upon the reference score.
    verbose: int, optional (default=0)
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, np.random.RandomStateInstance or None, 
```
57 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    
```
58 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    
```
59 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    
```
**60 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
**61 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
# ...
```
62 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    
```
**63 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(BaseForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
```
64 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : BaseEstimator, None, optional (default=None)
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : BaseEstimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
65 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# ...


class IsolationForest(BaseBagging):
    """Isolation Forest Algorithm

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    
```
66 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# ...


class IsolationForest(BaseBagging, OutlierMixin):
    """Isolation Forest Algorithm

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, optional (default='auto')
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
            original paper.
            - If float, the contamination should be in the range [0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    behaviour : str, default='deprecated'
        This parameter has not effect, is deprecated, and will be removed.

        .. versionadded:: 0.20
           ``behaviour`` is added in 0.20 for back-compatibility purpose.

        .. deprecated:: 0.20
           ``behaviour='old'`` is deprecated in 0.20 and will not be possible
           in 0.22.

        .. deprecated:: 0.22
           ``behaviour`` parameter is deprecated in 0.22 and removed in
           0.24.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    
```
67 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator, optional
        An estimator object that is used to compute the initial
        predictions. ``init`` has to provide ``fit`` and ``predict``.
        If None it uses ``loss.init_estimator``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape = [n_estimators]
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape = [n_estimators]
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    
```
68 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
class IsolationForest(BaseBagging, OutlierMixin):


    @property
    def threshold_(self):
        warnings.warn("threshold_ attribute is deprecated in 0.20 and will"
                      " be removed in 0.22.", DeprecationWarning)
        if self.contamination == 'auto':
            return self._threshold_
        return self.offset_
```
69 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
# ...


class HistGradientBoostingClassifier(BaseHistGradientBoosting,
                                     ClassifierMixin):
    """Histogram-based Gradient Boosting Classification Tree.

    This estimator is much faster than
    :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
    for big datasets (n_samples >= 10 000). The input data ``X`` is pre-binned
    into integer-valued bins, which considerably reduces the number of
    splitting points to consider, and allows the algorithm to leverage
    integer-based data structures. For small sample sizes,
    :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
    might be preferred since binning may lead to split points that are too
    approximate in this setting.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import ``enable_hist_gradient_boosting``::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        >>> # now you can import normally from ensemble
        >>> from sklearn.ensemble import HistGradientBoostingClassifier

    Parameters
    ----------
    loss : {'auto', 'binary_crossentropy', 'categorical_crossentropy'}, \
            optional (default='auto')
        The loss function to use in the boosting process. 'binary_crossentropy'
        (also known as logistic loss) is used for binary classification and
        generalizes to 'categorical_crossentropy' for multiclass
        classification. 'auto' will automatically choose either loss depending
        on the nature of the problem.
    learning_rate : float, optional (default=0.1)
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, optional (default=100)
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    max_leaf_nodes : int or None, optional (default=31)
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, optional (default=None)
        The maximum depth of each tree. The depth of a tree is the number of
        nodes to go from the root to the deepest leaf. Must be strictly greater
        than 1. Depth isn't constrained by default.
    min_samples_leaf : int, optional (default=20)
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, optional (default=0)
        The L2 regularization parameter. Use 0 for no regularization.
    max_bins : int, optional (default=256)
        The maximum number of bins to use. Before training, each feature of
        the input array ``X`` is binned into at most ``max_bins`` bins, which
        allows for a much faster training stage. Features with a small
        number of unique values may use less than ``max_bins`` bins. Must be no
        larger than 256.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    scoring : str or callable or None, optional (default=None)
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer
        is used. If ``scoring='loss'``, early stopping is checked
        w.r.t the loss value. Only used if ``n_iter_no_change`` is not None.
    validation_fraction : int or float or None, optional (default=0.1)
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data.
    n_iter_no_change : int or None, optional (default=None)
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. If None or 0, no early-stopping is done.
    tol : float or None, optional (default=1e-7)
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    verbose: int, optional (default=0)
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, np.random.RandomStateInstance or None, 
```
**70 - /tmp/repos/scikit-learn/examples/ensemble/plot_random_forest_regression_multioutput.py**:
```python
"""
============================================================
Comparing random forests and the multi-output meta estimator
============================================================

An example to compare multi-output regression with random forest and
the :ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator.

This example illustrates the use of the
:ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator
to perform multi-output regression. A random forest regressor is used,
which supports multi-output regression natively, so the results can be
compared.

The random forest regressor will only ever predict values within the
range of observations or closer to zero for each of the targets. As a
result the predictions are biased towards the centre of the circle.

Using a single underlying feature the model learns both the
x and y coordinate as output.

"""
print(__doc__)

# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=400,
                                                    random_state=4)

max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
            c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
```
71 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
"""

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, six.string_types):
            if self.max_samples == 'auto':
                max_samples = min(256, n_samples)
            else:
                raise ValueError('max_samples (%s) is not supported.'
                                 'Valid choices are: "auto", int or'
                                 'float' % self.max_samples)

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super(IsolationForest, self)._fit(X, y, max_samples,
                                          max_depth=max_depth,
                                          sample_weight=sample_weight)

        self.threshold_ = -sp.stats.scoreatpercentile(
            -self.decision_function(X), 100. * (1. - self.contamination))

        return self

    
```
**72 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):
    """Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(ForestClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)
```
**73 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=9)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert_equal(hash(new_value), hash(original_value),
                     "Estimator %s should not change or mutate "
                     " the parameter %s from %s to %s during fit."
                     % (name, param_name, original_value, new_value))
```
**74 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class ForestRegressor(six.with_metaclass(ABCMeta, BaseForest, RegressorMixin)):


    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples, ))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_
```
75 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

from __future__ import division

import numpy as np
import scipy as sp
import warnings
from warnings import warn
from sklearn.utils.fixes import euler_gamma

from scipy.sparse import issparse

import numbers
from ..externals import six
from ..tree import ExtraTreeRegressor
from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..base import OutlierMixin

from .bagging import BaseBagging

__all__ = ["IsolationForest"]

INTEGER_TYPES = (numbers.Integral, np.integer)
```
76 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# ...


class IsolationForest(BaseBagging):
    """Isolation Forest Algorithm

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function. If 'auto', the decision function threshold is
        determined as in the original paper.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - offset_.
        The offset is set to 0.5 as in the original paper, except when a
        contamination parameter different than "auto" is provided. In that
        case, the offset is defined in such a way we obtain the expected
        number of outliers (samples with decision function < 0) in training.

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    
```
77 - /tmp/repos/scikit-learn/examples/ensemble/plot_forest_importances.py
```python
"""
=========================================
Feature importances with forests of trees
=========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artificial classification task. The red bars are the feature
importances of the forest, along with their inter-trees variability.

As expected, the plot suggests that 3 features are informative, while the
remaining are not.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
```
78 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
# ...


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide `fit` and `predict_proba`. If 'zero', the
        initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor,
```
79 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    
```
80 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python
def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.

    Parameters
    ----------
    estimator : object
        The estimator
    random_state : int, RandomState instance or None, optional, default=0
        Pseudo random number generator state.  If int, random_state is the seed
        used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


try:
    import pytest

    skip_if_32bit = pytest.mark.skipif(_IS_32BIT,
                                       reason='skipped on 32bit platforms')
    skip_travis = pytest.mark.skipif(os.environ.get('TRAVIS') == 'true',
                                     reason='skip on travis')
    fails_if_pypy = pytest.mark.xfail(IS_PYPY, raises=NotImplementedError,
                                      reason='not compatible with PyPy')
    skip_if_no_parallel = pytest.mark.skipif(not joblib.parallel.mp,
                                             reason="joblib is in serial mode")

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
            sys.platform == 'darwin',
            reason="Possible multi-process bug with some BLAS")
except ImportError:
    pass


def clean_warning_registry():
    """Clean Python warning registry for easier testing of warning messages.

    When changing warning filters this function is not necessary with
    Python3.5+, as __warningregistry__ will be re-set internally.
    See https://bugs.python.org/issue4180 and
    https://bugs.python.org/issue21724 for more details.

    """
    for mod in sys.modules.values():
        registry = getattr(mod, "__warningregistry__", None)
        if registry is not None:
            registry.clear()


def check_skip_network():
    if int(os.environ.get('SKLEARN_SKIP_NETWORK_TESTS', 0)):
        raise SkipTest("Text tutorial requires large dataset download")



```
**81 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=DeprecationWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    rnd = np.random.RandomState(0)
    X_train_finite = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                                  estimator_orig)
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    error_string_fit = "Estimator doesn't check for NaN and inf in fit."
    error_string_predict = ("Estimator doesn't check for NaN and inf in"
                            " predict.")
    error_string_transform = ("Estimator doesn't check for NaN and inf in"
                              " transform.")
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            try:
                estimator.fit(X_train, y)
            except ValueError as e:
                if 'inf' not in repr(e) and 'NaN' not in repr(e):
                    print(error_string_fit, estimator, e)
                    traceback.print_exc(file=sys.stdout)
                    raise e
            except Exception as exc:
                print(error_string_fit, estimator, exc)
                traceback.print_exc(file=sys.stdout)
                raise exc
            else:
                raise AssertionError(error_string_fit, estimator)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                try:
                    estimator.predict(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_predict, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_predict, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_predict, estimator)

            # transform
            if hasattr(estimator, "transform"):
                try:
                    estimator.transform(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_transform, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_transform, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_transform, estimator)
```
**82 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class RandomTreesEmbedding(BaseForest):


    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")
```
83 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
# ...


class IsolationForest(BaseBagging, OutlierMixin):
    """Isolation Forest Algorithm

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function. If 'auto', the decision function threshold is
        determined as in the original paper.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - offset_.
        When the contamination parameter is set to "auto", the offset is equal
        to -0.5 as the scores of inliers are close to 0 and the scores of
        outliers are close to -1. When a contamination parameter different
        than "auto" is provided, the offset is defined in such a way we obtain
        the expected number of outliers (samples with decision function < 0)
        in training.

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    
```
84 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
# ...


# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    
```
85 - /tmp/repos/scikit-learn/sklearn/ensemble/_gb_losses.py
```python
"""Losses and corresponding default initial estimators for gradient boosting
decision trees.
"""

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from scipy.special import expit

from ..tree._tree import TREE_LEAF
from ..utils.fixes import logsumexp
from ..utils.stats import _weighted_percentile
from ..dummy import DummyClassifier
from ..dummy import DummyRegressor



LOSS_FUNCTIONS = {
    'ls': LeastSquaresError,
    'lad': LeastAbsoluteError,
    'huber': HuberLossFunction,
    'quantile': QuantileLossFunction,
    'deviance': None,  # for both, multinomial and binomial
    'exponential': ExponentialLoss,
}
```
**86 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices
```
87 - /tmp/repos/scikit-learn/examples/ensemble/plot_random_forest_embedding.py
```python
# ..."""
=========================================================
Hashing feature transformation using Totally Random Trees
=========================================================

RandomTreesEmbedding provides a way to map data to a
very high-dimensional, sparse representation, which might
be beneficial for classification.
The mapping is completely unsupervised and very efficient.

This example visualizes the partitions given by several
trees and shows how the transformation can also be used for
non-linear dimensionality reduction or non-linear classification.

Points that are neighboring often share the same leaf of a tree and therefore
share large parts of their hashed representation. This allows to
separate two concentric circles simply based on the principal components
of the transformed data with truncated SVD.

In high-dimensional spaces, linear classifiers often achieve
excellent accuracy. For sparse binary data, BernoulliNB
is particularly well-suited. The bottom row compares the
decision boundary obtained by BernoulliNB in the transformed
space with an ExtraTreesClassifier forests learned on the
original data.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB

# make a synthetic dataset
X, y = make_circles(factor=0.5, random_state=0, noise=0.05)

# use RandomTreesEmbedding to transform data
hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
X_transformed = hasher.fit_transform(X)

# Visualize result after dimensionality reduction using truncated SVD
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_transformed)

# Learn a Naive Bayes classifier on the transformed data
nb = BernoulliNB()
nb.fit(X_transformed, y)


# Learn an ExtraTreesClassifier for comparison
trees = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0)
trees.fit(X, y)


# scatter plot of original and reduced data
fig = plt.figure(figsize=(9, 8))

ax = plt.subplot(221)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
ax.set_title("Original Data (2d)")
ax.set_xticks(())
ax.set_yticks(())

ax = plt.subplot(222)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor='k')
ax.set_title("Truncated SVD reduction (2d) of transformed data (%dd)" %
             X_transformed.shape[1])
ax.set_xticks(())
ax.set_yticks(())

# Plot the decision in original space. For that, we will assign a color
# to each point in the mesh [x_min, x_max]x[y_min, y_max].
h = .01
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# transform grid using RandomTreesEmbedding
transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]

ax = plt.subplot(223)
ax.set_title("Naive Bayes on Transformed data")
ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
ax.set_ylim(-1.4, 1.4)
ax.set_xlim(-1.4, 1.4)
ax.set_xticks(())
ax.set_yticks(())

# transform grid using ExtraTreesClassifier
y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

ax = plt.subplot(224)
ax.set_title("ExtraTrees predictions"
```
**88 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [key for key in dict_after_fit.keys()
                             if is_public_parameter(key)]

    attrs_added_by_fit = [key for key in public_keys_after_fit
                          if key not in dict_before_fit.keys()]

    # check that fit doesn't add any public attribute
    assert_true(not attrs_added_by_fit,
                ('Estimator adds public attribute(s) during'
                 ' the fit method.'
                 ' Estimators are only allowed to add private attributes'
                 ' either started with _ or ended'
                 ' with _ but %s added' % ', '.join(attrs_added_by_fit)))

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [key for key in public_keys_after_fit
                            if (dict_before_fit[key]
                                is not dict_after_fit[key])]

    assert_true(not attrs_changed_by_fit,
                ('Estimator changes public attribute(s) during'
                 ' the fit method. Estimators are only allowed'
                 ' to change attributes started'
                 ' or ended with _, but'
                 ' %s changed' % ', '.join(attrs_changed_by_fit)))
```
**89 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):


    def _set_oob_score(self, X, y):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = []

        for k in range(self.n_outputs_):
            predictions.append(np.zeros((n_samples, n_classes_[k])))

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_
```
90 - /tmp/repos/scikit-learn/sklearn/ensemble/base.py
```python
class BaseEnsemble(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):


    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")
```
91 - /tmp/repos/scikit-learn/benchmarks/bench_glmnet.py
```python
# ..."""
To run this, you'll need to have installed.

  * glmnet-python
  * scikit-learn (of course)

Does two benchmarks

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""
import numpy as np
import gc
from time import time
from sklearn.datasets.samples_generator import make_regression

alpha = 0.1
# alpha = 0.01


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def bench(factory, X, Y, X_test, Y_test, ref_coef):
    gc.collect()

    # start time
    tstart = time()
    clf = factory(alpha=alpha).fit(X, Y)
    delta = (time() - tstart)
    # stop time

    print("duration: %0.3fs" % delta)
    print("rmse: %f" % rmse(Y_test, clf.predict(X_test)))
    print("mean coef abs diff: %f" % abs(ref_coef - clf.coef_.ravel()).mean())
    return delta


if __name__ == '__main__':
    from glmnet.elastic_net import Lasso as GlmnetLasso
    from sklearn.linear_model import Lasso as ScikitLasso
    # Delayed import of matplotlib.pyplot
    import matplotlib.pyplot as plt

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 500
    n_features = 1000
    n_informative = n_features / 10
    n_test_samples = 1000
    for i in range(1, n + 1):
        print('==================')
        print('Iteration %s of %s' % (i, n))
        print('==================')

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples, n_features=n_features,
            noise=0.1, n_informative=n_informative, coef=True)

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:(i * step)]
        Y = Y[:(i * step)]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    plt.clf()
    xx = range(0, n * step, step)
    plt.title('Lasso regression on sample dataset (%d features)' % n_features)
    plt.plot(xx, scikit_results, 'b-', label='scikit-learn')
    plt.plot(xx, glmnet_results, 'r-', label='glmnet')
    plt.legend()
    plt.xlabel('number of samples to classify')
    plt.ylabel('Time (s)')
    plt.show()

    # now do a benchmark where the number of points is fixed
    # and the variable is the number of features

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 100
    n_samples = 500

    for i in range(1, n + 1):
        print('==================')
        print('Iteration %02d of %02d' % (i, n))
        print('==================')
        n_features = i * step
        n_informative = n_features / 10

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples, n_features=n_features,
            noise=0.1, n_informative=n_informative, coef=True)

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:n_samples]
        Y = Y[:n_samples]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: "
```
92 - /tmp/repos/scikit-learn/benchmarks/bench_isolation_forest.py
```python
# ..."""
==========================================
IsolationForest benchmark
==========================================
A test of IsolationForest on classical anomaly detection datasets.

The benchmark is run as follows:
1. The dataset is randomly split into a training set and a test set, both
assumed to contain outliers.
2. Isolation Forest is trained on the training set.
3. The ROC curve is computed on the test set using the knowledge of the labels.

Note that the smtp dataset contains a very small proportion of outliers.
Therefore, depending on the seed of the random number generator, randomly
splitting the data set might lead to a test set containing no outliers. In this
case a warning is raised when computing the ROC curve.
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle as sh

print(__doc__)


def print_outlier_ratio(y):
    """
    Helper function to show the distinct value count of element in the target.
    Useful indicator for the datasets used in bench_isolation_forest.py.
    """
    uniq, cnt = np.unique(y, return_counts=True)
    print("----- Target count values: ")
    for u, c in zip(uniq, cnt):
        print("------ %s -> %d occurrences" % (str(u), c))
    print("----- Outlier ratio: %.5f" % (np.min(cnt) / len(y)))


random_state = 1
fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 5))

# Set this to true for plotting score histograms for each dataset:
with_decision_function_histograms = False

# datasets available = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']

# Loop over all datasets for fitting and scoring the estimator:
for dat in datasets:

    # Loading and vectorizing the data:
    print('====== %s ======' % dat)
    print('--- Fetching data...')
    if dat in ['http', 'smtp', 'SF', 'SA']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True,
                                 percent10=True, random_state=random_state)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y, random_state=random_state)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)
        print('----- ')

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True, random_state=random_state)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)
        print_outlier_ratio(y)

    print('--- Vectorizing data...')

    if dat == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    if dat == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    if dat in ('http', 'smtp'):
        y = (y != b'normal.').astype(int)
        print_outlier_ratio(y)

    n_samples, n_features = X.shape
    n_samples_train = n_samples // 2

    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    print('--- Fitting the IsolationForest estimator...')
    model = IsolationForest(n_jobs=-1, random_state=random_state)
    tstart = time()
    model.fit(X_train)
    fit_time = time() - tstart
    tstart = time()

    scoring = - model.decision_function(X_test)  # the lower, the more abnormal

    print("--- Preparing the plot elements...")
    if with_decision_function_histograms:
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        bins = np.linspace(-0.5, 0.5, 200)
        ax[0].hist(scoring, bins, color='black')
        ax[0].set_title('Decision function for %s dataset' % dat)
        ax[1].hist(scoring[y_test == 0], bins, color='b', label='normal data')
        ax[1].legend(loc="lower right")
        ax[2].hist(scoring[y_test == 1], bins, color='r', label='outliers')
        ax[2].legend(loc="lower right")

    # Show ROC Curves
    predict_time = time() - tstart
    fpr, tpr, thresholds = roc_curve(y_test, scoring)
    auc_score = auc(fpr, tpr)
    label = ('%s (AUC: %0.3f, train_time= %0.2fs, '
             'test_time= %0.2fs)' % (dat, auc_score, fit_time, predict_time))
    # Print AUC score and train/test time:
    print(label)
    ax_roc.plot(fpr, tpr, lw=1, label=label)


ax_roc.set_xlim([-0.05, 1.05])
ax_roc.set_ylim([-0.05, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver operating characteristic (ROC) curves')
ax_roc.legend(loc="lower right")
fig_roc.tight_layout()
plt.show()

```
93 - /tmp/repos/scikit-learn/sklearn/decomposition/online_lda.py
```python
"""

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_method='batch',
                 learning_decay=.7, learning_offset=10., max_iter=10,
                 batch_size=128, evaluate_every=-1, total_samples=1e6,
                 perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=1, verbose=0, random_state=None, n_topics=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.n_topics = n_topics

    def _check_params(self):
        """Check model parameters."""
        if self.n_topics is not None:
            self._n_components = self.n_topics
            warnings.warn("n_topics has been renamed to n_components in "
                          "version 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        else:
            self._n_components = self.n_components

        if self._n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self._n_components)

        if self.total_samples <= 0:
            raise ValueError("Invalid 'total_samples' parameter: %r"
                             % self.total_samples)

        if self.learning_offset < 0:
            raise ValueError("Invalid 'learning_offset' parameter: %r"
                             % self.learning_offset)

        if self.learning_method not in ("batch", "online"):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self._n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self._n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self._n_components, n_features))

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_))

    
```
**94 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""

    def __init__(self,
                 n_estimators=10,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomTreesEmbedding, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object

        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    
```
95 - /tmp/repos/scikit-learn/sklearn/decomposition/online_lda.py
```python
"""

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_method=None,
                 learning_decay=.7, learning_offset=10., max_iter=10,
                 batch_size=128, evaluate_every=-1, total_samples=1e6,
                 perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=1, verbose=0, random_state=None, n_topics=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.n_topics = n_topics

    def _check_params(self):
        """Check model parameters."""
        if self.n_topics is not None:
            self._n_components = self.n_topics
            warnings.warn("n_topics has been renamed to n_components in "
                          "version 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        else:
            self._n_components = self.n_components

        if self._n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self._n_components)

        if self.total_samples <= 0:
            raise ValueError("Invalid 'total_samples' parameter: %r"
                             % self.total_samples)

        if self.learning_offset < 0:
            raise ValueError("Invalid 'learning_offset' parameter: %r"
                             % self.learning_offset)

        if self.learning_method not in ("batch", "online", None):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self._n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self._n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self._n_components, n_features))

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_))

    
```
**96 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
"""

    def __init__(self,
                 n_estimators=10,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomTreesEmbedding, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = 'mse'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Returns self.

        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    
```
97 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python


rcv1 = fetch_rcv1()
X = rcv1.data
n_samples, n_features = X.shape

# consider the binary classification problem 'CCAT' vs the rest
ccat_idx = rcv1.target_names.tolist().index('CCAT')
y = rcv1.target.tocsc()[:, ccat_idx].toarray().ravel().astype(np.float64)
y[y == 0] = -1

# parameters
C = 1.
fit_intercept = True
tol = 1.0e-14

# max_iter range
sgd_iter_range = list(range(1, 121, 10))
newton_iter_range = list(range(1, 25, 3))
lbfgs_iter_range = list(range(1, 242, 12))
liblinear_iter_range = list(range(1, 37, 3))
liblinear_dual_iter_range = list(range(1, 85, 6))
sag_iter_range = list(range(1, 37, 3))

clfs = [
    ("LR-liblinear",
     LogisticRegression(C=C, tol=tol,
                        solver="liblinear", fit_intercept=fit_intercept,
                        intercept_scaling=1),
     liblinear_iter_range, [], [], [], []),
    ("LR-liblinear-dual",
     LogisticRegression(C=C, tol=tol, dual=True,
                        solver="liblinear", fit_intercept=fit_intercept,
                        intercept_scaling=1),
     liblinear_dual_iter_range, [], [], [], []),
    ("LR-SAG",
     LogisticRegression(C=C, tol=tol,
                        solver="sag", fit_intercept=fit_intercept),
     sag_iter_range, [], [], [], []),
    ("LR-newton-cg",
     LogisticRegression(C=C, tol=tol, solver="newton-cg",
                        fit_intercept=fit_intercept),
     newton_iter_range, [], [], [], []),
    ("LR-lbfgs",
     LogisticRegression(C=C, tol=tol,
                        solver="lbfgs", fit_intercept=fit_intercept),
     lbfgs_iter_range, [], [], [], []),
    ("SGD",
     SGDClassifier(alpha=1.0 / C / n_samples, penalty='l2', loss='log',
                   fit_intercept=fit_intercept, verbose=0),
     sgd_iter_range, [], [], [], [])]



```
98 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
# ...


@deprecated("The class RandomizedLogisticRegression is deprecated in 0.19"
            " and will be removed in 0.21.")
class RandomizedLogisticRegression(BaseRandomizedLinearModel):
    """Randomized Logistic Regression

    Randomized Logistic Regression works by subsampling the training
    data and fitting a L1-penalized LogisticRegression model where the
    penalty of a random subset of coefficients has been scaled. By
    performing this double randomization several times, the method
    assigns high scores to features that are repeatedly selected across
    randomizations. This is known as stability selection. In short,
    features selected more often are considered good features.

    Parameters
    ----------
    C : float or array-like of shape [n_reg_parameter], optional, default=1
        The regularization parameter C in the LogisticRegression.
        When C is an array, fit will take each regularization parameter in C
        one by one for LogisticRegression and store results for each one
        in ``all_scores_``, where columns and rows represent corresponding
        reg_parameters and features.

    scaling : float, optional, default=0.5
        The s parameter used to randomly scale the penalty of different
        features.
        Should be between 0 and 1.

    sample_fraction : float, optional, default=0.75
        The fraction of samples to be used in each randomized design.
        Should be between 0 and 1. If 1, all samples are used.

    n_resampling : int, optional, default=200
        Number of randomized models.

    selection_threshold : float, optional, default=0.25
        The score above which features should be selected.

    tol : float, optional, default=1e-3
         tolerance for stopping criteria of LogisticRegression

    fit_intercept : boolean, optional, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    normalize : boolean, optional, default True
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : integer, optional
        Number of CPUs to use during the resampling. If '-1', use
        all the CPUs

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    memory : None, str or object with the joblib.Memory interface, optional 
```
**99 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = ['Ridge', 'SVR', 'NuSVR', 'NuSVC',
                            'RidgeClassifier', 'SVC', 'RandomizedLasso',
                            'LogisticRegressionCV', 'LinearSVC',
                            'LogisticRegression']

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = multioutput_estimator_convert_y_2d(estimator, y_)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
            estimator.fit(X, y_)

        # HuberRegressor depends on scipy.optimize.fmin_l_bfgs_b
        # which doesn't return a n_iter for old versions of SciPy.
        if not (name == 'HuberRegressor' and estimator.n_iter_ is None):
            assert_greater_equal(estimator.n_iter_, 1)
```
**100 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
def _boston_subset(n_samples=200):
    global BOSTON
    if BOSTON is None:
        boston = load_boston()
        X, y = boston.data, boston.target
        X, y = shuffle(X, y, random_state=0)
        X, y = X[:n_samples], y[:n_samples]
        X = StandardScaler().fit_transform(X)
        BOSTON = X, y
    return BOSTON
```
101 - /tmp/repos/scikit-learn/examples/ensemble/plot_adaboost_regression.py
```python
"""
======================================
Decision Tree Regression with AdaBoost
======================================

A decision tree is boosted using the AdaBoost.R2 [1]_ algorithm on a 1D
sinusoidal dataset with a small amount of Gaussian noise.
299 boosts (300 decision trees) is compared with a single decision tree
regressor. As the number of boosts is increased the regressor can fit more
detail.

.. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

"""
print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
```
102 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python
X_train = X_train_[:n_samples]
    X_test = X_test_[:n_samples]
    y_train = y_train_[:n_samples]
    y_test = y_test_[:n_samples]
    assert X_train.shape[0] == n_samples
    assert X_test.shape[0] == n_samples
    print("Data size: %d samples train, %d samples test."
          % (n_samples, n_samples))
    print("Fitting a sklearn model...")
    tic = time()
    est = Estimator(learning_rate=lr,
                    max_iter=n_trees,
                    max_bins=max_bins,
                    max_leaf_nodes=n_leaf_nodes,
                    n_iter_no_change=None,
                    random_state=0,
                    verbose=0)
    est.fit(X_train, y_train)
    sklearn_fit_duration = time() - tic
    tic = time()
    sklearn_score = est.score(X_test, y_test)
    sklearn_score_duration = time() - tic
    print("score: {:.4f}".format(sklearn_score))
    print("fit duration: {:.3f}s,".format(sklearn_fit_duration))
    print("score duration: {:.3f}s,".format(sklearn_score_duration))

    lightgbm_score = None
    lightgbm_fit_duration = None
    lightgbm_score_duration = None
    if args.lightgbm:
        print("Fitting a LightGBM model...")
        # get_lightgbm does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')

        tic = time()
        lightgbm_est.fit(X_train, y_train)
        lightgbm_fit_duration = time() - tic
        tic = time()
        lightgbm_score = lightgbm_est.score(X_test, y_test)
        lightgbm_score_duration = time() - tic
        print("score: {:.4f}".format(lightgbm_score))
        print("fit duration: {:.3f}s,".format(lightgbm_fit_duration))
        print("score duration: {:.3f}s,".format(lightgbm_score_duration))

    xgb_score = None
    xgb_fit_duration = None
    xgb_score_duration = None
    if args.xgboost:
        print("Fitting an XGBoost model...")
        # get_xgb does not accept loss='auto'
        if args.problem == 'classification':
            loss = 'binary_crossentropy' if args.n_classes == 2 else \
                'categorical_crossentropy'
            est.set_params(loss=loss)
        xgb_est = get_equivalent_estimator(est, lib='xgboost')

        tic = time()
        xgb_est.fit(X_train, y_train)
        xgb_fit_duration = time() - tic
        tic = time()
        xgb_score = xgb_est.score(X_test, y_test)
        xgb_score_duration = time() - tic
        print("score: {:.4f}".format(xgb_score))
        print("fit duration: {:.3f}s,".format(xgb_fit_duration))
        print("score duration: {:.3f}s,".format(xgb_score_duration))

    cat_score = None
    cat_fit_duration = None
    cat_score_duration = None
    
```
103 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingRegressor(BaseBagging, RegressorMixin):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
104 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingRegressor(BaseBagging, RegressorMixin):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
105 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
@deprecated("The class RandomizedLogisticRegression is deprecated in 0.19"
            " and will be removed in 0.21.")
class RandomizedLogisticRegression(BaseRandomizedLinearModel):


    def _make_estimator_and_params(self, X, y):
        params = dict(C=self.C, tol=self.tol,
                      fit_intercept=self.fit_intercept)
        return _randomized_logistic, params
```
**106 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class ForestClassifier(six.with_metaclass(ABCMeta, BaseForest,
                                          ClassifierMixin)):


    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, six.string_types):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample". Given "%s".'
                                     % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight("balanced", '
                         'classes, y). In place of y you can use a large '
                         'enough sample of the full training set target to '
                         'properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight != 'balanced_subsample' or
                    not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight,
                                                              y_original)

        return y, expanded_class_weight
```
107 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* constructor parameter.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
**108 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):


    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return self.estimators_[0]._validate_X_predict(X, check_input=True)
```
109 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

        .. versionadded:: 0.17
           *warm_start* constructor parameter.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
110 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
class TSNE(BaseEstimator):


    @property
    @deprecated("Attribute n_iter_final was deprecated in version 0.19 and "
                "will be removed in 0.21. Use ``n_iter_`` instead")
    def n_iter_final(self):
        return self.n_iter_
```
111 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* constructor parameter.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
112 - /tmp/repos/scikit-learn/sklearn/decomposition/online_lda.py
```python
"""

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_method='batch',
                 learning_decay=.7, learning_offset=10., max_iter=10,
                 batch_size=128, evaluate_every=-1, total_samples=1e6,
                 perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=None, verbose=0, random_state=None):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _check_params(self):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self.n_components)

        if self.total_samples <= 0:
            raise ValueError("Invalid 'total_samples' parameter: %r"
                             % self.total_samples)

        if self.learning_offset < 0:
            raise ValueError("Invalid 'learning_offset' parameter: %r"
                             % self.learning_offset)

        if self.learning_method not in ("batch", "online"):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1. / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1. / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.
        init_var = 1. / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features))

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_))

    
```
113 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingClassifier(BaseBagging, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* constructor parameter.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
114 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

from __future__ import division


import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse

from ..base import BaseEstimator
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import is_classifier
from ..externals import six
from ..utils import check_array
from ..utils import check_random_state
from ..utils import compute_sample_weight
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted

from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import DepthFirstTreeBuilder
from ._tree import BestFirstTreeBuilder
from ._tree import Tree
from . import _tree, _splitter, _criterion

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse": _criterion.MSE, "friedman_mse": _criterion.FriedmanMSE,
                "mae": _criterion.MAE}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
                    "random": _splitter.RandomSparseSplitter}
```
115 - /tmp/repos/scikit-learn/sklearn/preprocessing/_encoders.py
```python
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
        
```
116 - /tmp/repos/scikit-learn/examples/ensemble/plot_adaboost_multiclass.py
```python
"""
print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()

```
117 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingRegressor(BaseBagging, RegressorMixin):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
118 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
# ...


class BaggingRegressor(BaseBagging, RegressorMixin):
    """A Bagging regressor.

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    
```
119 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if (self.is_data_valid is not None
                    and not self.is_data_valid(X_subset, y_subset)):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator.fit(X_subset, y_subset)
            else:
                base_estimator.fit(X_subset, y_subset,
                                   sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)

            # XXX: Deprecation: Remove this if block in 0.20
            if self.residual_metric is not None:
                diff = y_pred - y
                if diff.ndim == 1:
                    diff = diff.reshape(-1, 1)
                residuals_subset = self.residual_metric(diff)
            else:
                residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = base_estimator.score(X_inlier_subset,
                                                y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        
```
120 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if (self.is_data_valid is not None
                    and not self.is_data_valid(X_subset, y_subset)):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator.fit(X_subset, y_subset)
            else:
                base_estimator.fit(X_subset, y_subset,
                                   sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)

            # XXX: Deprecation: Remove this if block in 0.20
            if self.residual_metric is not None:
                diff = y_pred - y
                if diff.ndim == 1:
                    diff = diff.reshape(-1, 1)
                residuals_subset = self.residual_metric(diff)
            else:
                residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = base_estimator.score(X_inlier_subset,
                                                y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        
```
121 - /tmp/repos/scikit-learn/examples/ensemble/plot_isolation_forest.py
```python
"""
==========================================
IsolationForest example
==========================================

An example using IsolationForest for anomaly detection.

The IsolationForest 'isolates' observations by randomly selecting a feature
and then randomly selecting a split value between the maximum and minimum
values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure
of normality and our decision function.

Random partitioning produces noticeable shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path lengths
for particular samples, they are highly likely to be anomalies.

.. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
    Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
```
122 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
"""
Benchmarks of Non-Negative Matrix Factorization
"""
# Authors: Tom Dupre la Tour (benchmark)
#          Chih-Jen Linn (original projected gradient NMF implementation)
#          Anthony Di Franco (projected gradient, Python and NumPy port)
# License: BSD 3 clause

from __future__ import print_function
from time import time
import sys
import warnings
import numbers

import numpy as np
import matplotlib.pyplot as plt
import pandas

from sklearn.utils.testing import ignore_warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.nmf import NMF
from sklearn.decomposition.nmf import _initialize_nmf
from sklearn.decomposition.nmf import _beta_divergence
from sklearn.decomposition.nmf import INTEGER_TYPES, _check_init
from sklearn.externals.joblib import Memory
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_non_negative


mem = Memory(cachedir='.', verbose=0)



if __name__ == '__main__':
    alpha = 0.
    l1_ratio = 0.5
    n_components = 10
    tol = 1e-15

    # first benchmark on 20 newsgroup dataset: sparse, shape(11314, 39116)
    plot_name = "20 Newsgroups sparse dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 6)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_20news = load_20news()
    run_bench(X_20news, clfs, plot_name, n_components, tol, alpha, l1_ratio)

    # second benchmark on Olivetti faces dataset: dense, shape(400, 4096)
    plot_name = "Olivetti Faces dense dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 12)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_faces = load_faces()
    run_bench(X_faces, clfs, plot_name, n_components, tol, alpha, l1_ratio,)

    plt.show()
```
123 - /tmp/repos/scikit-learn/benchmarks/bench_plot_randomized_svd.py
```python
"""
    Measure time for decomposition
    """
    print("... running SVD ...")
    if method is not 'fbpca':
        gc.collect()
        t0 = time()
        U, mu, V = randomized_svd(X, n_comps, n_oversamples, n_iter,
                                  power_iteration_normalizer,
                                  random_state=random_state, transpose=False)
        call_time = time() - t0
    else:
        gc.collect()
        t0 = time()
        # There is a different convention for l here
        U, mu, V = fbpca.pca(X, n_comps, raw=True, n_iter=n_iter,
                             l=n_oversamples+n_comps)
        call_time = time() - t0

    return U, mu, V, call_time


def norm_diff(A, norm=2, msg=True):
    """
    Compute the norm diff with the original matrix, when randomized
    SVD is called with *params.

    norm: 2 => spectral; 'fro' => Frobenius
    """

    if msg:
        print("... computing %s norm ..." % norm)
    if norm == 2:
        # s = sp.linalg.norm(A, ord=2)  # slow
        value = sp.sparse.linalg.svds(A, k=1, return_singular_vectors=False)
    else:
        if sp.sparse.issparse(A):
            value = sp.sparse.linalg.norm(A, ord=norm)
        else:
            value = sp.linalg.norm(A, ord=norm)
    return value


def scalable_frobenius_norm_discrepancy(X, U, s, V):
    # if the input is not too big, just call scipy
    if X.shape[0] * X.shape[1] < MAX_MEMORY:
        A = X - U.dot(np.diag(s).dot(V))
        return norm_diff(A, norm='fro')

    print("... computing fro norm by batches...")
    batch_size = 1000
    Vhat = np.diag(s).dot(V)
    cum_norm = .0
    for batch in gen_batches(X.shape[0], batch_size):
        M = X[batch, :] - U[batch, :].dot(Vhat)
        cum_norm += norm_diff(M, norm='fro', msg=False)
    return np.sqrt(cum_norm)


def bench_a(X, dataset_name, power_iter, n_oversamples, n_comps):

    all_time = defaultdict(list)
    if enable_spectral_norm:
        all_spectral = defaultdict(list)
        X_spectral_norm = norm_diff(X, norm=2, msg=False)
    all_frobenius = defaultdict(list)
    X_fro_norm = norm_diff(X, norm='fro', msg=False)

    for pi in power_iter:
        for pm in ['none', 'LU', 'QR']:
            print("n_iter = %d on sklearn - %s" % (pi, pm))
            U, s, V, time = svd_timing(X, n_comps, n_iter=pi,
                                       power_iteration_normalizer=pm,
                                       n_oversamples=n_oversamples)
            label = "sklearn - %s" % pm
            all_time[label].append(time)
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(norm_diff(X - A, norm=2) /
                                           X_spectral_norm)
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

        if fbpca_available:
            print("n_iter = %d on fbca" % (pi))
            U, s, V, time = svd_timing(X, n_comps, n_iter=pi,
                                       power_iteration_normalizer=pm,
                                       n_oversamples=n_oversamples,
                                       method='fbpca')
            label = "fbpca"
            all_time[label].append(time)
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(norm_diff(X - A, norm=2) /
                                           X_spectral_norm)
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

    if enable_spectral_norm:
        title = "%s: spectral norm diff vs running time" % (dataset_name)
        plot_time_vs_s(all_time, all_spectral, power_iter, title)
    title = "%s: Frobenius norm diff vs running time" % (dataset_name)
    plot_time_vs_s(all_time, all_frobenius, power_iter, title)


def bench_b(power_list):

    n_samples, n_features = 1000, 10000
    data_params = {'n_samples': n_samples, 'n_features': n_features,
                   'tail_strength': .7, 'random_state': random_state}
    dataset_name = "low rank matrix %d x %d" % (n_samples, n_features)
    ranks = [10, 50, 100]

    if enable_spectral_norm:
        all_spectral = defaultdict(list)
    all_frobenius = defaultdict(list)
    for rank in ranks:
        X = make_low_rank_matrix(effective_rank=rank, **data_params)
        if enable_spectral_norm:
            X_spectral_norm = norm_diff(X, norm=2, msg=False)
        X_fro_norm = norm_diff(X, norm='fro', msg=False)

        for n_comp in [np.int(rank/2), rank, rank*2]:
            label = "rank=%d, n_comp=%d" % (rank, n_comp)
            print(label)
            for pi in power_list:
                U, s, V, _ = svd_timing(X, n_comp, n_iter=pi, n_oversamples=2,
                                        power_iteration_normalizer='LU')
                if enable_spectral_norm:
                    A = U.dot(np.diag(s).dot(V))
                    all_spectral[label].append(norm_diff(X - A, norm=2) /
                                               X_spectral_norm)
                f = scalable_frobenius_norm_discrepancy(X, U, s, V)
                all_frobenius[label].append(f / X_fro_norm)

    if enable_spectral_norm:
        title = "%s: spectral norm diff vs n power iteration" % (dataset_name)
        plot_power_iter_vs_s(power_iter, all_spectral, title)
    title = "%s: Frobenius norm diff vs n power iteration" % (dataset_name)
    plot_power_iter_vs_s(power_iter, all_frobenius, title)


def bench_c(datasets, n_comps):
    all_time = defaultdict(list)
    if enable_spectral_norm:
        all_spectral = defaultdict(list)
    all_frobenius = defaultdict(list)

    for dataset_name in datasets:
        X = get_data(dataset_name)
        if X is None:
            continue

        if enable_spectral_norm:
            X_spectral_norm = norm_diff(X, norm=2, msg=False)
        X_fro_norm = norm_diff(X, norm='fro', msg=False)
        n_comps = np.minimum(n_comps, np.min(X.shape))

        label = "sklearn"
        print("%s %d x %d - %s" %
              (dataset_name, X.shape[0], X.shape[1], label))
        U, s, V, time = svd_timing(X, n_comps, n_iter=2, n_oversamples=10,
                                   method=label)

        all_time[label].append(time)
        if enable_spectral_norm:
            A = U.dot(np.diag(s).dot(V))
            all_spectral[label].append(norm_diff(X - A, norm=2) /
                                       X_spectral_norm)
        f = scalable_frobenius_norm_discrepancy(X, U, s, V)
        all_frobenius[label].append(f / X_fro_norm)

        if fbpca_available:
            label = "fbpca"
            print("%s %d x %d - %s" %
                  (dataset_name, X.shape[0], X.shape[1], label))
            U, s, V, time = svd_timing(X, n_comps, n_iter=2, n_oversamples=2,
                                       method=label)
            all_time[label].append(time)
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(norm_diff(X - A, norm=2) /
                                           X_spectral_norm)
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

    if len(all_time) == 0:
        raise ValueError("No tests ran. Aborting.")

    if enable_spectral_norm:
        title = "normalized spectral norm diff vs running time"
        scatter_time_vs_s(all_time, all_spectral, datasets, title)
    title = "normalized Frobenius norm diff vs running time"
    scatter_time_vs_s(all_time, all_frobenius, datasets, title)


if __name__ == '__main__':
    random_state = check_random_state(1234)

    power_iter = np.linspace(0, 6, 7, dtype=int)
    n_comps = 50

    for dataset_name in datasets:
        X = get_data(dataset_name)
        if X is None:
            continue
        print(" >>>>>> Benching sklearn and fbpca on %s %d x %d" %
              (dataset_name, X.shape[0], X.shape[1]))
        bench_a(X, dataset_name, power_iter, n_oversamples=2,
                n_comps=np.minimum(n_comps, np.min(X.shape)))

    print(" >>>>>> Benching on simulated low rank matrix with variable rank")
    bench_b(power_iter)

    print(" >>>>>> Benching sklearn and fbpca default configurations")
    bench_c(datasets + big_sparse_datasets, n_comps)

    plt.show()

```
124 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
    accept a custom RNG instance and always use the singleton RNG from
    ``numpy.random``. Hence setting ``random_state`` will not guarantee a
    deterministic iteration whenever ``scipy.stats`` distributions are used to
    define the parameter search space.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=False
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds.

        .. deprecated:: 0.22
            Parameter ``iid`` is deprecated in 0.22 and will be removed in 0.24

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. When refit is callable, ``best_score_`` is disabled.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``.

    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.


    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> from scipy.stats import uniform
    >>> iris = load_iris()
    >>> logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
    ...                               random_state=0)
    >>> distributions = dict(C=uniform(loc=0, scale=4),
    ...                      penalty=['l2', 'l1'])
    >>> clf = RandomizedSearchCV(logistic, distributions, random_state=0)
    >>> search = clf.fit(iris.data, iris.target)
    >>> search.best_params_
    {'C': 2..., 'penalty': 'l1'}
    
```
125 - /tmp/repos/scikit-learn/sklearn/ensemble/base.py
```python
def _set_random_states(estimator, random_state=None):
    """Sets fixed random_state parameters for an estimator

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------

    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(MAX_RAND_SEED)

    if to_set:
        estimator.set_params(**to_set)
```
126 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
127 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
128 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
129 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
130 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    
```
131 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
class IsolationForest(BaseBagging, OutlierMixin):


    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
        """
        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, six.string_types):
            if self.max_samples == 'auto':
                max_samples = min(256, n_samples)
            else:
                raise ValueError('max_samples (%s) is not supported.'
                                 'Valid choices are: "auto", int or'
                                 'float' % self.max_samples)

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super(IsolationForest, self)._fit(X, y, max_samples,
                                          max_depth=max_depth,
                                          sample_weight=sample_weight)

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            # need to save (depreciated) threshold_ in this case:
            self._threshold_ = sp.stats.scoreatpercentile(
                self.score_samples(X), 100. * 0.1)
        elif self.contamination == "legacy":  # to be rm in 0.22
            self.offset_ = sp.stats.scoreatpercentile(
                self.score_samples(X), 100. * 0.1)
        else:
            self.offset_ = sp.stats.scoreatpercentile(
                self.score_samples(X), 100. * self.contamination)

        return self
```
132 - /tmp/repos/scikit-learn/benchmarks/bench_lof.py
```python
# ..."""
============================
LocalOutlierFactor benchmark
============================

A test of LocalOutlierFactor on classical anomaly detection datasets.

Note that LocalOutlierFactor is not meant to predict on a test set and its
performance is assessed in an outlier detection context:
1. The model is trained on the whole dataset which is assumed to contain
outliers.
2. The ROC curve is computed on the same dataset using the knowledge of the
labels.
In this context there is no need to shuffle the dataset because the model
is trained and tested on the whole dataset. The randomness of this benchmark
is only caused by the random selection of anomalies in the SA dataset.

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.preprocessing import LabelBinarizer

print(__doc__)

random_state = 2  # to control the random selection of anomalies in SA

# datasets available: ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']

plt.figure()
for dataset_name in datasets:
    # loading and vectorization
    print('loading data')
    if dataset_name in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dataset_name, percent10=True,
                                 random_state=random_state)
        X = dataset.data
        y = dataset.target

    if dataset_name == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dataset_name == 'forestcover':
        dataset = fetch_covtype()
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dataset_name == 'SF':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'SA':
        lb = LabelBinarizer()
        x1 = lb.fit_transform(X[:, 1].astype(str))
        x2 = lb.fit_transform(X[:, 2].astype(str))
        x3 = lb.fit_transform(X[:, 3].astype(str))
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b'normal.').astype(int)

    if dataset_name == 'http' or dataset_name == 'smtp':
        y = (y != b'normal.').astype(int)

    X = X.astype(float)

    print('LocalOutlierFactor processing...')
    model = LocalOutlierFactor(n_neighbors=20)
    tstart = time()
    model.fit(X)
    fit_time = time() - tstart
    scoring = -model.negative_outlier_factor_  # the lower, the more normal
    fpr, tpr, thresholds = roc_curve(y, scoring)
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1,
             label=('ROC for %s (area = %0.3f, train-time: %0.2fs)'
                    % (dataset_name, AUC, fit_time)))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

```
133 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
"""
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        
```
134 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
"""
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        
```
135 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
"""
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        
```
136 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
"""
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        
```
137 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
"""
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle

    def _fit(self, X, skip_num_points=0):
        
```
138 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
@deprecated("The class RandomizedLasso is deprecated in 0.19"
            " and will be removed in 0.21.")
class RandomizedLasso(BaseRandomizedLinearModel):


    def _make_estimator_and_params(self, X, y):
        alpha = self.alpha
        if isinstance(alpha, six.string_types) and alpha in ('aic', 'bic'):
            model = LassoLarsIC(precompute=self.precompute,
                                criterion=self.alpha,
                                max_iter=self.max_iter,
                                eps=self.eps)
            model.fit(X, y)
            self.alpha_ = alpha = model.alpha_

        precompute = self.precompute
        # A precomputed Gram array is useless, since _randomized_lasso
        # change X a each iteration
        if hasattr(precompute, '__array__'):
            precompute = 'auto'
        assert precompute in (True, False, None, 'auto')
        return _randomized_lasso, dict(alpha=alpha, max_iter=self.max_iter,
                                       eps=self.eps,
                                       precompute=precompute)
```
139 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python


    all_classes = []
    # get parent folder
    path = sklearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='sklearn.', onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] != 'BaseEstimator')]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if not include_dont_test:
        estimators = [c for c in estimators if not c[0] in DONT_TEST]

    if not include_other:
        estimators = [c for c in estimators if not c[0] in OTHER]
    # possibly get rid of meta estimators
    if not include_meta_estimators:
        estimators = [c for c in estimators if not c[0] in META_ESTIMATORS]
    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin,
                   'regressor': RegressorMixin,
                   'transformer': TransformerMixin,
                   'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def skip_if_32bit(func):
    """Test decorator that skips tests on 32bit platforms."""
    @wraps(func)
    def run_test(*args, **kwargs):
        bits = 8 * struct.calcsize("P")
        if bits == 32:
            raise SkipTest('Test skipped on 32bit platforms.')
        else:
            return func(*args, **kwargs)
    return run_test



```
140 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python


    all_classes = []
    # get parent folder
    path = sklearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='sklearn.', onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] != 'BaseEstimator')]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if not include_dont_test:
        estimators = [c for c in estimators if not c[0] in DONT_TEST]

    if not include_other:
        estimators = [c for c in estimators if not c[0] in OTHER]
    # possibly get rid of meta estimators
    if not include_meta_estimators:
        estimators = [c for c in estimators if not c[0] in META_ESTIMATORS]
    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin,
                   'regressor': RegressorMixin,
                   'transformer': TransformerMixin,
                   'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def skip_if_32bit(func):
    """Test decorator that skips tests on 32bit platforms."""
    @wraps(func)
    def run_test(*args, **kwargs):
        bits = 8 * struct.calcsize("P")
        if bits == 32:
            raise SkipTest('Test skipped on 32bit platforms.')
        else:
            return func(*args, **kwargs)
    return run_test



```
141 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python


    all_classes = []
    # get parent folder
    path = sklearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='sklearn.', onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] != 'BaseEstimator')]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if not include_dont_test:
        estimators = [c for c in estimators if not c[0] in DONT_TEST]

    if not include_other:
        estimators = [c for c in estimators if not c[0] in OTHER]
    # possibly get rid of meta estimators
    if not include_meta_estimators:
        estimators = [c for c in estimators if not c[0] in META_ESTIMATORS]
    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin,
                   'regressor': RegressorMixin,
                   'transformer': TransformerMixin,
                   'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def skip_if_32bit(func):
    """Test decorator that skips tests on 32bit platforms."""
    @wraps(func)
    def run_test(*args, **kwargs):
        bits = 8 * struct.calcsize("P")
        if bits == 32:
            raise SkipTest('Test skipped on 32bit platforms.')
        else:
            return func(*args, **kwargs)
    return run_test



```
142 - /tmp/repos/scikit-learn/sklearn/utils/testing.py
```python


    all_classes = []
    # get parent folder
    path = sklearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='sklearn.', onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseEstimator) and
                      c[0] != 'BaseEstimator')]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if not include_dont_test:
        estimators = [c for c in estimators if not c[0] in DONT_TEST]

    if not include_other:
        estimators = [c for c in estimators if not c[0] in OTHER]
    # possibly get rid of meta estimators
    if not include_meta_estimators:
        estimators = [c for c in estimators if not c[0] in META_ESTIMATORS]
    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'classifier': ClassifierMixin,
                   'regressor': RegressorMixin,
                   'transformer': TransformerMixin,
                   'cluster': ClusterMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators
                                            if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'classifier', "
                             "'regressor', 'transformer', 'cluster' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def set_random_state(estimator, random_state=0):
    """Set random state of an estimator if it has the `random_state` param.
    """
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=random_state)


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed."""
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def skip_if_32bit(func):
    """Test decorator that skips tests on 32bit platforms."""
    @wraps(func)
    def run_test(*args, **kwargs):
        bits = 8 * struct.calcsize("P")
        if bits == 32:
            raise SkipTest('Test skipped on 32bit platforms.')
        else:
            return func(*args, **kwargs)
    return run_test



```
**143 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_classifiers(name, classifier_orig):
    if name == "NuSVC":
        # the sparse version has a parameter that doesn't do anything
        raise SkipTest("Not testing NuSVC class weight as it is ignored.")
    if name.endswith("NB"):
        # NaiveBayes classifiers have a somewhat different interface.
        # FIXME SOON!
        raise SkipTest

    for n_centers in [2, 3]:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0)

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _is_pairwise(classifier_orig):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(
            class_weight=class_weight)
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
        #      0.88 (Issue #9111)
        assert_greater(np.mean(y_pred == 0), 0.87)
```
144 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py
```python
# Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
# ...
```
145 - /tmp/repos/scikit-learn/sklearn/metrics/ranking.py
```python
# Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds
# ...
```
146 - /tmp/repos/scikit-learn/sklearn/tree/_reingold_tilford.py
```python
# taken from https://github.com/llimllib/pymag-trees/blob/master/buchheim.py
# with slight modifications

#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#                    Version 2, December 2004
#
# Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>
#
# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.
#
#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

#  0. You just DO WHAT THE FUCK YOU WANT TO.


import numpy as np
```
147 - /tmp/repos/scikit-learn/examples/ensemble/plot_gradient_boosting_early_stopping.py
```python
memory usage and prediction latency.
"""

# Authors: Vighnesh Birodkar <vighneshbirodkar@nyu.edu>
#          Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.model_selection import train_test_split

print(__doc__)

data_list = [datasets.load_iris(), datasets.load_digits()]
data_list = [(d.data, d.target) for d in data_list]
data_list += [datasets.make_hastie_10_2()]
names = ['Iris Data', 'Digits Data', 'Hastie Data']

n_gb = []
score_gb = []
time_gb = []
n_gbes = []
score_gbes = []
time_gbes = []

n_estimators = 500

for X, y in data_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    # We specify that if the scores don't improve by atleast 0.01 for the last
    # 10 stages, stop fitting additional stages
    gbes = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                               validation_fraction=0.2,
                                               n_iter_no_change=5, tol=0.01,
                                               random_state=0)
    gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                             random_state=0)
    start = time.time()
    gb.fit(X_train, y_train)
    time_gb.append(time.time() - start)

    start = time.time()
    gbes.fit(X_train, y_train)
    time_gbes.append(time.time() - start)

    score_gb.append(gb.score(X_test, y_test))
    score_gbes.append(gbes.score(X_test, y_test))

    n_gb.append(gb.n_estimators_)
    n_gbes.append(gbes.n_estimators_)

bar_width = 0.2
n = len(data_list)
index = np.arange(0, n * bar_width, bar_width) * 2.5
index = index[0:n]

#######################################################################
# Compare scores with and without early stopping
# ----------------------------------------------

plt.figure(figsize=(9, 5))

bar1 = plt.bar(index, score_gb, bar_width, label='Without early stopping',
               color='crimson')
bar2 = plt.bar(index + bar_width, score_gbes, bar_width,
               label='With early stopping', color='coral')

plt.xticks(index + bar_width, names)
plt.yticks(np.arange(0, 1.3, 0.1))


def autolabel(rects, n_estimators):
    """
    Attach a text label above each bar displaying n_estimators of each model
    """
    for i, rect in enumerate(rects):
        plt.text(rect.get_x() + rect.get_width() / 2.,
                 1.05 * rect.get_height(), 'n_est=%d' % n_estimators[i],
                 ha='center', va='bottom')


autolabel(bar1, n_gb)
autolabel(bar2, n_gbes)

plt.ylim([0, 1.3])
plt.legend(loc='best')
plt.grid(True)

plt.xlabel('Datasets')
plt.ylabel('Test score')

plt.show()


#######################################################################
# Compare fit times with and without early stopping
# ----------------------------------------------

plt.figure(figsize=(9, 5))


```
148 - /tmp/repos/scikit-learn/examples/plot_missing_values.py
```python
"""
======================================================
Imputing missing values before building an estimator
======================================================

This example shows that imputing the missing values can give better
results than discarding the samples containing any missing value.
Imputing does not always improve the predictions, so please check via
cross-validation.  Sometimes dropping rows or using marker values is
more effective.

Missing values can be replaced by the mean, the median or the most frequent
value using the ``strategy`` hyper-parameter.
The median is a more robust estimator for data with high magnitude variables
which could dominate results (otherwise known as a 'long tail').

Script output::

  Score with the entire dataset = 0.56
  Score without the samples containing missing values = 0.48
  Score after imputation of the missing values = 0.55

In this case, imputing helps the classifier get close to the original score.

"""
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", SimpleImputer(missing_values=0,
                                                strategy="mean",
                                                axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
```
149 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.21, to correspond to the standard
        definition of cross-validation.

        ..versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settins, constructed from
        param_distributions.

    
```
150 - /tmp/repos/scikit-learn/sklearn/grid_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    .. deprecated:: 0.18
        This module will be removed in 0.20.
        Use :class:`sklearn.model_selection.RandomizedSearchCV` instead.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_jobs: int, default: 1 :
        The maximum number of estimators fit in parallel.

            - If -1 all CPUs are used.

            - If 1 is given, no parallel computing code is used at all,
              which is useful for debugging.

            - For ``n_jobs`` below -1, ``(n_cpus + n_jobs + 1)`` are used.
              For example, with ``n_jobs = -2`` all CPUs but one are used.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.


    Attributes
    ----------
    grid_scores_ : list of namedtuples
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each namedtuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    
```
151 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
    accept a custom RNG instance and always use the singleton RNG from
    ``numpy.random``. Hence setting ``random_state`` will not guarantee a
    deterministic iteration whenever ``scipy.stats`` distributions are used to
    define the parameter search space.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.21, to correspond to the standard
        definition of cross-validation.

        ..versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    
```
152 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.21, to correspond to the standard
        definition of cross-validation.

        ..versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settins, constructed from
        param_distributions.

    
```
153 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
    accept a custom RNG instance and always use the singleton RNG from
    ``numpy.random``. Hence setting ``random_state`` will not guarantee a
    deterministic iteration whenever ``scipy.stats`` distributions are used to
    define the parameter search space.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.21, to correspond to the standard
        definition of cross-validation.

        .. versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    
```
154 - /tmp/repos/scikit-learn/examples/plot_johnson_lindenstrauss_bound.py
```python
from 56436 down to 10000 while reasonably preserving pairwise distances.

"""
print(__doc__)

import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances

# Part 1: plot the theoretical dependency between n_components_min and
# n_samples

# range of admissible distortions
eps_range = np.linspace(0.1, 0.99, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

# range of number of samples (observation) to embed
n_samples_range = np.logspace(1, 9, 9)

plt.figure()
for eps, color in zip(eps_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
    plt.loglog(n_samples_range, min_n_components, color=color)

plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
plt.xlabel("Number of observations to eps-embed")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Lindenstrauss bounds:
```
155 - /tmp/repos/scikit-learn/sklearn/__init__.py
```python
"""
import sys
import re
import warnings
import logging
import os

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.22.dev0'


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKLEARN_SETUP__
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write('Partial import of sklearn during the build process.\n')
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import __check_build
    from .base import clone
    from .utils._show_versions import show_versions

    __check_build  # avoid flakes unused variable error

    __all__ = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
               'datasets', 'decomposition', 'dummy', 'ensemble', 'exceptions',
               'experimental', 'externals', 'feature_extraction',
               'feature_selection', 'gaussian_process', 'inspection',
               'isotonic', 'kernel_approximation', 'kernel_ridge',
               'linear_model', 'manifold', 'metrics', 'mixture',
               'model_selection', 'multiclass', 'multioutput',
               'naive_bayes', 'neighbors', 'neural_network', 'pipeline',
               'preprocessing', 'random_projection', 'semi_supervised',
               'svm', 'tree', 'discriminant_analysis', 'impute', 'compose',
               # Non-modules:
               'clone', 'get_config', 'set_config', 'config_context',
               'show_versions']


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs
```
156 - /tmp/repos/scikit-learn/examples/ensemble/plot_forest_iris.py
```python
# ..."""
====================================================================
Plot the decision surfaces of ensembles of trees on the iris dataset
====================================================================

Plot the decision surfaces of forests of randomized trees trained on pairs of
features of the iris dataset.

This plot compares the decision surfaces learned by a decision tree classifier
(first column), by a random forest classifier (second column), by an extra-
trees classifier (third column) and by an AdaBoost classifier (fourth column).

In the first row, the classifiers are built using the sepal width and
the sepal length features only, on the second row using the petal length and
sepal length only, and on the third row using the petal width and the
petal length only.

In descending order of quality, when trained (outside of this example) on all
4 features using 30 estimators and scored using 10 fold cross validation,
we see::

    ExtraTreesClassifier()  # 0.95 score
    RandomForestClassifier()  # 0.94 score
    AdaBoost(DecisionTree(max_depth=3))  # 0.94 score
    DecisionTree(max_depth=None)  # 0.94 score

Increasing `max_depth` for AdaBoost lowers the standard deviation of
the scores (but the average score does not improve).

See the console's output for further details about each model.

In this example you might try to:

1) vary the ``max_depth`` for the ``DecisionTreeClassifier`` and
   ``AdaBoostClassifier``, perhaps try ``max_depth=3`` for the
   ``DecisionTreeClassifier`` or ``max_depth=None`` for ``AdaBoostClassifier``
2) vary ``n_estimators``

It is worth noting that RandomForests and ExtraTrees can be fitted in parallel
on many cores as each tree is built independently of the others. AdaBoost's
samples are built sequentially and so do not use multiple cores.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
iris = load_iris()

plot_idx = 1

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight"
```
157 - /tmp/repos/scikit-learn/sklearn/model_selection/_search.py
```python
# ...


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds. Default is True, but
        will change to False in version 0.21, to correspond to the standard
        definition of cross-validation.

        ..versionchanged:: 0.20
            Parameter ``iid`` will change from True to False by default in
            version 0.22, and will be removed in 0.24.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.

    :class:`ParameterSampler`:
        A generator over parameter settings, constructed from
        param_distributions.

    
```
158 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting.py
```python



n_samples_list = [1000, 10000, 100000, 500000, 1000000, 5000000, 10000000]
n_samples_list = [n_samples for n_samples in n_samples_list
                  if n_samples <= args.n_samples_max]

sklearn_scores = []
sklearn_fit_durations = []
sklearn_score_durations = []
lightgbm_scores = []
lightgbm_fit_durations = []
lightgbm_score_durations = []
xgb_scores = []
xgb_fit_durations = []
xgb_score_durations = []
cat_scores = []
cat_fit_durations = []
cat_score_durations = []

for n_samples in n_samples_list:
    (sklearn_score,
     sklearn_fit_duration,
     sklearn_score_duration,
     lightgbm_score,
     lightgbm_fit_duration,
     lightgbm_score_duration,
     xgb_score,
     xgb_fit_duration,
     xgb_score_duration,
     cat_score,
     cat_fit_duration,
     cat_score_duration) = one_run(n_samples)

    for scores, score in (
            (sklearn_scores, sklearn_score),
            (sklearn_fit_durations, sklearn_fit_duration),
            (sklearn_score_durations, sklearn_score_duration),
            (lightgbm_scores, lightgbm_score),
            (lightgbm_fit_durations, lightgbm_fit_duration),
            (lightgbm_score_durations, lightgbm_score_duration),
            (xgb_scores, xgb_score),
            (xgb_fit_durations, xgb_fit_duration),
            (xgb_score_durations, xgb_score_duration),
            (cat_scores, cat_score),
            (cat_fit_durations, cat_fit_duration),
            (cat_score_durations, cat_score_duration)):
        scores.append(score)

fig, axs = plt.subplots(3, sharex=True)

axs[0].plot(n_samples_list, sklearn_scores, label='sklearn')
axs[1].plot(n_samples_list, sklearn_fit_durations, label='sklearn')
axs[2].plot(n_samples_list, sklearn_score_durations, label='sklearn')

if args.lightgbm:
    axs[0].plot(n_samples_list, lightgbm_scores, label='lightgbm')
    axs[1].plot(n_samples_list, lightgbm_fit_durations, label='lightgbm')
    axs[2].plot(n_samples_list, lightgbm_score_durations, label='lightgbm')

if args.xgboost:
    axs[0].plot(n_samples_list, xgb_scores, label='XGBoost')
    axs[1].plot(n_samples_list, xgb_fit_durations, label='XGBoost')
    axs[2].plot(n_samples_list, xgb_score_durations, label='XGBoost')

if args.catboost:
    axs[0].plot(n_samples_list, cat_scores, label='CatBoost')
    axs[1].plot(n_samples_list, cat_fit_durations, label='CatBoost')
    axs[2].plot(n_samples_list, cat_score_durations, label='CatBoost')

for ax in axs:
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.set_xlabel('n_samples')

axs[0].set_title('scores')
axs[1].set_title('fit duration (s)')
axs[2].set_title('score duration (s)')

title = args.problem
if args.problem == 'classification':
    title += ' n_classes = {}'.format(args.n_classes)
fig.suptitle(title)


plt.tight_layout()
plt.show()

```
159 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
# ...


class RFECV(RFE, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination and cross-validated
    selection of the best number of features.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel while fitting across folds.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Notes
    -----
    The size of ``grid_scores_`` is equal to ceil((n_features - 1) / step) + 1,
    where step is the number of features removed at each iteration.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFE : Recursive feature elimination

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    
```
160 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
# ...


class RFECV(RFE, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination and cross-validated
    selection of the best number of features.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel while fitting across folds.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Notes
    -----
    The size of ``grid_scores_`` is equal to ceil((n_features - 1) / step) + 1,
    where step is the number of features removed at each iteration.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFE : Recursive feature elimination

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    
```
161 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
# ...


class RFECV(RFE, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination and cross-validated
    selection of the best number of features.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    min_features_to_select : int, (default=1)
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : string, callable or None, optional, (default=None)
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, (default=0)
        Controls verbosity of output.

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        ``grid_scores_[i]`` corresponds to
        the CV score of the i-th subset of features.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Notes
    -----
    The size of ``grid_scores_`` is equal to
    ``ceil((n_features - min_features_to_select) / step) + 1``,
    where step is the number of features removed at each iteration.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFE : Recursive feature elimination

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    
```
162 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if (self.is_data_valid is not None
                    and not self.is_data_valid(X_subset, y_subset)):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator.fit(X_subset, y_subset)
            else:
                base_estimator.fit(X_subset, y_subset,
                                   sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = base_estimator.score(X_inlier_subset,
                                                y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        
```
163 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (self.n_skips_no_inliers_ + self.n_skips_invalid_data_ +
                    self.n_skips_invalid_model_) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(n_samples, min_samples,
                                                     random_state=random_state)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if (self.is_data_valid is not None
                    and not self.is_data_valid(X_subset, y_subset)):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator.fit(X_subset, y_subset)
            else:
                base_estimator.fit(X_subset, y_subset,
                                   sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator, X_subset, y_subset)):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = base_estimator.predict(X)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset < residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = base_estimator.score(X_inlier_subset,
                                                y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if (n_inliers_subset == n_inliers_best
                    and score_subset < score_best):
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(n_inliers_best, n_samples,
                                    min_samples, self.stop_probability))

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or \
                            score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        
```
164 - /tmp/repos/scikit-learn/examples/ensemble/plot_gradient_boosting_oob.py
```python
# ..."""
======================================
Gradient Boosting Out-of-Bag estimates
======================================

Out-of-bag (OOB) estimates can be a useful heuristic to estimate
the "optimal" number of boosting iterations.
OOB estimates are almost identical to cross-validation estimates but
they can be computed on-the-fly without the need for repeated model
fitting.
OOB estimates are only available for Stochastic Gradient Boosting
(i.e. ``subsample < 1.0``), the estimates are derived from the improvement
in loss based on the examples not included in the bootstrap sample
(the so-called out-of-bag examples).
The OOB estimator is a pessimistic estimator of the true
test loss, but remains a fairly good approximation for a small number of trees.

The figure shows the cumulative sum of the negative OOB improvements
as a function of the boosting iteration. As you can see, it tracks the test
loss for the first hundred iterations but then diverges in a
pessimistic way.
The figure also shows the performance of 3-fold cross validation which
usually gives a better estimate of the test loss
but is computationally more demanding.
"""
print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# Generate data (adapted from G. Ridgeway's gbm example)
n_samples = 1000
random_state = np.random.RandomState(13)
x1 = random_state.uniform(size=n_samples)
x2 = random_state.uniform(size=n_samples)
x3 = random_state.randint(0, 4, size=n_samples)

p = 1 / (1.0 + np.exp(-(np.sin(3 * x1) - 4 * x2 + x3)))
y = random_state.binomial(1, p, size=n_samples)

X = np.c_[x1, x2, x3]

X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=9)

# Fit classifier with out-of-bag estimates
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
clf = ensemble.GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy: {:.4f}".format(acc))

n_estimators = params['n_estimators']
x = np.arange(n_estimators) + 1


def heldout_score(clf, X_test, y_test):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score



```
**165 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import struct

from sklearn.externals.six.moves import zip
from sklearn.externals.joblib import hash, Memory
from sklearn.utils.testing import assert_raises, _get_args
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import META_ESTIMATORS
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_dict_equal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, TransformerMixin, ClusterMixin,
                          BaseEstimator, is_classifier, is_regressor,
                          is_outlier_detector)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.fixes import signature
from sklearn.utils.validation import has_fit_parameter, _num_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GaussianProcess',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']
```
166 - /tmp/repos/scikit-learn/benchmarks/bench_tree.py
```python
def bench_scikit_tree_regressor(X, Y):
    """Benchmark with scikit-learn decision tree regressor"""

    from sklearn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_regressor_results.append(
        delta.seconds + delta.microseconds / mu_second)
```
167 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
class BaseGradientBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):


    @property
    @deprecated("Attribute n_features was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def n_features(self):
        return self.n_features_
```
168 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
# ...# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import gc
import time

from sklearn.externals.joblib import Memory
from sklearn.linear_model import (LogisticRegression, SGDClassifier)
from sklearn.datasets import fetch_rcv1
from sklearn.linear_model.sag import get_auto_step_size

try:
    import lightning.classification as lightning_clf
except ImportError:
    lightning_clf = None

m = Memory(cachedir='.', verbose=0)


# compute logistic loss
def get_loss(w, intercept, myX, myy, C):
    n_samples = myX.shape[0]
    w = w.ravel()
    p = np.mean(np.log(1. + np.exp(-myy * (myX.dot(w) + intercept))))
    print("%f + %f" % (p, w.dot(w) / 2. / C / n_samples))
    p += w.dot(w) / 2. / C / n_samples
    return p


# We use joblib to cache individual fits. Note that we do not pass the dataset
# as argument as the hashing would be too slow, so we assume that the dataset
# never changes.
@m.cache()
def bench_one(name, clf_type, clf_params, n_iter):
    clf = clf_type(**clf_params)
    try:
        clf.set_params(max_iter=n_iter, random_state=42)
    except:
        clf.set_params(n_iter=n_iter, random_state=42)

    st = time.time()
    clf.fit(X, y)
    end = time.time()

    try:
        C = 1.0 / clf.alpha / n_samples
    except:
        C = clf.C

    try:
        intercept = clf.intercept_
    except:
        intercept = 0.

    train_loss = get_loss(clf.coef_, intercept, X, y, C)
    train_score = clf.score(X, y)
    test_score = clf.score(X_test, y_test)
    duration = end - st

    return train_loss, train_score, test_score, duration


def bench(clfs):
    for (name, clf, iter_range, train_losses, train_scores,
         test_scores, durations) in clfs:
        print("training %s" % name)
        clf_type = type(clf)
        clf_params = clf.get_params()

        for n_iter in iter_range:
            gc.collect()

            train_loss, train_score, test_score, duration = bench_one(
                name, clf_type, clf_params, n_iter)

            train_losses.append(train_loss)
            train_scores.append(train_score)
            test_scores.append(test_score)
            durations.append(duration)
            print("classifier: %s" % name)
            print("train_loss: %.8f" % train_loss)
            print("train_score: %.8f" % train_score)
            print("test_score: %.8f" % test_score)
            print("time for fit: %.8f seconds" % duration)
            print("")

        print("")
    return clfs


def plot_train_losses(clfs):
    plt.figure()
    for (name, _, _, train_losses, _, _, durations) in clfs:
        plt.plot(durations, train_losses, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train loss")



```
**169 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...
```
**170 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
# ...
```
171 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length `2*class_sep` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Prior to shuffling, `X` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension `n_informative`. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if `len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.
        More than `n_samples` samples may be returned if the sum of `weights`
        exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See also
    --------
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    
```
172 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length `2*class_sep` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Prior to shuffling, `X` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension `n_informative`. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if `len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.
        More than `n_samples` samples may be returned if the sum of `weights`
        exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See also
    --------
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    
```
173 - /tmp/repos/scikit-learn/sklearn/random_projection.py
```python
if np.any(n_samples) <= 0:
        raise ValueError(
            "The JL bound is defined for n_samples greater than zero, got %r"
            % n_samples)

    denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int)
# ...
```
174 - /tmp/repos/scikit-learn/sklearn/random_projection.py
```python
if np.any(n_samples) <= 0:
        raise ValueError(
            "The JL bound is defined for n_samples greater than zero, got %r"
            % n_samples)

    denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int)
# ...
```
175 - /tmp/repos/scikit-learn/sklearn/linear_model/stochastic_gradient.py
```python
class BaseSGD(six.with_metaclass(ABCMeta, BaseEstimator, SparseCoefMixin)):


    def _validate_params(self, set_max_iter=True, for_partial_fit=False):
        """Validate input params. """
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be > zero. Got %f" % self.max_iter)
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if self.learning_rate in ("constant", "invscaling"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError("alpha must be > 0 since "
                             "learning_rate is 'optimal'. alpha is used "
                             "to compute the optimal learning rate.")

        # raises ValueError if not registered
        self._get_penalty_type(self.penalty)
        self._get_learning_rate_type(self.learning_rate)

        if self.loss not in self.loss_functions:
            raise ValueError("The loss %s is not supported. " % self.loss)

        if not set_max_iter:
            return
        # n_iter deprecation, set self._max_iter, self._tol
        self._tol = self.tol
        if self.n_iter is not None:
            warnings.warn("n_iter parameter is deprecated in 0.19 and will be"
                          " removed in 0.21. Use max_iter and tol instead.",
                          DeprecationWarning)
            # Same behavior as before 0.19
            max_iter = self.n_iter
            self._tol = None

        elif self.tol is None and self.max_iter is None:
            if not for_partial_fit:
                warnings.warn(
                    "max_iter and tol parameters have been "
                    "added in %s in 0.19. If both are left unset, "
                    "they default to max_iter=5 and tol=None. "
                    "If tol is not None, max_iter defaults to max_iter=1000. "
                    "From 0.21, default max_iter will be 1000, and"
                    " default tol will be 1e-3." % type(self).__name__,
                    FutureWarning)
                # Before 0.19, default was n_iter=5
            max_iter = 5
        else:
            max_iter = self.max_iter if self.max_iter is not None else 1000
        self._max_iter = max_iter
```
176 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
class RFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):


    def _more_tags(self):
        return {'poor_score': True}
```
177 - /tmp/repos/scikit-learn/examples/ensemble/plot_feature_transformation.py
```python
# ..."""
===============================================
Feature transformations with ensembles of trees
===============================================

Transform your features into a higher dimensional, sparse space. Then
train a linear model on these features.

First fit an ensemble of trees (totally random trees, a random
forest, or gradient boosted trees) on the training set. Then each leaf
of each tree in the ensemble is assigned a fixed arbitrary feature
index in a new feature space. These leaf indices are then encoded in a
one-hot fashion.

Each sample goes through the decisions of each tree of the ensemble
and ends up in one leaf per tree. The sample is encoded by setting
feature values for these leaves to 1 and the other feature values to 0.

The resulting transformer has then learned a supervised, sparse,
high-dimensional categorical embedding of the data.

"""

# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import CategoricalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = CategoricalEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = CategoricalEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, 
```
178 - /tmp/repos/scikit-learn/sklearn/tree/tree.py
```python
class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator)):


    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X
```
**179 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_int(name, regressor_orig):
    X, _ = _boston_subset()
    X = pairwise_estimator_convert_X(X[:50], regressor_orig)
    rnd = np.random.RandomState(0)
    y = rnd.randint(3, size=X.shape[0])
    y = multioutput_estimator_convert_y_2d(regressor_orig, y)
    rnd = np.random.RandomState(0)
    # separate estimators to control random seeds
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    # fit
    regressor_1.fit(X, y_)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y_.astype(np.float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)
```
180 - /tmp/repos/scikit-learn/examples/ensemble/plot_forest_importances_faces.py
```python
"""
=================================================
Pixel importances with a parallel forest of trees
=================================================

This example shows the use of forests of trees to evaluate the importance
of the pixels in an image classification task (faces). The hotter the pixel,
the more important.

The code below also illustrates how the construction and the computation
of the predictions can be parallelized within multiple jobs.
"""
print(__doc__)

from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

# Number of cores to use to perform parallel fitting of the forest model
n_jobs = 1

# Load the faces dataset
data = fetch_olivetti_faces()
X = data.images.reshape((len(data.images), -1))
y = data.target

mask = y < 5  # Limit to 5 classes
X = X[mask]
y = y[mask]

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(X, y)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.show()
```
181 - /tmp/repos/scikit-learn/benchmarks/bench_20newsgroups.py
```python
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=ESTIMATORS)
    args = vars(parser.parse_args())

    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = check_array(data_train.data, dtype=np.float32,
                          accept_sparse="csc")
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")
    y_train = data_train.target
    y_test = data_test.target

    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train.shape))
    print("X_train.format = {0}".format(X_train.format))
    print("X_train.dtype = {0}".format(X_train.dtype))
    print("X_train density = {0}"
          "".format(X_train.nnz / np.product(X_train.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test.shape))
    print("X_test.format = {0}".format(X_test.format))
    print("X_test.dtype = {0}".format(X_test.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}
    for name in sorted(args["estimators"]):
        clf = ESTIMATORS[name]
        try:
            clf.set_params(random_state=0)
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time",
                           "Accuracy"))
    print("-" * 44)
    for name in sorted(accuracy, key=accuracy.get):
        print("%s %s %s %s" % (name.ljust(16),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % accuracy[name]).center(10)))

    print()

```
182 - /tmp/repos/scikit-learn/benchmarks/bench_20newsgroups.py
```python
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=ESTIMATORS)
    args = vars(parser.parse_args())

    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = check_array(data_train.data, dtype=np.float32,
                          accept_sparse="csc")
    X_test = check_array(data_test.data, dtype=np.float32, accept_sparse="csr")
    y_train = data_train.target
    y_test = data_test.target

    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train.shape))
    print("X_train.format = {0}".format(X_train.format))
    print("X_train.dtype = {0}".format(X_train.dtype))
    print("X_train density = {0}"
          "".format(X_train.nnz / np.product(X_train.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test.shape))
    print("X_test.format = {0}".format(X_test.format))
    print("X_test.dtype = {0}".format(X_test.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Classifier Training")
    print("===================")
    accuracy, train_time, test_time = {}, {}, {}
    for name in sorted(args["estimators"]):
        clf = ESTIMATORS[name]
        try:
            clf.set_params(random_state=0)
        except (TypeError, ValueError):
            pass

        print("Training %s ... " % name, end="")
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time",
                           "Accuracy"))
    print("-" * 44)
    for name in sorted(accuracy, key=accuracy.get):
        print("%s %s %s %s" % (name.ljust(16),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % accuracy[name]).center(10)))

    print()

```
183 - /tmp/repos/scikit-learn/examples/calibration/plot_compare_calibration.py
```python
# ..."""
========================================
Comparison of Calibration of Classifiers
========================================

Well calibrated classifiers are probabilistic classifiers for which the output
of the predict_proba method can be directly interpreted as a confidence level.
For instance a well calibrated (binary) classifier should classify the samples
such that among the samples to which it gave a predict_proba value close to
0.8, approx. 80% actually belong to the positive class.

LogisticRegression returns well calibrated predictions as it directly
optimizes log-loss. In contrast, the other methods return biased probabilities,
with different biases per method:

* GaussianNaiveBayes tends to push probabilities to 0 or 1 (note the counts in
  the histograms). This is mainly because it makes the assumption that features
  are conditionally independent given the class, which is not the case in this
  dataset which contains 2 redundant features.

* RandomForestClassifier shows the opposite behavior: the histograms show
  peaks at approx. 0.2 and 0.9 probability, while probabilities close to 0 or 1
  are very rare. An explanation for this is given by Niculescu-Mizil and Caruana
  [1]: "Methods such as bagging and random forests that average predictions from
  a base set of models can have difficulty making predictions near 0 and 1
  because variance in the underlying base models will bias predictions that
  should be near zero or one away from these values. Because predictions are
  restricted to the interval [0,1], errors caused by variance tend to be one-
  sided near zero and one. For example, if a model should predict p = 0 for a
  case, the only way bagging can achieve this is if all bagged trees predict
  zero. If we add noise to the trees that bagging is averaging over, this noise
  will cause some trees to predict values larger than 0 for this case, thus
  moving the average prediction of the bagged ensemble away from 0. We observe
  this effect most strongly with random forests because the base-level trees
  trained with random forests have relatively high variance due to feature
  subsetting." As a result, the calibration curve shows a characteristic
  sigmoid shape, indicating that the classifier could trust its "intuition"
  more and return probabilities closer to 0 or 1 typically.

* Support Vector Classification (SVC) shows an even more sigmoid curve as
  the  RandomForestClassifier, which is typical for maximum-margin methods
  (compare Niculescu-Mizil and Caruana [1]), which focus on hard samples
  that are close to the decision boundary (the support vectors).

.. topic:: References:

    .. [1] Predicting Good Probabilities with Supervised Learning,
          A. Niculescu-Mizil & R. Caruana, ICML 2005

```
**184 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class BaseForest(six.with_metaclass(ABCMeta, BaseEnsemble)):


    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        check_is_fitted(self, 'estimators_')

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   backend="threading")(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_)

        return sum(all_importances) / len(self.estimators_)
```
185 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
# ...


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See also
    --------
    make_blobs: simplified variant
    make_multilabel_classification: unrelated generator for multilabel tasks
    
```
186 - /tmp/repos/scikit-learn/benchmarks/bench_plot_randomized_svd.py
```python
# ..."""
Benchmarks on the power iterations phase in randomized SVD.

We test on various synthetic and real datasets the effect of increasing
the number of power iterations in terms of quality of approximation
and running time. A number greater than 0 should help with noisy matrices,
which are characterized by a slow spectral decay.

We test several policy for normalizing the power iterations. Normalization
is crucial to avoid numerical issues.

The quality of the approximation is measured by the spectral norm discrepancy
between the original input matrix and the reconstructed one (by multiplying
the randomized_svd's outputs). The spectral norm is always equivalent to the
largest singular value of a matrix. (3) justifies this choice. However, one can
notice in these experiments that Frobenius and spectral norms behave
very similarly in a qualitative sense. Therefore, we suggest to run these
benchmarks with `enable_spectral_norm = False`, as Frobenius' is MUCH faster to
compute.

The benchmarks follow.

(a) plot: time vs norm, varying number of power iterations
    data: many datasets
    goal: compare normalization policies and study how the number of power
    iterations affect time and norm

(b) plot: n_iter vs norm, varying rank of data and number of components for
    randomized_SVD
    data: low-rank matrices on which we control the rank
    goal: study whether the rank of the matrix and the number of components
    extracted by randomized SVD affect "the optimal" number of power iterations

(c) plot: time vs norm, varing datasets
    data: many datasets
    goal: compare default configurations

We compare the following algorithms:
-   randomized_svd(..., power_iteration_normalizer='none')
-   randomized_svd(..., power_iteration_normalizer='LU')
-   randomized_svd(..., power_iteration_normalizer='QR')
-   randomized_svd(..., power_iteration_normalizer='auto')
-   fbpca.pca() from https://github.com/facebook/fbpca (if installed)

Conclusion
----------
- n_iter=2 appears to be a good default value
- power_iteration_normalizer='none' is OK if n_iter is small, otherwise LU
  gives similar errors to QR but is cheaper. That's what 'auto' implements.

References
----------
(1) Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

(2) A randomized algorithm for the decomposition of matrices
    Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

(3) An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
"""

# Author: Giorgio Patrini

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import gc
import pickle
from time import time
from collections import defaultdict
import os.path

from sklearn.utils import gen_batches
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets.samples_generator import (make_low_rank_matrix,
                                                make_sparse_uncorrelated)
from sklearn.datasets import (fetch_lfw_people,
                              fetch_mldata,
                              fetch_20newsgroups_vectorized,
                              fetch_olivetti_faces,
                              fetch_rcv1)

try:
    import fbpca
    fbpca_available = True
except ImportError:
    fbpca_available = False

# If this is enabled, tests are much slower and will crash with the large data
enable_spectral_norm = False

# TODO: compute approximate spectral norms with the power method as in
# Estimating the largest eigenvalues by the power and Lanczos methods with
# a random start, Jacek Kuczynski and Henryk Wozniakowski, SIAM Journal on
# Matrix Analysis and Applications, 13 (4): 1094-1122, 1992.
# This approximation is a very fast estimate of the spectral norm, but depends
# on starting random vectors.

# Determine when to switch to batch computation for matrix norms,
# in case the reconstructed (dense) matrix is too large
MAX_MEMORY = np.int(2e9)

# The following datasets can be dowloaded manually from:
# CIFAR 10: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# SVHN: http://ufldl.stanford.edu/housenumbers/train_32x32.mat
CIFAR_FOLDER = "./cifar-10-batches-py/"
SVHN_FOLDER = "./SVHN/"

datasets = ['low rank matrix', 'lfw_people', 'olivetti_faces', '20newsgroups',
            'MNIST original', 'CIFAR', 'a1a', 'SVHN', 'uncorrelated matrix']

big_sparse_datasets = ['big sparse matrix', 'rcv1']


def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')["data"]


def handle_missing_dataset(file_folder):
    if not os.path.isdir(file_folder):
        print("%s file folder not found. Test skipped." % file_folder)
        return 0


def get_data(dataset_name):
    print("Getting dataset: %s" % dataset_name)

    if dataset_name == 'lfw_people':
        X = fetch_lfw_people().data
    elif dataset_name == '20newsgroups':
        X = fetch_20newsgroups_vectorized().data[:, :100000]
    elif dataset_name == 'olivetti_faces':
        X = fetch_olivetti_faces().data
    elif dataset_name == 'rcv1':
        X = fetch_rcv1().data
    elif dataset_name == 'CIFAR':
        if handle_missing_dataset(CIFAR_FOLDER) == "skip":
            return
        X1 = [unpickle("%sdata_batch_%d" % (CIFAR_FOLDER, i + 1))
              for i in range(5)]
        X = np.vstack(X1)
        del X1
    elif dataset_name == 'SVHN':
        if handle_missing_dataset(SVHN_FOLDER) == 0:
            return
        X1 = sp.io.loadmat("%strain_32x32.mat" % SVHN_FOLDER)['X']
        X2 = [X1[:, :, :, i].reshape(32 * 32 * 3) for i in range(X1.shape[3])]
        X = np.vstack(X2)
        del X1
        del X2
    elif dataset_name == 'low rank matrix':
        X = make_low_rank_matrix(n_samples=500, n_features=np.int(1e4),
                                 effective_rank=100, tail_strength=.5,
                                 random_state=random_state)
    elif dataset_name == 'uncorrelated matrix':
        X, _ = make_sparse_uncorrelated(n_samples=500, n_features=10000,
                                        random_state=random_state)
    elif dataset_name == 'big sparse matrix':
        sparsity = np.int(1e6)
        size = np.int(1e6)
        small_size = np.int(1e4)
        data = np.random.normal(0, 1, np.int(sparsity/10))
        data = np.repeat(data, 10)
        row = np.random.uniform(0, small_size, sparsity)
        col = np.random.uniform(0, small_size, sparsity)
        X = sp.sparse.csr_matrix((data, (row, col)), shape=(size, small_size))
        del data
        del row
        del col
    else:
        X = fetch_mldata(dataset_name).data
    return X


def plot_time_vs_s(time, norm, point_labels, title):
    plt.figure()
    colors = ['g', 'b', 'y']
    for i, l in enumerate(sorted(norm.keys())):
        if l != "fbpca":
            plt.plot(time[l], norm[l], label=l, marker='o', c=colors.pop())
        else:
            plt.plot(time[l], norm[l], label=l, marker='^', c='red')

        for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
            plt.annotate(label, xy=(x, y), xytext=(0, -20),
                         textcoords='offset points', ha='right', va='bottom')
    plt.legend(loc="upper right")
    plt.suptitle(title)
    plt.ylabel("norm discrepancy")
    plt.xlabel("running time [s]")


def scatter_time_vs_s(time, norm, point_labels, title):
    plt.figure()
    size = 100
    for i, l in enumerate(sorted(norm.keys())):
        if l != "fbpca":
            plt.scatter(time[l], norm[l], label=l, marker='o', c='b', s=size)
            for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
                plt.annotate(label, xy=(x, y), xytext=(0, -80),
                             textcoords='offset points', ha='right',
                             arrowprops=dict(arrowstyle="->",
                                             connectionstyle="arc3"),
                             va='bottom', size=11, rotation=90)
        else:
            plt.scatter(time[l], norm[l], label=l, marker='^', c='red', s=size)
            for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
                plt.annotate(label, xy=(x, y), xytext=(0, 30),
                             textcoords='offset points', ha='right',
                             arrowprops=dict(arrowstyle="->",
                                             connectionstyle="arc3"),
                             va='bottom', size=11, rotation=90)

    plt.legend(loc="best")
    plt.suptitle(title)
    plt.ylabel("norm discrepancy")
    plt.xlabel("running time [s]")


def plot_power_iter_vs_s(power_iter, s, title):
    plt.figure()
    for l in sorted(s.keys()):
        plt.plot(power_iter, s[l], label=l, marker='o')
    plt.legend(loc="lower right", prop={'size': 10})
    plt.suptitle(title)
    plt.ylabel("norm discrepancy")
    plt.xlabel("n_iter")


def svd_timing(X, n_comps, n_iter, n_oversamples,
               power_iteration_normalizer='auto', method=None):
    
```
187 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
class BaseGradientBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):


    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_.resize((total_n_estimators, self.loss_.K))
        self.train_score_.resize(total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_.resize(total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)
```
**188 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_set_params(name, estimator_orig):
    # Check that get_params() returns the same thing
    # before and after set_params() with some fuzz
    estimator = clone(estimator_orig)

    orig_params = estimator.get_params(deep=False)
    msg = ("get_params result does not match what was passed to set_params")

    estimator.set_params(**orig_params)
    curr_params = estimator.get_params(deep=False)
    assert set(orig_params.keys()) == set(curr_params.keys()), msg
    for k, v in curr_params.items():
        assert orig_params[k] is v, msg

    # some fuzz values
    test_values = [-np.inf, np.inf, None]

    test_params = deepcopy(orig_params)
    for param_name in orig_params.keys():
        default_value = orig_params[param_name]
        for value in test_values:
            test_params[param_name] = value
            try:
                estimator.set_params(**test_params)
            except (TypeError, ValueError) as e:
                e_type = e.__class__.__name__
                # Exception occurred, possibly parameter validation
                warnings.warn("{0} occurred during set_params of param {1} on "
                              "{2}. It is recommended to delay parameter "
                              "validation until fit.".format(e_type,
                                                             param_name,
                                                             name))

                change_warning_msg = "Estimator's parameters changed after " \
                                     "set_params raised {}".format(e_type)
                params_before_exception = curr_params
                curr_params = estimator.get_params(deep=False)
                try:
                    assert (set(params_before_exception.keys()) ==
                                 set(curr_params.keys()))
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                curr_params = estimator.get_params(deep=False)
                assert (set(test_params.keys()) ==
                        set(curr_params.keys())), msg
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg
        test_params[param_name] = default_value
```
189 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
class IsolationForest(BaseBagging, OutlierMixin):


    def _parallel_args(self):
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies.
        return _joblib_parallel_args(prefer='threads')
```
**190 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
def check_outliers_train(name, estimator_orig):
    X, _ = make_blobs(n_samples=300, random_state=0)
    X = shuffle(X, random_state=7)
    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    # fit
    estimator.fit(X)
    # with lists
    estimator.fit(X.tolist())

    y_pred = estimator.predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == 'i'
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    decision = estimator.decision_function(X)
    assert decision.dtype == np.dtype('float')

    score = estimator.score_samples(X)
    assert score.dtype == np.dtype('float')

    # raises error on malformed input for predict
    assert_raises(ValueError, estimator.predict, X.T)

    # decision_function agrees with predict
    decision = estimator.decision_function(X)
    assert decision.shape == (n_samples,)
    dec_pred = (decision >= 0).astype(np.int)
    dec_pred[dec_pred == 0] = -1
    assert_array_equal(dec_pred, y_pred)

    # raises error on malformed input for decision_function
    assert_raises(ValueError, estimator.decision_function, X.T)

    # decision_function is a translation of score_samples
    y_scores = estimator.score_samples(X)
    assert y_scores.shape == (n_samples,)
    y_dec = y_scores - estimator.offset_
    assert_array_equal(y_dec, decision)

    # raises error on malformed input for score_samples
    assert_raises(ValueError, estimator.score_samples, X.T)

    # contamination parameter (not for OneClassSVM which has the nu parameter)
    if hasattr(estimator, "contamination"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'
        contamination = 0.1
        estimator.set_params(contamination=contamination)
        estimator.fit(X)
        y_pred = estimator.predict(X)
        assert_almost_equal(np.mean(y_pred != 1), contamination)

        # raises error when contamination is a scalar and not in [0,1]
        for contamination in [-0.5, 2.3]:
            estimator.set_params(contamination=contamination)
            assert_raises(ValueError, estimator.fit, X)
```
191 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
"""
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")

        oob_decision_function = (predictions /
                                 predictions.sum(axis=1)[:, np.newaxis])
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    
```
192 - /tmp/repos/scikit-learn/benchmarks/bench_mnist.py
```python
# ..."""
=======================
MNIST dataset benchmark
=======================

Benchmark on the MNIST dataset.  The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes -  digits from 0 to 9 from their raw images. By contrast to the
covertype dataset, the feature space is homogenous.

Example of output :
    [..]

    Classification performance:
    ===========================
    Classifier               train-time   test-time   error-rate
    ------------------------------------------------------------
    MLP_adam                     53.46s       0.11s       0.0224
    Nystroem-SVM                112.97s       0.92s       0.0228
    MultilayerPerceptron         24.33s       0.14s       0.0287
    ExtraTrees                   42.99s       0.57s       0.0294
    RandomForest                 42.70s       0.49s       0.0318
    SampledRBF-SVM              135.81s       0.56s       0.0486
    LinearRegression-SAG         16.67s       0.06s       0.0824
    CART                         20.69s       0.02s       0.1219
    dummy                         0.00s       0.01s       0.8973
"""
from __future__ import division, print_function

# Author: Issam H. Laradji
#         Arnaud Joly <arnaud.v.joly@gmail.com>
# License: BSD 3 clause

import os
from time import time
import argparse
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.datasets import get_data_home
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.externals.joblib import Memory
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'mnist_benchmark_data'),
                mmap_mode='r')


@memory.cache
def load_data(dtype=np.float32, order='F'):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = check_array(data['data'], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


ESTIMATORS = {
    "dummy": DummyClassifier(),
    'CART': DecisionTreeClassifier(),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'Nystroem-SVM': make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'SampledRBF-SVM': make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'LogisticRegression-SAG': LogisticRegression(solver='sag', tol=1e-1,
                                                 C=1e4),
    'LogisticRegression-SAGA': LogisticRegression(solver='saga', tol=1e-1,
                                                  C=1e4),
    'MultilayerPerceptron': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='sgd', learning_rate_init=0.2, momentum=0.9, verbose=1,
        tol=1e-4, random_state=1),
    'MLP-adam': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='adam', learning_rate_init=0.001, verbose=1,
        tol=1e-4, random_state=1)
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifiers', nargs="+",
                        choices=ESTIMATORS, type=str,
                        default=['ExtraTrees', 'Nystroem-SVM'],
                        help="list of classifiers to benchmark.")
    parser.add_argument('--n-jobs', nargs="?", default=1, type=int,
                        help="Number of concurrently running workers for "
                             "models that support parallelism.")
    parser.add_argument('--order', nargs="?", default="C", type=str,
                        choices=["F", "C"],
                        help="Allow to choose between fortran and C ordered "
                             "data")
    parser.add_argument('--random-seed', nargs="?", default=0, type=int,
                        help="Common seed used by random number generator.")
    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (size=%dMB)" % ("number of train samples:".ljust(25),
                                 X_train.shape[0], int(X_train.nbytes / 1e6)))
    print("%s %d (size=%dMB)" % ("number of test samples:".ljust(25),
                                 X_test.shape[0], int(X_test.nbytes / 1e6)))

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        estimator.set_params(**{p: args["random_seed"]
                                for p in estimator_params
                                if p.endswith("random_state")})

        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("{0: <24} {1: >10} {2: >11} {3: >12}"
          "".format("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 60)
    for name in sorted(args["classifiers"], key=error.get):

        print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
              "".format(name, train_time[name], test_time[name], error[name]))

    print()

```
193 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1.0, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

    @property
    @deprecated("Attribute n_iter_final was deprecated in version 0.19 and "
                "will be removed in 0.21. Use ``n_iter_`` instead")
    def n_iter_final(self):
        return self.n_iter_

    
```
194 - /tmp/repos/scikit-learn/examples/tree/plot_tree_regression_multioutput.py
```python
"""
===================================================================
Multi-output Decision Tree Regression
===================================================================

An example to illustrate multi-output regression with decision tree.

The :ref:`decision trees <tree>`
is used to predict simultaneously the noisy x and y observations of a circle
given a single underlying feature. As a result, it learns local linear
regressions approximating the circle.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# Predict
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Plot the results
plt.figure()
s = 25
plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
            edgecolor="black", label="data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s,
            edgecolor="black", label="max_depth=2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s,
            edgecolor="black", label="max_depth=5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s,
            edgecolor="black", label="max_depth=8")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()
```
195 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
class RFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):


    def _fit(self, X, y, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self
```
196 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1.0, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

    @property
    @deprecated("Attribute n_iter_final was deprecated in version 0.19 and "
                "will be removed in 0.21. Use ``n_iter_`` instead")
    def n_iter_final(self):
        return self.n_iter_

    
```
197 - /tmp/repos/scikit-learn/sklearn/datasets/samples_generator.py
```python
def make_friedman1(n_samples=100, n_features=10, noise=0.0, random_state=None):
    """Generate the "Friedman #1" regression problem

    This dataset is described in Friedman [1] and Breiman [2].

    Inputs `X` are independent features uniformly distributed on the interval
    [0, 1]. The output `y` is created according to the formula::

        y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
     * X[:, 3] + 5 * X[:, 4] + noise * N(0, 1).

    Out of the `n_features` features, only 5 are actually used to compute
    `y`. The remaining features are independent of `y`.

    The number of features has to be >= 5.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least 5.

    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.

    References
    ----------
    .. [1] J. Friedman, "Multivariate adaptive regression splines", The Annals
           of Statistics 19 (1), pages 1-67, 1991.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning 24,
           pages 123-140, 1996.
    """
    if n_features < 5:
        raise ValueError("n_features must be at least five.")

    generator = check_random_state(random_state)

    X = generator.rand(n_samples, n_features)
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
        + 10 * X[:, 3] + 5 * X[:, 4] + noise * generator.randn(n_samples)

    return X, y
```
198 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting_higgsboson.py
```python
# ...from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import argparse

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.utils import (
    get_equivalent_estimator)


parser = argparse.ArgumentParser()
parser.add_argument('--n-leaf-nodes', type=int, default=31)
parser.add_argument('--n-trees', type=int, default=10)
parser.add_argument('--lightgbm', action="store_true", default=False)
parser.add_argument('--xgboost', action="store_true", default=False)
parser.add_argument('--catboost', action="store_true", default=False)
parser.add_argument('--learning-rate', type=float, default=1.)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--max-bins', type=int, default=255)
args = parser.parse_args()

HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')

n_leaf_nodes = args.n_leaf_nodes
n_trees = args.n_trees
subsample = args.subsample
lr = args.learning_rate
max_bins = args.max_bins


@m.cache
def load_data():
    filename = os.path.join(HERE, URL.rsplit('/', 1)[-1])
    if not os.path.exists(filename):
        print(f"Downloading {URL} to {filename} (2.6 GB)...")
        urlretrieve(URL, filename)
        print("done.")

    print(f"Parsing {filename}...")
    tic = time()
    with GzipFile(filename) as f:
        df = pd.read_csv(f, header=None, dtype=np.float32)
    toc = time()
    print(f"Loaded {df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")
    return df


df = load_data()
target = df.values[:, 0]
data = np.ascontiguousarray(df.values[:, 1:])
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.2, random_state=0)

if subsample is not None:
    data_train, target_train = data_train[:subsample], target_train[:subsample]

n_samples, n_features = data_train.shape
print(f"Training set with {n_samples} records with {n_features} features.")

print("Fitting a sklearn model...")
tic = time()
est = HistGradientBoostingClassifier(loss='binary_crossentropy',
                                     learning_rate=lr,
                                     max_iter=n_trees,
                                     max_bins=max_bins,
                                     max_leaf_nodes=n_leaf_nodes,
                                     n_iter_no_change=None,
                                     random_state=0,
                                     verbose=1)
est.fit(data_train, target_train)
toc = time()
predicted_test = est.predict(data_test)
predicted_proba_test = est.predict_proba(data_test)
roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
acc = accuracy_score(target_test, predicted_test)

```
199 - /tmp/repos/scikit-learn/sklearn/manifold/t_sne.py
```python
if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

    @property
    @deprecated("Attribute n_iter_final was deprecated in version 0.19 and "
                "will be removed in 0.21. Use ``n_iter_`` instead")
    def n_iter_final(self):
        return self.n_iter_

    
```
200 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
"""Gradient Boosted Regression Trees

This module contains methods for fitting gradient boosted regression trees for
both classification and regression.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  regression problems.
"""

# Authors: Peter Prettenhofer, Scott White, Gilles Louppe, Emanuele Olivetti,
#          Arnaud Joly, Jacob Schreiber
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from .base import BaseEnsemble
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..externals import six

from ._gradient_boosting import predict_stages
from ._gradient_boosting import predict_stage
from ._gradient_boosting import _random_sample_mask

import numbers
import numpy as np

from scipy import stats
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.special import expit

from time import time
from ..model_selection import train_test_split
from ..tree.tree import DecisionTreeRegressor
from ..tree._tree import DTYPE
from ..tree._tree import TREE_LEAF

from ..utils import check_random_state
from ..utils import check_array
from ..utils import check_X_y
from ..utils import column_or_1d
from ..utils import check_consistent_length
from ..utils import deprecated
from ..utils.fixes import logsumexp
from ..utils.stats import _weighted_percentile
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..exceptions import NotFittedError



LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'lad': LeastAbsoluteError,
                  'huber': HuberLossFunction,
                  'quantile': QuantileLossFunction,
                  'deviance': None,    # for both, multinomial and binomial
                  'exponential': ExponentialLoss,
                  }


INIT_ESTIMATORS = {'zero': ZeroEstimator}
```
201 - /tmp/repos/scikit-learn/benchmarks/bench_random_projections.py
```python
# ..."""
===========================
Random projection benchmark
===========================

Benchmarks for random projections.

"""
from __future__ import division
from __future__ import print_function

import gc
import sys
import optparse
from datetime import datetime
import collections

import numpy as np
import scipy.sparse as sp

from sklearn import clone
from sklearn.externals.six.moves import xrange
from sklearn.random_projection import (SparseRandomProjection,
                                       GaussianRandomProjection,
                                       johnson_lindenstrauss_min_dim)


def type_auto_or_float(val):
    if val == "auto":
        return "auto"
    else:
        return float(val)


def type_auto_or_int(val):
    if val == "auto":
        return "auto"
    else:
        return int(val)


def compute_time(t_start, delta):
    mu_second = 0.0 + 10 ** 6  # number of microseconds in a second

    return delta.seconds + delta.microseconds / mu_second


def bench_scikit_transformer(X, transfomer):
    gc.collect()

    clf = clone(transfomer)

    # start time
    t_start = datetime.now()
    clf.fit(X)
    delta = (datetime.now() - t_start)
    # stop time
    time_to_fit = compute_time(t_start, delta)

    # start time
    t_start = datetime.now()
    clf.transform(X)
    delta = (datetime.now() - t_start)
    # stop time
    time_to_transform = compute_time(t_start, delta)

    return time_to_fit, time_to_transform


# Make some random data with uniformly located non zero entries with
# Gaussian distributed values
def make_sparse_random_data(n_samples, n_features, n_nonzeros,
                            random_state=None):
    rng = np.random.RandomState(random_state)
    data_coo = sp.coo_matrix(
        (rng.randn(n_nonzeros),
        (rng.randint(n_samples, size=n_nonzeros),
         rng.randint(n_features, size=n_nonzeros))),
        shape=(n_samples, n_features))
    return data_coo.toarray(), data_coo.tocsr()


def print_row(clf_type, time_fit, time_transform):
    print("%s | %s | %s" % (clf_type.ljust(30),
                           ("%.4fs" % time_fit).center(12),
                           ("%.4fs" % time_transform).center(12)))


if __name__ == "__main__":
    ###########################################################################
    # Option parser
    ###########################################################################
    op = optparse.OptionParser()
    op.add_option("--n-times",
                  dest="n_times", default=5, type=int,
                  help="Benchmark results are average over n_times experiments")

    op.add_option("--n-features",
                  dest="n_features", default=10 ** 4, type=int,
                  help="Number of features in the benchmarks")

    op.add_option("--n-components",
                  dest="n_components", default="auto",
                  help="Size of the random subspace."
                       " ('auto' or int > 0)")

    op.add_option("--ratio-nonzeros",
                  dest="ratio_nonzeros", default=10 ** -3, type=float,
                  help="Number of features in the benchmarks")

    op.add_option("--n-samples",
                  dest="n_samples", default=500, type=int,
                  help="Number of samples in the benchmarks")

    op.add_option("--random-seed",
                  dest="random_seed", default=13, type=int,
                  help="Seed used by the random number generators.")

    op.add_option("--density",
                  dest="density", default=1 / 3,
                  help="Density used by the sparse random projection."
                       " ('auto' or float (0.0, 1.0]")

    op.add_option("--eps",
                  dest="eps", default=0.5, type=float,
                  help="See the documentation of the underlying transformers.")

    op.add_option("--transformers",
                  dest="selected_transformers",
                  default='GaussianRandomProjection,SparseRandomProjection',
                  type=str,
                  help="Comma-separated list of transformer to benchmark. "
                       "Default: %default. Available: "
                       "GaussianRandomProjection,SparseRandomProjection")

    op.add_option("--dense",
                  dest="dense",
                  default=False,
                  action="store_true",
                  help="Set input space as a dense matrix.")

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    opts.n_components = type_auto_or_int(opts.n_components)
    opts.density = type_auto_or_float(opts.density)
    selected_transformers = opts.selected_transformers.split(',')

    ###########################################################################
    # Generate dataset
    ###########################################################################
    n_nonzeros = int(opts.ratio_nonzeros * opts.n_features)

    print('Dataset statics')
    print("===========================")
    print('n_samples 
```
**202 - /tmp/repos/scikit-learn/sklearn/utils/estimator_checks.py**:
```python
@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_32 = pairwise_estimator_convert_X(X_train_32, estimator_orig)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = X_train_int_64[:, 0]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        if name == 'PowerTransformer':
            # Box-Cox requires positive, non-zero data
            X_train = np.abs(X_train) + 1
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)
```
203 - /tmp/repos/scikit-learn/sklearn/ensemble/voting.py
```python
# ...


class VotingClassifier(_BaseVoting, ClassifierMixin):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <voting_classifier>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``None`` or ``'drop'``
        using ``set_params``.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like, shape (n_classifiers,), optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    flatten_transform : bool, optional (default=True)
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not `None`.

    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    classes_ : array-like, shape (n_predictions,)
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)

    See also
    --------
    VotingRegressor: Prediction voting regressor.
    
```
204 - /tmp/repos/scikit-learn/sklearn/ensemble/iforest.py
```python
class IsolationForest(BaseBagging, OutlierMixin):


    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")
```
205 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/grower.py
```python
if should_split_left or should_split_right:

            # We will compute the histograms of both nodes even if one of them
            # is a leaf, since computing the second histogram is very cheap
            # (using histogram subtraction).
            n_samples_left = left_child_node.sample_indices.shape[0]
            n_samples_right = right_child_node.sample_indices.shape[0]
            if n_samples_left < n_samples_right:
                smallest_child = left_child_node
                largest_child = right_child_node
            else:
                smallest_child = right_child_node
                largest_child = left_child_node

            # We use the brute O(n_samples) method on the child that has the
            # smallest number of samples, and the subtraction trick O(n_bins)
            # on the other one.
            tic = time()
            smallest_child.histograms = \
                self.histogram_builder.compute_histograms_brute(
                    smallest_child.sample_indices)
            largest_child.histograms = \
                self.histogram_builder.compute_histograms_subtraction(
                    node.histograms, smallest_child.histograms)
            self.total_compute_hist_time += time() - tic

            tic = time()
            if should_split_left:
                self._compute_best_split_and_push(left_child_node)
            if should_split_right:
                self._compute_best_split_and_push(right_child_node)
            self.total_find_split_time += time() - tic

        return left_child_node, right_child_node

    def _finalize_leaf(self, node):
        """Compute the prediction value that minimizes the objective function.

        This sets the node.value attribute (node is a leaf iff node.value is
        not None).

        See Equation 5 of:
        XGBoost: A Scalable Tree Boosting System, T. Chen, C. Guestrin, 2016
        https://arxiv.org/abs/1603.02754
        """
        node.value = -self.shrinkage * node.sum_gradients / (
            node.sum_hessians + self.splitter.l2_regularization + EPS)
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self, bin_thresholds=None):
        """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        bin_thresholds : array-like of floats, optional (default=None)
            The actual thresholds values of each bin.

        Returns
        -------
        A TreePredictor object.
        """
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        _fill_predictor_node_array(predictor_nodes, self.root,
                                   bin_thresholds=bin_thresholds)
        return TreePredictor(predictor_nodes)
# ...
```
206 - /tmp/repos/scikit-learn/benchmarks/bench_sgd_regression.py
```python
# ...# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

import gc

from time import time

from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets.samples_generator import make_regression

"""
Benchmark for SGD regression

Compares SGD regression against coordinate descent and Ridge
on synthetic data.
"""

print(__doc__)

if __name__ == "__main__":
    list_n_samples = np.linspace(100, 10000, 5).astype(np.int)
    list_n_features = [10, 100, 1000]
    n_test = 1000
    max_iter = 1000
    noise = 0.1
    alpha = 0.01
    sgd_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    elnet_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    ridge_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    asgd_results = np.zeros((len(list_n_samples), len(list_n_features), 2))
    
```
207 - /tmp/repos/scikit-learn/examples/ensemble/plot_feature_transformation.py
```python
ensembles of trees
===============================================

Transform your features into a higher dimensional, sparse space. Then
train a linear model on these features.

First fit an ensemble of trees (totally random trees, a random
forest, or gradient boosted trees) on the training set. Then each leaf
of each tree in the ensemble is assigned a fixed arbitrary feature
index in a new feature space. These leaf indices are then encoded in a
one-hot fashion.

Each sample goes through the decisions of each tree of the ensemble
and ends up in one leaf per tree. The sample is encoded by setting
feature values for these leaves to 1 and the other feature values to 0.

The resulting transformer has then learned a supervised, sparse,
high-dimensional categorical embedding of the data.

"""

# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression(max_iter=1000)
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# Supervised transformation based on gradient boosted trees
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression(max_iter=1000)
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, 
```
208 - /tmp/repos/scikit-learn/benchmarks/bench_sample_without_replacement.py
```python
# ..."""
Benchmarks for sampling without replacement of integer.

"""
from __future__ import division
from __future__ import print_function

import gc
import sys
import optparse
from datetime import datetime
import operator

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.externals.six.moves import xrange
from sklearn.utils.random import sample_without_replacement


def compute_time(t_start, delta):
    mu_second = 0.0 + 10 ** 6  # number of microseconds in a second

    return delta.seconds + delta.microseconds / mu_second


def bench_sample(sampling, n_population, n_samples):
    gc.collect()
    # start time
    t_start = datetime.now()
    sampling(n_population, n_samples)
    delta = (datetime.now() - t_start)
    # stop time
    time = compute_time(t_start, delta)
    return time

if __name__ == "__main__":
    ###########################################################################
    # Option parser
    ###########################################################################
    op = optparse.OptionParser()
    op.add_option("--n-times",
                  dest="n_times", default=5, type=int,
                  help="Benchmark results are average over n_times experiments")

    op.add_option("--n-population",
                  dest="n_population", default=100000, type=int,
                  help="Size of the population to sample from.")

    op.add_option("--n-step",
                  dest="n_steps", default=5, type=int,
                  help="Number of step interval between 0 and n_population.")

    default_algorithms = "custom-tracking-selection,custom-auto," \
                         "custom-reservoir-sampling,custom-pool,"\
                         "python-core-sample,numpy-permutation"

    op.add_option("--algorithm",
                  dest="selected_algorithm",
                  default=default_algorithms,
                  type=str,
                  help="Comma-separated list of transformer to benchmark. "
                       "Default: %default. \nAvailable: %default")

    # op.add_option("--random-seed",
    #               dest="random_seed", default=13, type=int,
    #               help="Seed used by the random number generators.")

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    selected_algorithm = opts.selected_algorithm.split(',')
    for key in selected_algorithm:
        if key not in default_algorithms.split(','):
            raise ValueError("Unknown sampling algorithm \"%s\" not in (%s)."
                             % (key, default_algorithms))

    ###########################################################################
    # List sampling algorithm
    ###########################################################################
    # We assume that sampling algorithm has the following signature:
    #   sample(n_population, n_sample)
    #
    sampling_algorithm = {}

    ###########################################################################
    # Set Python core input
    sampling_algorithm["python-core-sample"] = \
        lambda n_population, n_sample: \
            random.sample(xrange(n_population), n_sample)

   ###########################################################################
    # Set custom automatic method selection
    sampling_algorithm["custom-auto"] = \
        lambda n_population, n_samples, random_state=None: \
            sample_without_replacement(n_population,
                                       n_samples,
                                       method="auto",
                                       random_state=random_state)

    ###########################################################################
    # Set custom tracking based method
    sampling_algorithm["custom-tracking-selection"] = \
        lambda n_population, n_samples, random_state=None: \
            sample_without_replacement(n_population,
                                       n_samples,
                                       method="tracking_selection",
                                       random_state=random_state)

    ###########################################################################
    # Set custom reservoir based method
    sampling_algorithm["custom-reservoir-sampling"] = \
        lambda n_population, n_samples, random_state=None: \
            sample_without_replacement(n_population,
                                       n_samples,
                                       method="reservoir_sampling",
                                       random_state=random_state)

    ###########################################################################
    # Set custom reservoir based method
    sampling_algorithm["custom-pool"] = \
        lambda n_population, n_samples, random_state=None: \
            sample_without_replacement(n_population,
                                       n_samples,
                                       method="pool",
                                       random_state=random_state)

    ###########################################################################
    # Numpy permutation based
    sampling_algorithm["numpy-permutation"] = \
        lambda n_population, n_sample: \
            np.random.permutation(n_population)[:n_sample]

    ###########################################################################
    # Remove unspecified algorithm
    sampling_algorithm = dict((key, value)
                              for key, value in sampling_algorithm.items()
                              if key in selected_algorithm)

    ###########################################################################
    # Perform benchmark
    ###########################################################################
    time = {}
    n_samples = np.linspace(start=0, stop=opts.n_population,
        num=opts.n_steps).astype(np.int)

    ratio = n_samples / opts.n_population

    print('Benchmarks')
    print("===========================")

    for name in sorted(sampling_algorithm):
        print("Perform benchmarks for %s..." % name, end="")
        time[name] = np.zeros(shape=(opts.n_steps, opts.n_times))

        for step in xrange(opts.n_steps):
            for it in xrange(opts.n_times):
                time[name][step, it] = bench_sample(sampling_algorithm[name],
                                                      opts.n_population,
                                                      n_samples[step])

        print("done")

    print("Averaging results...", end="")
    for name in sampling_algorithm:
        time[name] = np.mean(time[name], axis=1)
    print("done
```
209 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
@deprecated("The class BaseRandomizedLinearModel is deprecated in 0.19"
            " and will be removed in 0.21.")
class BaseRandomizedLinearModel(six.with_metaclass(ABCMeta, BaseEstimator,
                                                   SelectorMixin)):
    """Base class to implement randomized linear models for feature selection

    This implements the strategy by Meinshausen and Buhlman:
    stability selection with randomized sampling, and random re-weighting of
    the penalty.
    """

    @abstractmethod
    def __init__(self):
        pass

    _preprocess_data = staticmethod(_preprocess_data)
```
210 - /tmp/repos/scikit-learn/sklearn/random_projection.py
```python
def _check_input_size(n_components, n_features):
    """Factorize argument checking for random matrix generation"""
    if n_components <= 0:
        raise ValueError("n_components must be strictly positive, got %d" %
                         n_components)
    if n_features <= 0:
        raise ValueError("n_features must be strictly positive, got %d" %
                         n_components)
```
211 - /tmp/repos/scikit-learn/examples/ensemble/plot_bias_variance.py
```python
# ..."""
============================================================
Single estimator versus bagging: bias-variance decomposition
============================================================

This example illustrates and compares the bias-variance decomposition of the
expected mean squared error of a single estimator against a bagging ensemble.

In regression, the expected mean squared error of an estimator can be
decomposed in terms of bias, variance and noise. On average over datasets of
the regression problem, the bias term measures the average amount by which the
predictions of the estimator differ from the predictions of the best possible
estimator for the problem (i.e., the Bayes model). The variance term measures
the variability of the predictions of the estimator when fit over different
instances LS of the problem. Finally, the noise measures the irreducible part
of the error which is due the variability in the data.

The upper left figure illustrates the predictions (in dark red) of a single
decision tree trained over a random dataset LS (the blue dots) of a toy 1d
regression problem. It also illustrates the predictions (in light red) of other
single decision trees trained over other (and different) randomly drawn
instances LS of the problem. Intuitively, the variance term here corresponds to
the width of the beam of predictions (in light red) of the individual
estimators. The larger the variance, the more sensitive are the predictions for
`x` to small changes in the training set. The bias term corresponds to the
difference between the average prediction of the estimator (in cyan) and the
best possible model (in dark blue). On this problem, we can thus observe that
the bias is quite low (both the cyan and the blue curves are close to each
other) while the variance is large (the red beam is rather wide).

The lower left figure plots the pointwise decomposition of the expected mean
squared error of a single decision tree. It confirms that the bias term (in
blue) is low while the variance is large (in green). It also illustrates the
noise part of the error which, as expected, appears to be constant and around
`0.01`.

The right figures correspond to the same plots but using instead a bagging
ensemble of decision trees. In both figures, we can observe that the bias term
is larger than in the previous case. In the upper right figure, the difference
between the average prediction (in cyan) and the best possible model is larger
(e.g., notice the offset around `x=2`). In the lower right figure, the bias
curve is also slightly higher than in the lower left figure. In terms of
variance however, the beam of predictions is narrower, which suggests that the
variance is lower. Indeed, as the lower right figure confirms, the variance
term (in green) is lower than for single decision trees. Overall, the bias-
variance decomposition is therefore no longer the same. The tradeoff is better
for bagging: averaging several decision trees fit on bootstrap copies of the
dataset slightly increases the bias term but allows for a larger reduction of
the variance, which results in a lower overall mean squared error (compare the
red curves int the lower figures). The script output also confirms this
intuition. The total error of the bagging ensemble is lower than the total
error of a single decision tree, and this difference indeed mainly stems from a
reduced variance.

For further details on bias-variance decomposition, see section 7.3 of [1]_.

References
----------

.. [1] T. Hastie, R. Tibshirani and J. Friedman,
       "Elements of Statistical Learning", Springer, 2009.

"""
print(__doc__)

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Settings
n_repeat = 50       # Number of iterations for computing expectations
n_train = 50        # Size of the training set
n_test = 1000       # Size of the test set
noise = 0.1         # Standard deviation of the noise
np.random.seed(0)

# Change this for exploring the bias-variance decomposition of other
# estimators. This should work well for estimators with high variance (e.g.,
# decision trees or KNN), but poorly for estimators with low variance (e.g.,
# linear models).
estimators = [("Tree", DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

n_estimators = len(estimators)


# Generate data
def f(x):
    x = x.ravel()

    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y


X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

plt.figure(figsize=(10, 8))

# Loop over estimators to compare
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
        else:
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

    plt.plot(X_test, np.mean(y_predict, axis=1), "c",
             label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlim([-5, 5])
    plt.title(name)

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))

    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    plt.plot(X_test, y_noise, "c", label="$noise(x)$"
```
212 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def load_20news():
    print("Loading 20 newsgroups dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(dataset.data)
    return tfidf
```
213 - /tmp/repos/scikit-learn/benchmarks/bench_hist_gradient_boosting_higgsboson.py
```python
print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.lightgbm:
    print("Fitting a LightGBM model...")
    tic = time()
    lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')
    lightgbm_est.fit(data_train, target_train)
    toc = time()
    predicted_test = lightgbm_est.predict(data_test)
    predicted_proba_test = lightgbm_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.xgboost:
    print("Fitting an XGBoost model...")
    tic = time()
    xgboost_est = get_equivalent_estimator(est, lib='xgboost')
    xgboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = xgboost_est.predict(data_test)
    predicted_proba_test = xgboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

if args.catboost:
    print("Fitting a Catboost model...")
    tic = time()
    catboost_est = get_equivalent_estimator(est, lib='catboost')
    catboost_est.fit(data_train, target_train)
    toc = time()
    predicted_test = catboost_est.predict(data_test)
    predicted_proba_test = catboost_est.predict_proba(data_test)
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
    acc = accuracy_score(target_test, predicted_test)
    print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")

```
214 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
# ...


@deprecated("The class RandomizedLasso is deprecated in 0.19"
            " and will be removed in 0.21.")
class RandomizedLasso(BaseRandomizedLinearModel):
    """Randomized Lasso.

    Randomized Lasso works by subsampling the training data and
    computing a Lasso estimate where the penalty of a random subset of
    coefficients has been scaled. By performing this double
    randomization several times, the method assigns high scores to
    features that are repeatedly selected across randomizations. This
    is known as stability selection. In short, features selected more
    often are considered good features.

    Parameters
    ----------
    alpha : float, 'aic', or 'bic', optional
        The regularization parameter alpha parameter in the Lasso.
        Warning: this is not the alpha parameter in the stability selection
        article which is scaling.

    scaling : float, optional
        The s parameter used to randomly scale the penalty of different
        features.
        Should be between 0 and 1.

    sample_fraction : float, optional
        The fraction of samples to be used in each randomized design.
        Should be between 0 and 1. If 1, all samples are used.

    n_resampling : int, optional
        Number of randomized models.

    selection_threshold : float, optional
        The score above which features should be selected.

    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    normalize : boolean, optional, default True
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learned more robust and almost independent of
        the number of samples. The same property is not valid for
        standardized data. However, if you wish to standardize, please
        use `preprocessing.StandardScaler` before calling `fit` on an
        estimator with `normalize=False`.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up calculations.
        If set to 'auto' let us decide.
        The Gram matrix can also be passed as argument, but it will be used
        only for the selection of parameter alpha, if alpha is 'aic' or 'bic'.

    max_iter : integer, optional
        Maximum number of iterations to perform in the Lars algorithm.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the 'tol' parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : integer, optional
        Number of CPUs to use during the resampling. If '-1', use
        all the CPUs

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    memory : None, str or object with the joblib.Memory interface, optional 
```
215 - /tmp/repos/scikit-learn/sklearn/gaussian_process/gaussian_process.py
```python
@deprecated("GaussianProcess was deprecated in version 0.18 and will be "
            "removed in 0.20. Use the GaussianProcessRegressor instead.")
class GaussianProcess(BaseEstimator, RegressorMixin):


    def _check_params(self, n_samples=None):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.corr))

        # Check storage mode
        if self.storage_mode != 'full' and self.storage_mode != 'light':
            raise ValueError("Storage mode should either be 'full' or "
                             "'light', %s was given." % self.storage_mode)

        # Check correlation parameters
        self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            self.thetaL = np.atleast_2d(self.thetaL)
            self.thetaU = np.atleast_2d(self.thetaU)
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        self.nugget = np.asarray(self.nugget)
        if np.any(self.nugget) < 0.:
            raise ValueError("nugget must be positive or zero.")
        if (n_samples is not None
                and self.nugget.shape not in [(), (n_samples,)]):
            raise ValueError("nugget must be either a scalar "
                             "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)
```
216 - /tmp/repos/scikit-learn/benchmarks/bench_rcv1_logreg_convergence.py
```python
if lightning_clf is not None and not fit_intercept:
    alpha = 1. / C / n_samples
    # compute the same step_size than in LR-sag
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log",
                                   fit_intercept)

    clfs.append(
        ("Lightning-SVRG",
         lightning_clf.SVRGClassifier(alpha=alpha, eta=step_size,
                                      tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))
    clfs.append(
        ("Lightning-SAG",
         lightning_clf.SAGClassifier(alpha=alpha, eta=step_size,
                                     tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))

    # We keep only 200 features, to have a dense dataset,
    # and compare to lightning SAG, which seems incorrect in the sparse case.
    X_csc = X.tocsc()
    nnz_in_each_features = X_csc.indptr[1:] - X_csc.indptr[:-1]
    X = X_csc[:, np.argsort(nnz_in_each_features)[-200:]]
    X = X.toarray()
    print("dataset: %.3f MB" % (X.nbytes / 1e6))


# Split training and testing. Switch train and test subset compared to
# LYRL2004 split, to have a larger training dataset.
n = 23149
X_test = X[:n, :]
y_test = y[:n]
X = X[n:, :]
y = y[n:]

clfs = bench(clfs)

plot_train_scores(clfs)
plot_test_scores(clfs)
plot_train_losses(clfs)
plot_dloss(clfs)
plt.show()

```
217 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class BaseHistGradientBoosting(BaseEstimator, ABC):


    def _validate_parameters(self):
        """Validate parameters passed to __init__.

        The parameters that are directly passed to the grower are checked in
        TreeGrower."""

        if self.loss not in self._VALID_LOSSES:
            raise ValueError(
                "Loss {} is not supported for {}. Accepted losses: "
                "{}.".format(self.loss, self.__class__.__name__,
                             ', '.join(self._VALID_LOSSES)))

        if self.learning_rate <= 0:
            raise ValueError('learning_rate={} must '
                             'be strictly positive'.format(self.learning_rate))
        if self.max_iter < 1:
            raise ValueError('max_iter={} must not be smaller '
                             'than 1.'.format(self.max_iter))
        if self.n_iter_no_change is not None and self.n_iter_no_change < 0:
            raise ValueError('n_iter_no_change={} must be '
                             'positive.'.format(self.n_iter_no_change))
        if (self.validation_fraction is not None and
                self.validation_fraction <= 0):
            raise ValueError(
                'validation_fraction={} must be strictly '
                'positive, or None.'.format(self.validation_fraction))
        if self.tol is not None and self.tol < 0:
            raise ValueError('tol={} '
                             'must not be smaller than 0.'.format(self.tol))
```
218 - /tmp/repos/scikit-learn/sklearn/ensemble/gradient_boosting.py
```python
\
    e (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.ensemble.HistGradientBoostingClassifier,
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    _SUPPORTED_LOSS = ('deviance', 'exponential')

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, presort=presort,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol)

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError("y contains %d class after sample_weight "
                             "trimmed classes with zero weights, while a "
                             "minimum of 2 classes are required."
                             % n_trim_classes)
        self.n_classes_ = len(self.classes_)
        return y

    
```
**219 - /tmp/repos/scikit-learn/sklearn/ensemble/forest.py**:
```python
class RandomTreesEmbedding(BaseForest):


    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object

        """
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self
```
220 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
# ...


class RANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin,
                      MultiOutputMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    A detailed description of the algorithm can be found in the documentation
    of the ``linear_model`` sub-package.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the following methods:

         * `fit(X, y)`: Fit model to given training data and target values.
         * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.
         * `predict(X)`: Returns predicted values using the linear model,
           which is used to compute residual error using loss function.

        If `base_estimator` is None, then
        ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), optional
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0]`) for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `base_estimator`. By default a
        ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``.

    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`.

    is_data_valid : callable, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    max_skips : int, optional
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

        .. versionadded:: 0.19

    stop_n_inliers : int, optional
        Stop iteration if at least this number of inliers are found.

    stop_score : float, optional
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    loss : string, callable, optional, default "absolute_loss"
        String inputs, "absolute_loss" and "squared_loss" are supported which
        find the absolute loss and squared loss per sample
        respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

    random_state : int, RandomState instance or None, optional, default None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `base_estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    n_skips_no_inliers_ : int
        Number of iterations skipped due to finding zero inliers.

        .. versionadded:: 0.19

    n_skips_invalid_data_ : int
        Number of iterations skipped due to invalid data defined by
        ``is_data_valid``.

        .. versionadded:: 0.19

    n_skips_invalid_model_ : int
        Number of iterations skipped due to an invalid model defined by
        ``is_model_valid``.

        .. versionadded:: 0.19

    Examples
    --------
    >>> from sklearn.linear_model import RANSACRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, random_state=0)
    >>> reg = RANSACRegressor(random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.9885...
    >>> reg.predict(X[:1,])
    array([-31.9417...])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] https://www.sri.com/sites/default/files/publications/ransac-publication.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
    
```
221 - /tmp/repos/scikit-learn/sklearn/linear_model/randomized_l1.py
```python
@deprecated("The class BaseRandomizedLinearModel is deprecated in 0.19"
            " and will be removed in 0.21.")
class BaseRandomizedLinearModel(six.with_metaclass(ABCMeta, BaseEstimator,
                                                   SelectorMixin)):


    def _make_estimator_and_params(self, X, y):
        """Return the parameters passed to the estimator"""
        raise NotImplementedError
```
222 - /tmp/repos/scikit-learn/examples/ensemble/plot_bias_variance.py
```python
)

    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if n == n_estimators - 1:

        plt.legend(loc=(1.1, .5))

plt.subplots_adjust(right=.75)
plt.show()

```
223 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.

    n_samples : int
        Total number of samples in the data.

    min_samples : int
        Minimum number of samples chosen randomly from original data.

    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.

    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float('inf')
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))
```
224 - /tmp/repos/scikit-learn/examples/ensemble/plot_gradient_boosting_oob.py
```python
def cv_estimate(n_splits=None):
    cv = KFold(n_splits=n_splits)
    cv_clf = ensemble.GradientBoostingClassifier(**params)
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train, y_train):
        cv_clf.fit(X_train[train], y_train[train])
        val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
    val_scores /= n_splits
    return val_scores


# Estimate best n_estimator using cross-validation
cv_score = cv_estimate(3)

# Compute best n_estimator for test data
test_score = heldout_score(clf, X_test, y_test)

# negative cumulative sum of oob improvements
cumsum = -np.cumsum(clf.oob_improvement_)

# min loss according to OOB
oob_best_iter = x[np.argmin(cumsum)]

# min loss according to test (normalize such that first loss is 0)
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]

# min loss according to cv (normalize such that first loss is 0)
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]

# color brew for the three curves
oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

# plot curves and vertical lines for best iterations
plt.plot(x, cumsum, label='OOB loss', color=oob_color)
plt.plot(x, test_score, label='Test loss', color=test_color)
plt.plot(x, cv_score, label='CV loss', color=cv_color)
plt.axvline(x=oob_best_iter, color=oob_color)
plt.axvline(x=test_best_iter, color=test_color)
plt.axvline(x=cv_best_iter, color=cv_color)

# add three vertical lines to xticks
xticks = plt.xticks()
xticks_pos = np.array(xticks[0].tolist() +
                      [oob_best_iter, cv_best_iter, test_best_iter])
xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
                        ['OOB', 'CV', 'Test'])
ind = np.argsort(xticks_pos)
xticks_pos = xticks_pos[ind]
xticks_label = xticks_label[ind]
plt.xticks(xticks_pos, xticks_label)

plt.legend(loc='upper right')
plt.ylabel('normalized loss')
plt.xlabel('number of iterations')

plt.show()

```
225 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class BaseHistGradientBoosting(BaseEstimator, ABC):


    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # A higher score is always better. Higher tol means that it will be
        # harder for subsequent iteration to be considered an improvement upon
        # the reference score, and therefore it is more likely to early stop
        # because of the lack of significant improvement.
        tol = 0 if self.tol is None else self.tol
        reference_score = scores[-reference_position] + tol
        recent_scores = scores[-reference_position + 1:]
        recent_improvements = [score > reference_score
                               for score in recent_scores]
        return not any(recent_improvements)
```
226 - /tmp/repos/scikit-learn/examples/ensemble/plot_adaboost_hastie_10_2.py
```python
# ..."""
=============================
Discrete versus Real AdaBoost
=============================

This example is based on Figure 10.2 from Hastie et al 2009 [1]_ and
illustrates the difference in performance between the discrete SAMME [2]_
boosting algorithm and real SAMME.R boosting algorithm. Both algorithms are
evaluated on a binary classification task where the target Y is a non-linear
function of 10 input features.

Discrete SAMME AdaBoost adapts based on errors in predicted class labels
whereas real SAMME.R uses the predicted class probabilities.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
    Learning Ed. 2", Springer, 2009.

.. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

"""
print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>,
#         Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier


n_estimators = 400
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)


```
227 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
"""Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    
```
228 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
"""Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, (default=0)
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    
```
229 - /tmp/repos/scikit-learn/sklearn/feature_selection/rfe.py
```python
"""Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    
```
230 - /tmp/repos/scikit-learn/examples/impute/plot_iterative_imputer_variants_comparison.py
```python
# ..."""
=========================================================
Imputing missing values with variants of IterativeImputer
=========================================================

The :class:`sklearn.impute.IterativeImputer` class is very flexible - it can be
used with a variety of estimators to do round-robin regression, treating every
variable as an output in turn.

In this example we compare some estimators for the purpose of missing feature
imputation with :class:`sklearn.impute.IterativeImputer`:

* :class:`~sklearn.linear_model.BayesianRidge`: regularized linear regression
* :class:`~sklearn.tree.DecisionTreeRegressor`: non-linear regression
* :class:`~sklearn.ensemble.ExtraTreesRegressor`: similar to missForest in R
* :class:`~sklearn.neighbors.KNeighborsRegressor`: comparable to other KNN
  imputation approaches

Of particular interest is the ability of
:class:`sklearn.impute.IterativeImputer` to mimic the behavior of missForest, a
popular imputation package for R. In this example, we have chosen to use
:class:`sklearn.ensemble.ExtraTreesRegressor` instead of
:class:`sklearn.ensemble.RandomForestRegressor` (as in missForest) due to its
increased speed.

Note that :class:`sklearn.neighbors.KNeighborsRegressor` is different from KNN
imputation, which learns from samples with missing values by using a distance
metric that accounts for missing values, rather than imputing them.

The goal is to compare different estimators to see which one is best for the
:class:`sklearn.impute.IterativeImputer` when using a
:class:`sklearn.linear_model.BayesianRidge` estimator on the California housing
dataset with a single value randomly removed from each row.

For this particular pattern of missing values we see that
:class:`sklearn.ensemble.ExtraTreesRegressor` and
:class:`sklearn.linear_model.BayesianRidge` give the best results.

```
231 - /tmp/repos/scikit-learn/sklearn/preprocessing/_discretization.py
```python
if 'onehot' in self.encode:
            self._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.n_bins_],
                sparse=self.encode == 'onehot')
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self._encoder.fit(np.zeros((1, len(self.n_bins_)), dtype=int))

        return self

    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature.
        """
        orig_bins = self.n_bins
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. "
                                 "Received {}, expected int."
                                 .format(KBinsDiscretizer.__name__,
                                         type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number "
                                 "of bins. Received {}, expected at least 2."
                                 .format(KBinsDiscretizer.__name__, orig_bins))
            return np.full(n_features, orig_bins, dtype=np.int)

        n_bins = check_array(orig_bins, dtype=np.int, copy=True,
                             ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array "
                             "of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid number "
                             "of bins at indices {}. Number of bins "
                             "must be at least 2, and must be an int."
                             .format(KBinsDiscretizer.__name__, indices))
        return n_bins

    
```
232 - /tmp/repos/scikit-learn/sklearn/ensemble/voting_classifier.py
```python
# ...


class VotingClassifier(_BaseComposition, ClassifierMixin, TransformerMixin):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <voting_classifier>`.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to `None` using
        ``set_params``.

    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.

    flatten_transform : bool, optional (default=None)
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not `None`.

    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    classes_ : array-like, shape = [n_predictions]
        The classes labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    >>>
    
```
233 - /tmp/repos/scikit-learn/examples/model_selection/plot_randomized_search.py
```python
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
```
234 - /tmp/repos/scikit-learn/sklearn/random_projection.py
```python
if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for i in xrange(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(n_features, n_nonzero_i,
                                                   random_state=rng)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix((data, indices, indptr),
                                   shape=(n_components, n_features))

        return np.sqrt(1 / density) / np.sqrt(n_components) * components
# ...
```
235 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        check_is_fitted(self, "estimators_features_")
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
```
236 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
# ...


class RANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    A detailed description of the algorithm can be found in the documentation
    of the ``linear_model`` sub-package.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the following methods:

         * `fit(X, y)`: Fit model to given training data and target values.
         * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.

        If `base_estimator` is None, then
        ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), optional
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0]`) for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `base_estimator`. By default a
        ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``.

    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`.

    is_data_valid : callable, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    max_skips : int, optional
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

        .. versionadded:: 0.19

    stop_n_inliers : int, optional
        Stop iteration if at least this number of inliers are found.

    stop_score : float, optional
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    residual_metric : callable, optional
        Metric to reduce the dimensionality of the residuals to 1 for
        multi-dimensional target values ``y.shape[1] > 1``. By default the sum
        of absolute differences is used::

            lambda dy: np.sum(np.abs(dy), axis=1)

        .. deprecated:: 0.18
           ``residual_metric`` is deprecated from 0.18 and will be removed in
           0.20. Use ``loss`` instead.

    loss : string, callable, optional, default "absolute_loss"
        String inputs, "absolute_loss" and "squared_loss" are supported which
        find the absolute loss and squared loss per sample
        respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

    random_state : int, RandomState instance or None, optional, default None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `base_estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    n_skips_no_inliers_ : int
        Number of iterations skipped due to finding zero inliers.

        .. versionadded:: 0.19

    n_skips_invalid_data_ : int
        Number of iterations skipped due to invalid data defined by
        ``is_data_valid``.

        .. versionadded:: 0.19

    n_skips_invalid_model_ : int
        Number of iterations skipped due to an invalid model defined by
        ``is_model_valid``.

        .. versionadded:: 0.19

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] https://www.sri.com/sites/default/files/publications/ransac-publication.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
    
```
237 - /tmp/repos/scikit-learn/sklearn/linear_model/ransac.py
```python
# ...


class RANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    A detailed description of the algorithm can be found in the documentation
    of the ``linear_model`` sub-package.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the following methods:

         * `fit(X, y)`: Fit model to given training data and target values.
         * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.

        If `base_estimator` is None, then
        ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), optional
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0]`) for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `base_estimator`. By default a
        ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``.

    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`.

    is_data_valid : callable, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    max_skips : int, optional
        Maximum number of iterations that can be skipped due to finding zero
        inliers or invalid data defined by ``is_data_valid`` or invalid models
        defined by ``is_model_valid``.

        .. versionadded:: 0.19

    stop_n_inliers : int, optional
        Stop iteration if at least this number of inliers are found.

    stop_score : float, optional
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    residual_metric : callable, optional
        Metric to reduce the dimensionality of the residuals to 1 for
        multi-dimensional target values ``y.shape[1] > 1``. By default the sum
        of absolute differences is used::

            lambda dy: np.sum(np.abs(dy), axis=1)

        .. deprecated:: 0.18
           ``residual_metric`` is deprecated from 0.18 and will be removed in
           0.20. Use ``loss`` instead.

    loss : string, callable, optional, default "absolute_loss"
        String inputs, "absolute_loss" and "squared_loss" are supported which
        find the absolute loss and squared loss per sample
        respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the i-th value of the array corresponding to the loss
        on ``X[i]``.

        If the loss on a sample is greater than the ``residual_threshold``,
        then this sample is classified as an outlier.

    random_state : int, RandomState instance or None, optional, default None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `base_estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    n_skips_no_inliers_ : int
        Number of iterations skipped due to finding zero inliers.

        .. versionadded:: 0.19

    n_skips_invalid_data_ : int
        Number of iterations skipped due to invalid data defined by
        ``is_data_valid``.

        .. versionadded:: 0.19

    n_skips_invalid_model_ : int
        Number of iterations skipped due to an invalid model defined by
        ``is_model_valid``.

        .. versionadded:: 0.19

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] https://www.sri.com/sites/default/files/publications/ransac-publication.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
    
```
238 - /tmp/repos/scikit-learn/sklearn/utils/random.py
```python
# Author: Hamzeh Alsalhi <ha258@cornell.edu>
#
# License: BSD 3 clause
from __future__ import division
import numpy as np
import scipy.sparse as sp
import array

from sklearn.utils import check_random_state
from ._random import sample_without_replacement
from .deprecation import deprecated

__all__ = ['sample_without_replacement', 'choice']
```
239 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaggingRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        check_is_fitted(self, "estimators_features_")
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
```
240 - /tmp/repos/scikit-learn/sklearn/random_projection.py
```python
if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(n_features, n_nonzero_i,
                                                   random_state=rng)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix((data, indices, indptr),
                                   shape=(n_components, n_features))

        return np.sqrt(1 / density) / np.sqrt(n_components) * components
# ...
```
241 - /tmp/repos/scikit-learn/benchmarks/bench_plot_nmf.py
```python
def run_bench(X, clfs, plot_name, n_components, tol, alpha, l1_ratio):
    start = time()
    results = []
    for name, clf_type, iter_range, clf_params in clfs:
        print("Training %s:" % name)
        for rs, init in enumerate(('nndsvd', 'nndsvdar', 'random')):
            print("    %s %s: " % (init, " " * (8 - len(init))), end="")
            W, H = _initialize_nmf(X, n_components, init, 1e-6, rs)

            for max_iter in iter_range:
                clf_params['alpha'] = alpha
                clf_params['l1_ratio'] = l1_ratio
                clf_params['max_iter'] = max_iter
                clf_params['tol'] = tol
                clf_params['random_state'] = rs
                clf_params['init'] = 'custom'
                clf_params['n_components'] = n_components

                this_loss, duration = bench_one(name, X, W, H, X.shape,
                                                clf_type, clf_params,
                                                init, n_components, rs)

                init_name = "init='%s'" % init
                results.append((name, this_loss, duration, init_name))
                # print("loss: %.6f, time: %.3f sec" % (this_loss, duration))
                print(".", end="")
                sys.stdout.flush()
            print(" ")

    # Use a panda dataframe to organize the results
    results_df = pandas.DataFrame(results,
                                  columns="method loss time init".split())
    print("Total time = %0.3f sec\n" % (time() - start))

    # plot the results
    plot_results(results_df, plot_name)
    return results_df
```
242 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaggingRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        check_is_fitted(self, "estimators_features_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
```
243 - /tmp/repos/scikit-learn/sklearn/ensemble/bagging.py
```python
"""

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(BaggingRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        check_is_fitted(self, "estimators_features_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
```
244 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
"""Fast Gradient Boosting decision trees for classification and regression."""
# Author: Nicolas Hug

from abc import ABC, abstractmethod

import numpy as np
from timeit import default_timer as time
from ...base import BaseEstimator, RegressorMixin, ClassifierMixin
from ...utils import check_X_y, check_random_state, check_array
from ...utils.validation import check_is_fitted
from ...utils.multiclass import check_classification_targets
from ...metrics import check_scoring
from ...model_selection import train_test_split
from ...preprocessing import LabelEncoder
from ._gradient_boosting import _update_raw_predictions
from .types import Y_DTYPE, X_DTYPE, X_BINNED_DTYPE

from .binning import _BinMapper
from .grower import TreeGrower
from .loss import _LOSSES
```
245 - /tmp/repos/scikit-learn/examples/linear_model/plot_ard.py
```python
# ..."""
==================================================
Automatic Relevance Determination Regression (ARD)
==================================================

Fit regression model with Bayesian Ridge Regression.

See :ref:`bayesian_ridge_regression` for more information on the regressor.

Compared to the OLS (ordinary least squares) estimator, the coefficient
weights are slightly shifted toward zeros, which stabilises them.

The histogram of the estimated weights is very peaked, as a sparsity-inducing
prior is implied on the weights.

The estimation of the model is done by iteratively maximizing the
marginal log-likelihood of the observations.

We also plot predictions and uncertainties for ARD
for one dimensional regression using polynomial feature expansion.
Note the uncertainty starts going up on the right side of the plot.
This is because these test samples are outside of the range of the training
samples.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import ARDRegression, LinearRegression

# #############################################################################
# Generating simulated data with Gaussian weights

# Parameters of the example
np.random.seed(0)
n_samples, n_features = 100, 100
# Create Gaussian data
X = np.random.randn(n_samples, n_features)
# Create weights with a precision lambda_ of 4.
lambda_ = 4.
w = np.zeros(n_features)
# Only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# Create noise with a precision alpha of 50.
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# Create the target
y = np.dot(X, w) + noise

# #############################################################################
# Fit the ARD Regression
clf = ARDRegression(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

# #############################################################################
# Plot the true weights, the estimated weights, the histogram of the
# weights, and predictions with standard deviations
plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2,
         label="ARD estimate")
plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2,
         label="OLS estimate")
plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc=1)

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color='navy', log=True)
plt.scatter(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)),
            color='gold', marker='o', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc=1)

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color='navy', linewidth=2)
plt.ylabel("Score")
plt.xlabel("Iterations")


# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=1)
clf_poly = ARDRegression(threshold_lambda=1e5)
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
plt.figure(figsize=(6, 5))
plt.errorbar(X_plot, y_mean, y_std, color='navy',
             label="Polynomial ARD", linewidth=2)
plt.plot(X_plot, y_plot, color='gold', linewidth=2,
         label="Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()

```
246 - /tmp/repos/scikit-learn/benchmarks/bench_tree.py
```python
def bench_scikit_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_classifier_results.append(
        delta.seconds + delta.microseconds / mu_second)
```
247 - /tmp/repos/scikit-learn/examples/plot_johnson_lindenstrauss_bound.py
```python
\nn_samples vs n_components")

# range of admissible distortions
eps_range = np.linspace(0.01, 0.99, 100)

# range of number of samples (observation) to embed
n_samples_range = np.logspace(2, 6, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

plt.figure()
for n_samples, color in zip(n_samples_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
    plt.semilogy(eps_range, min_n_components, color=color)

plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
plt.xlabel("Distortion eps")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")

# Part 2: perform sparse random projection of some digits images which are
# quite low dimensional and dense or documents of the 20 newsgroups dataset
# which is both high dimensional and sparse

if '--twenty-newsgroups' in sys.argv:
    # Need an internet connection hence not enabled by default
    data = fetch_20newsgroups_vectorized().data[:500]
else:
    data = load_digits().data[:500]

n_samples, n_features = data.shape
print("Embedding %d samples with dim %d using various random projections"
      % (n_samples, n_features))

n_components_range = np.array([300, 1000, 10000])
dists = euclidean_distances(data, squared=True).ravel()

# select only non-identical samples pairs
nonzero = dists != 0
dists = dists[nonzero]

for n_components in n_components_range:
    t0 = time()
    rp = SparseRandomProjection(n_components=n_components)
    projected_data = rp.fit_transform(data)
    print("Projected %d samples from %d to %d in %0.3fs"
          % (n_samples, n_features, n_components, time() - t0))
    if hasattr(rp, 'components_'):
        n_bytes = rp.components_.data.nbytes
        n_bytes += rp.components_.indices.nbytes
        print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))

    projected_dists = euclidean_distances(
        projected_data, squared=True).ravel()[nonzero]

    plt.figure()
    plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components=%d" %
              n_components)
    cb = plt.colorbar()
    cb.set_label('Sample pairs counts')

    rates = projected_dists / dists
    print("Mean distances rate: %0.2f (%0.2f)"
          % (np.mean(rates), np.std(rates)))

    plt.figure()
    plt.hist(rates, bins=50, range=(0., 2.), edgecolor='k', **density_param)
    plt.xlabel("Squared distances rate: projected / original")
    plt.ylabel("Distribution of samples pairs")
    plt.title("Histogram of pairwise distance rates for n_components=%d" %
              n_components)

    # TODO: compute the expected value of eps and add them to the previous plot
    # as vertical lines / region

plt.show()

```
248 - /tmp/repos/scikit-learn/sklearn/preprocessing/_discretization.py
```python
class KBinsDiscretizer(BaseEstimator, TransformerMixin):


    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature.
        """
        orig_bins = self.n_bins
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. "
                                 "Received {}, expected int."
                                 .format(KBinsDiscretizer.__name__,
                                         type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number "
                                 "of bins. Received {}, expected at least 2."
                                 .format(KBinsDiscretizer.__name__, orig_bins))
            return np.full(n_features, orig_bins, dtype=np.int)

        n_bins = check_array(orig_bins, dtype=np.int, copy=True,
                             ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array "
                             "of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid number "
                             "of bins at indices {}. Number of bins "
                             "must be at least 2, and must be an int."
                             .format(KBinsDiscretizer.__name__, indices))
        return n_bins
```
249 - /tmp/repos/scikit-learn/sklearn/utils/__init__.py
```python
if replace:
        indices = random_state.randint(0, n_samples, size=(max_n_samples,))
    else:
        indices = np.arange(n_samples)
        random_state.shuffle(indices)
        indices = indices[:max_n_samples]

    # convert sparse matrices to CSR for row-based indexing
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        # syntactic sugar for the unit argument case
        return resampled_arrays[0]
    else:
        return resampled_arrays
# ...
```
