0 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py
```python
"""Encode categorical features as a numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot (aka one-of-K or dummy)
    encoding scheme (``encoding='onehot'``, the default) or converted
    to ordinal integers (``encoding='ordinal'``).

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):

        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.

    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[ 1.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    
```
**1 - /tmp/repos/scikit-learn/sklearn/preprocessing/_encoders.py**:
```python
def inverse_transform(self, X):
        """Convert the back data to the original representation.

        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.

        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        if self.drop is None:
            n_transformed_features = sum(len(cats)
                                         for cats in self.categories_)
        else:
            n_transformed_features = sum(len(cats) - 1
                                         for cats in self.categories_)

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if X.shape[1] != n_transformed_features:
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        j = 0
        found_unknown = {}

        for i in range(n_features):
            if self.drop is None:
                cats = self.categories_[i]
            else:
                cats = np.delete(self.categories_[i], self.drop_idx_[i])
            n_categories = len(cats)

            # Only happens if there was a column with a unique
            # category. In this case we just fill the column with this
            # unique category value.
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self.drop_idx_[i]]
                j += n_categories
                continue
            sub = X[:, j:j + n_categories]
            # for sparse X argmax returns 2D matrix, ensure 1D array
            labels = np.asarray(_argmax(sub, axis=1)).flatten()
            X_tr[:, i] = cats[labels]
            if self.handle_unknown == 'ignore':
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                # ignored unknown categories: we have a row of all zero
                if unknown.any():
                    found_unknown[i] = unknown
            # drop will either be None or handle_unknown will be error. If
            # self.drop is not None, then we can safely assume that all of
            # the nulls in each column are the dropped value
            elif self.drop is not None:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    X_tr[dropped, i] = self.categories_[i][self.drop_idx_[i]]

            j += n_categories

        # if ignored are found: potentially need to upcast result to
        # insert None values
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)

            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None

        return X_tr

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : array of string, length n_output_features

        """
        check_is_fitted(self, 'categories_')
        cats = self.categories_
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))

        feature_names = []
        for i in range(len(cats)):
            names = [
                input_features[i] + '_' + str(t) for t in cats[i]]
            if self.drop is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)



```
