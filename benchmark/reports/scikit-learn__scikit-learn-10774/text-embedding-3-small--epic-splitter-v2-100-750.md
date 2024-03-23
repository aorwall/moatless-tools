0 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_iris(return_X_y=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, 'DESCR', the full description of
        the dataset, 'filename', the physical location of
        iris csv dataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')
    iris_csv_filename = join(module_path, 'data', 'iris.csv')

    with open(join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'],
                 filename=iris_csv_filename)
```
**1 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
```python
"""Load the RCV1 multilabel dataset, downloading it if necessary.

    Version: RCV1-v2, vectors, full sets, topics multilabels.

    ==============     =====================
    Classes                              103
    Samples total                     804414
    Dimensionality                     47236
    Features           real, between 0 and 1
    ==============     =====================

    Read more in the :ref:`User Guide <datasets>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : string, 'train', 'test', or 'all', default='all'
        Select the dataset to load: 'train' for the training set
        (23149 samples), 'test' for the test set (781265 samples),
        'all' for both, with the training samples first if shuffle is False.
        This follows the official LYRL2004 chronological split.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : boolean, default=False. If True, returns ``(dataset.data,
    dataset.target)`` instead of a Bunch object. See below for more
    information about the `dataset.data` and `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : scipy csr array, dtype np.float64, shape (804414, 47236)
        The array has 0.16% of non zero values.

    dataset.target : scipy csr array, dtype np.uint8, shape (804414, 103)
        Each sample has a value of 1 in its categories, and 0 in others.
        The array has 3.15% of non zero values.

    dataset.sample_id : numpy array, dtype np.uint32, shape (804414,)
        Identification number of each sample, as ordered in dataset.data.

    dataset.target_names : numpy array, dtype object, length (103)
        Names of each target (RCV1 topics), as ordered in dataset.target.

    dataset.DESCR : string
        Description of the RCV1 dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    References
    ----------
    Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new
    benchmark collection for text categorization research. The Journal of
    Machine Learning Research, 5, 361-397.

    """
    N_SAMPLES = 804414
    N_FEATURES = 47236
    N_CATEGORIES = 103
    N_TRAIN = 23149

    data_home = get_data_home(data_home=data_home)
    rcv1_dir = join(data_home, "RCV1")
    if download_if_missing:
        if not exists(rcv1_dir):
            makedirs(rcv1_dir)

    samples_path = _pkl_filepath(rcv1_dir, "samples.pkl")
    sample_id_path = _pkl_filepath(rcv1_dir, "sample_id.pkl")
    sample_topics_path = _pkl_filepath(rcv1_dir, "sample_topics.pkl")
    topics_path = _pkl_filepath(rcv1_dir, "topics_names.pkl")

    # load data (X) and sample_id
    if download_if_missing and (not exists(samples_path) or
                                not exists(sample_id_path)):
        files = []
        for each in XY_METADATA:
            logger.info("Downloading %s" % each.url)
            file_path = _fetch_remote(each, dirname=rcv1_dir)
            files.append(GzipFile(filename=file_path))

        Xy = load_svmlight_files(files, n_features=N_FEATURES)

        # Training data is before testing data
        X = sp.vstack([Xy[8], Xy[0], Xy[2], Xy[4], Xy[6]]).tocsr()
        sample_id = np.hstack((Xy[9], Xy[1], Xy[3], Xy[5], Xy[7]))
        sample_id = sample_id.astype(np.uint32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(sample_id, sample_id_path, compress=9)

        # delete archives
        for f in files:
            f.close()
            remove(f.name)
    else:
        X = joblib.load(samples_path)
        sample_id = joblib.load(sample_id_path)

    # load target (y), categories, and sample_id_bis
    if download_if_missing and (not exists(sample_topics_path) or
                                not exists(topics_path)):
        logger.info("Downloading %s" % TOPICS_METADATA.url)
        topics_archive_path = _fetch_remote(TOPICS_METADATA,
                                            dirname=rcv1_dir)

        # parse the target file
        n_cat = -1
        n_doc = -1
        doc_previous = -1
        y = np.zeros((N_SAMPLES, N_CATEGORIES), dtype=np.uint8)
        sample_id_bis = np.zeros(N_SAMPLES, dtype=np.int32)
        category_names = {}
        with GzipFile(filename=topics_archive_path, mode='rb') as f:
            for line in f:
                line_components = line.decode("ascii").split(u" ")
                if len(line_components) == 3:
                    cat, doc, _ = line_components
                    if cat not in category_names:
                        n_cat += 1
                        category_names[cat] = n_cat

                    doc = int(doc)
                    if doc != doc_previous:
                        doc_previous = doc
                        n_doc += 1
                        sample_id_bis[n_doc] = doc
                    y[n_doc, category_names[cat]] = 1

        # delete archive
        remove(topics_archive_path)

        # Samples in X are ordered with sample_id,
        # whereas in y, they are ordered with sample_id_bis.
        permutation = _find_permutation(sample_id_bis, sample_id)
        y = y[permutation, :]

        # save category names in a list, with same order than y
        categories = np.empty(N_CATEGORIES, dtype=object)
        for k in category_names.keys():
            categories[category_names[k]] = k

        # reorder categories in lexicographic order
        order = np.argsort(categories)
        categories = categories[order]
        y = sp.csr_matrix(y[:, order])

        joblib.dump(y, sample_topics_path, compress=9)
        joblib.dump(categories, topics_path, compress=9)
    else:
        y = joblib.load(sample_topics_path)
        categories = joblib.load(topics_path)

    if subset == 'all':
        pass
    elif subset == 'train':
        X = X[:N_TRAIN, :]
        y = y[:N_TRAIN, :]
        sample_id = sample_id[:N_TRAIN]
    elif subset == 'test':
        X = X[N_TRAIN:, :]
        y = y[N_TRAIN:, :]
        sample_id = sample_id[N_TRAIN:]
    else:
        raise ValueError("Unknown subset parameter. Got '%s' instead of one"
                         " of ('all', 'train', test')" % subset)

    if shuffle:
        X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, sample_id=sample_id,
                 target_names=categories, DESCR=__doc__)


def _inverse_permutation(p):
    
```
2 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_linnerud(return_X_y=False):
    """Load and return the linnerud dataset (multivariate regression).

    ==============    ============================
    Samples total     20
    Dimensionality    3 (for both data and target)
    Features          integer
    Targets           integer
    ==============    ============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: 'data' and
        'targets', the two multivariate datasets, with 'data' corresponding to
        the exercise and 'targets' corresponding to the physiological
        measurements, as well as 'feature_names' and 'target_names'.
        In addition, you will also have access to 'data_filename',
        the physical location of linnerud data csv dataset, and
        'target_filename', the physical location of
        linnerud targets csv datataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18
    """
    base_dir = join(dirname(__file__), 'data/')
    data_filename = join(base_dir, 'linnerud_exercise.csv')
    target_filename = join(base_dir, 'linnerud_physiological.csv')

    # Read data
    data_exercise = np.loadtxt(data_filename, skiprows=1)
    data_physiological = np.loadtxt(target_filename, skiprows=1)

    # Read header
    with open(data_filename) as f:
        header_exercise = f.readline().split()
    with open(target_filename) as f:
        header_physiological = f.readline().split()

    with open(dirname(__file__) + '/descr/linnerud.rst') as f:
        descr = f.read()

    if return_X_y:
        return data_exercise, data_physiological

    return Bunch(data=data_exercise, feature_names=header_exercise,
                 target=data_physiological,
                 target_names=header_physiological,
                 DESCR=descr,
                 data_filename=data_filename,
                 target_filename=target_filename)
```
3 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
# XXX: col_slice_y should be all nominal or all numeric
    _verify_target_data_type(features_dict, target_column)

    col_slice_y = [int(features_dict[col_name]['index'])
                   for col_name in target_column]

    col_slice_x = [int(features_dict[col_name]['index'])
                   for col_name in data_columns]
    for col_idx in col_slice_y:
        feat = features_list[col_idx]
        nr_missing = int(feat['number_of_missing_values'])
        if nr_missing > 0:
            raise ValueError('Target column {} has {} missing values. '
                             'Missing values are not supported for target '
                             'columns. '.format(feat['name'], nr_missing))

    # determine arff encoding to return
    return_sparse = False
    if data_description['format'].lower() == 'sparse_arff':
        return_sparse = True

    if not return_sparse:
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = _get_data_shape(data_qualities)
        # if the data qualities were not available, we can still get the
        # n_features from the feature list, with the n_samples unknown
        if shape is None:
            shape = (-1, len(features_list))
    else:
        shape = None

    # obtain the data
    arff = _download_data_arff(data_description['file_id'], return_sparse,
                               data_home)

    # nominal attributes is a dict mapping from the attribute name to the
    # possible values. Includes also the target column (which will be popped
    # off below, before it will be packed in the Bunch object)
    nominal_attributes = {k: v for k, v in arff['attributes']
                          if isinstance(v, list) and
                          k in data_columns + target_column}

    X, y = _convert_arff_data(arff['data'], col_slice_x, col_slice_y, shape)

    is_classification = {col_name in nominal_attributes
                         for col_name in target_column}
    if not is_classification:
        # No target
        pass
    elif all(is_classification):
        y = np.hstack([np.take(np.asarray(nominal_attributes.pop(col_name),
                                          dtype='O'),
                               y[:, i:i+1].astype(int, copy=False))
                       for i, col_name in enumerate(target_column)])
    elif any(is_classification):
        raise ValueError('Mix of nominal and non-nominal targets is not '
                         'currently supported')

    description = "{}\n\nDownloaded from openml.org.".format(
        data_description.pop('description'))

    # reshape y back to 1-D array, if there is only 1 target column; back
    # to None if there are not target columns
    if y.shape[1] == 1:
        y = y.reshape((-1,))
    elif y.shape[1] == 0:
        y = None

    if return_X_y:
        return X, y

    bunch = Bunch(
        data=X, target=y, feature_names=data_columns,
        DESCR=description, details=data_description,
        categories=nominal_attributes,
        url="https://www.openml.org/d/{}".format(data_id))

    return bunch
```
**4 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
```python
"""Load the RCV1 multilabel dataset (classification).

    Download it if necessary.

    Version: RCV1-v2, vectors, full sets, topics multilabels.

    =================   =====================
    Classes                               103
    Samples total                      804414
    Dimensionality                      47236
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <rcv1_dataset>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : string, 'train', 'test', or 'all', default='all'
        Select the dataset to load: 'train' for the training set
        (23149 samples), 'test' for the test set (781265 samples),
        'all' for both, with the training samples first if shuffle is False.
        This follows the official LYRL2004 chronological split.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : boolean, default=False.
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : scipy csr array, dtype np.float64, shape (804414, 47236)
        The array has 0.16% of non zero values.

    dataset.target : scipy csr array, dtype np.uint8, shape (804414, 103)
        Each sample has a value of 1 in its categories, and 0 in others.
        The array has 3.15% of non zero values.

    dataset.sample_id : numpy array, dtype np.uint32, shape (804414,)
        Identification number of each sample, as ordered in dataset.data.

    dataset.target_names : numpy array, dtype object, length (103)
        Names of each target (RCV1 topics), as ordered in dataset.target.

    dataset.DESCR : string
        Description of the RCV1 dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """
    N_SAMPLES = 804414
    N_FEATURES = 47236
    N_CATEGORIES = 103
    N_TRAIN = 23149

    data_home = get_data_home(data_home=data_home)
    rcv1_dir = join(data_home, "RCV1")
    if download_if_missing:
        if not exists(rcv1_dir):
            makedirs(rcv1_dir)

    samples_path = _pkl_filepath(rcv1_dir, "samples.pkl")
    sample_id_path = _pkl_filepath(rcv1_dir, "sample_id.pkl")
    sample_topics_path = _pkl_filepath(rcv1_dir, "sample_topics.pkl")
    topics_path = _pkl_filepath(rcv1_dir, "topics_names.pkl")

    # load data (X) and sample_id
    if download_if_missing and (not exists(samples_path) or
                                not exists(sample_id_path)):
        files = []
        for each in XY_METADATA:
            logger.info("Downloading %s" % each.url)
            file_path = _fetch_remote(each, dirname=rcv1_dir)
            files.append(GzipFile(filename=file_path))

        Xy = load_svmlight_files(files, n_features=N_FEATURES)

        # Training data is before testing data
        X = sp.vstack([Xy[8], Xy[0], Xy[2], Xy[4], Xy[6]]).tocsr()
        sample_id = np.hstack((Xy[9], Xy[1], Xy[3], Xy[5], Xy[7]))
        sample_id = sample_id.astype(np.uint32, copy=False)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(sample_id, sample_id_path, compress=9)

        # delete archives
        for f in files:
            f.close()
            remove(f.name)
    else:
        X = joblib.load(samples_path)
        sample_id = joblib.load(sample_id_path)

    # load target (y), categories, and sample_id_bis
    if download_if_missing and (not exists(sample_topics_path) or
                                not exists(topics_path)):
        logger.info("Downloading %s" % TOPICS_METADATA.url)
        topics_archive_path = _fetch_remote(TOPICS_METADATA,
                                            dirname=rcv1_dir)

        # parse the target file
        n_cat = -1
        n_doc = -1
        doc_previous = -1
        y = np.zeros((N_SAMPLES, N_CATEGORIES), dtype=np.uint8)
        sample_id_bis = np.zeros(N_SAMPLES, dtype=np.int32)
        category_names = {}
        with GzipFile(filename=topics_archive_path, mode='rb') as f:
            for line in f:
                line_components = line.decode("ascii").split(" ")
                if len(line_components) == 3:
                    cat, doc, _ = line_components
                    if cat not in category_names:
                        n_cat += 1
                        category_names[cat] = n_cat

                    doc = int(doc)
                    if doc != doc_previous:
                        doc_previous = doc
                        n_doc += 1
                        sample_id_bis[n_doc] = doc
                    y[n_doc, category_names[cat]] = 1

        # delete archive
        remove(topics_archive_path)

        # Samples in X are ordered with sample_id,
        # whereas in y, they are ordered with sample_id_bis.
        permutation = _find_permutation(sample_id_bis, sample_id)
        y = y[permutation, :]

        # save category names in a list, with same order than y
        categories = np.empty(N_CATEGORIES, dtype=object)
        for k in category_names.keys():
            categories[category_names[k]] = k

        # reorder categories in lexicographic order
        order = np.argsort(categories)
        categories = categories[order]
        y = sp.csr_matrix(y[:, order])

        joblib.dump(y, sample_topics_path, compress=9)
        joblib.dump(categories, topics_path, compress=9)
    else:
        y = joblib.load(sample_topics_path)
        categories = joblib.load(topics_path)

    if subset == 'all':
        pass
    elif subset == 'train':
        X = X[:N_TRAIN, :]
        y = y[:N_TRAIN, :]
        sample_id = sample_id[:N_TRAIN]
    elif subset == 'test':
        X = X[N_TRAIN:, :]
        y = y[N_TRAIN:, :]
        sample_id = sample_id[N_TRAIN:]
    else:
        raise ValueError("Unknown subset parameter. Got '%s' instead of one"
                         " of ('all', 'train', test')" % subset)

    if shuffle:
        X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'rcv1.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, sample_id=sample_id,
                 target_names=categories, DESCR=fdescr)


def _inverse_permutation(p):
    
```
5 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_boston(return_X_y=False):
    """Load and return the boston house-prices dataset (regression).

    ==============     ==============
    Samples total                 506
    Dimensionality                 13
    Features           real, positive
    Targets             real 5. - 50.
    ==============     ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the regression targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of boston
        csv dataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()
    >>> print(boston.data.shape)
    (506, 13)
    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'boston_house_prices.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'boston_house_prices.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)
```
**6 - /tmp/repos/scikit-learn/sklearn/datasets/covtype.py**:
```python
# ...


def fetch_covtype(data_home=None, download_if_missing=True,
                  random_state=None, shuffle=False, return_X_y=False):
    
```
7 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_breast_cancer(return_X_y=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the full description of
        the dataset, 'filename', the physical location of
        breast cancer csv dataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://goo.gl/U2Uwz2

    Examples
    --------
    Let's say you are interested in the samples 10, 50, and 85, and want to
    know their class name.

    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> data.target[[10, 50, 85]]
    array([0, 1, 0])
    >>> list(data.target_names)
    ['malignant', 'benign']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'breast_cancer.csv')
    csv_filename = join(module_path, 'data', 'breast_cancer.csv')

    with open(join(module_path, 'descr', 'breast_cancer.rst')) as rst_file:
        fdescr = rst_file.read()

    feature_names = np.array(['mean radius', 'mean texture',
                              'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness',
                              'mean concavity', 'mean concave points',
                              'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error',
                              'perimeter error', 'area error',
                              'smoothness error', 'compactness error',
                              'concavity error', 'concave points error',
                              'symmetry error', 'fractal dimension error',
                              'worst radius', 'worst texture',
                              'worst perimeter', 'worst area',
                              'worst smoothness', 'worst compactness',
                              'worst concavity', 'worst concave points',
                              'worst symmetry', 'worst fractal dimension'])

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=feature_names,
                 filename=csv_filename)
```
**8 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
if subset == 'SF' or subset == 'http' or subset == 'smtp':
        # select all samples with positive logged_in attribute:
        s = data[:, 11] == 1
        data = np.c_[data[s, :11], data[s, 12:]]
        target = target[s]

        data[:, 0] = np.log((data[:, 0] + 0.1).astype(float, copy=False))
        data[:, 4] = np.log((data[:, 4] + 0.1).astype(float, copy=False))
        data[:, 5] = np.log((data[:, 5] + 0.1).astype(float, copy=False))

        if subset == 'http':
            s = data[:, 2] == b'http'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]

        if subset == 'smtp':
            s = data[:, 2] == b'smtp'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]

        if subset == 'SF':
            data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]

    if shuffle:
        data, target = shuffle_method(data, target, random_state=random_state)

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'kddcup99.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, DESCR=fdescr)
# ...
```
9 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_diabetes(return_X_y=False):
    """Load and return the diabetes dataset (regression).

    ==============      ==================
    Samples total       442
    Dimensionality      10
    Features            real, -.2 < x < .2
    Targets             integer 25 - 346
    ==============      ==================

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the regression target for each
        sample, 'data_filename', the physical location
        of diabetes data csv dataset, and 'target_filename', the physical
        location of diabetes targets csv datataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18
    """
    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data_filename = join(base_dir, 'diabetes_data.csv.gz')
    data = np.loadtxt(data_filename)
    target_filename = join(base_dir, 'diabetes_target.csv.gz')
    target = np.loadtxt(target_filename)

    with open(join(module_path, 'descr', 'diabetes.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, DESCR=fdescr,
                 feature_names=['age', 'sex', 'bmi', 'bp',
                                's1', 's2', 's3', 's4', 's5', 's6'],
                 data_filename=data_filename,
                 target_filename=target_filename)
```
10 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_digits(n_class=10, return_X_y=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    n_class : integer, between 0 and 10, optional (default=10)
        The number of classes to return.

    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'images', the images corresponding
        to each sample, 'target', the classification labels for each
        sample, 'target_names', the meaning of the labels, and 'DESCR',
        the full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    Examples
    --------
    To load the data and visualize the images::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> print(digits.data.shape)
        (1797, 64)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.gray() #doctest: +SKIP
        >>> plt.matshow(digits.images[0]) #doctest: +SKIP
        >>> plt.show() #doctest: +SKIP
    """
    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'data', 'digits.csv.gz'),
                      delimiter=',')
    with open(join(module_path, 'descr', 'digits.rst')) as f:
        descr = f.read()
    target = data[:, -1].astype(np.int)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    if return_X_y:
        return flat_data, target

    return Bunch(data=flat_data,
                 target=target,
                 target_names=np.arange(10),
                 images=images,
                 DESCR=descr)
```
11 - /tmp/repos/scikit-learn/sklearn/datasets/base.py
```python
def load_wine(return_X_y=False):
    """Load and return the wine dataset (classification).

    .. versionadded:: 0.18

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: 'data', the
        data to learn, 'target', the classification labels, 'target_names', the
        meaning of the labels, 'feature_names', the meaning of the features,
        and 'DESCR', the full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

    Examples
    --------
    Let's say you are interested in the samples 10, 80, and 140, and want to
    know their class name.

    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> data.target[[10, 80, 140]]
    array([0, 1, 2])
    >>> list(data.target_names)
    ['class_0', 'class_1', 'class_2']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'wine_data.csv')

    with open(join(module_path, 'descr', 'wine_data.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['alcohol',
                                'malic_acid',
                                'ash',
                                'alcalinity_of_ash',
                                'magnesium',
                                'total_phenols',
                                'flavanoids',
                                'nonflavanoid_phenols',
                                'proanthocyanins',
                                'color_intensity',
                                'hue',
                                'od280/od315_of_diluted_wines',
                                'proline'])
```
**12 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
# ...


def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
                                  download_if_missing=True, return_X_y=False):
    
```
**13 - /tmp/repos/scikit-learn/sklearn/datasets/rcv1.py**:
```python
"""Load the RCV1 multilabel dataset, downloading it if necessary.

    Version: RCV1-v2, vectors, full sets, topics multilabels.

    ==============     =====================
    Classes                              103
    Samples total                     804414
    Dimensionality                     47236
    Features           real, between 0 and 1
    ==============     =====================

    Read more in the :ref:`User Guide <datasets>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : string, 'train', 'test', or 'all', default='all'
        Select the dataset to load: 'train' for the training set
        (23149 samples), 'test' for the test set (781265 samples),
        'all' for both, with the training samples first if shuffle is False.
        This follows the official LYRL2004 chronological split.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, optional (default=None)
        Random state for shuffling the dataset.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : scipy csr array, dtype np.float64, shape (804414, 47236)
        The array has 0.16% of non zero values.

    dataset.target : scipy csr array, dtype np.uint8, shape (804414, 103)
        Each sample has a value of 1 in its categories, and 0 in others.
        The array has 3.15% of non zero values.

    dataset.sample_id : numpy array, dtype np.uint32, shape (804414,)
        Identification number of each sample, as ordered in dataset.data.

    dataset.target_names : numpy array, dtype object, length (103)
        Names of each target (RCV1 topics), as ordered in dataset.target.

    dataset.DESCR : string
        Description of the RCV1 dataset.

    References
    ----------
    Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new
    benchmark collection for text categorization research. The Journal of
    Machine Learning Research, 5, 361-397.

    """
    N_SAMPLES = 804414
    N_FEATURES = 47236
    N_CATEGORIES = 103
    N_TRAIN = 23149

    data_home = get_data_home(data_home=data_home)
    rcv1_dir = join(data_home, "RCV1")
    if download_if_missing:
        if not exists(rcv1_dir):
            makedirs(rcv1_dir)

    samples_path = _pkl_filepath(rcv1_dir, "samples.pkl")
    sample_id_path = _pkl_filepath(rcv1_dir, "sample_id.pkl")
    sample_topics_path = _pkl_filepath(rcv1_dir, "sample_topics.pkl")
    topics_path = _pkl_filepath(rcv1_dir, "topics_names.pkl")

    # load data (X) and sample_id
    if download_if_missing and (not exists(samples_path) or
                                not exists(sample_id_path)):
        files = []
        for each in XY_METADATA:
            logger.info("Downloading %s" % each.url)
            file_path = _fetch_remote(each, dirname=rcv1_dir)
            files.append(GzipFile(filename=file_path))

        Xy = load_svmlight_files(files, n_features=N_FEATURES)

        # Training data is before testing data
        X = sp.vstack([Xy[8], Xy[0], Xy[2], Xy[4], Xy[6]]).tocsr()
        sample_id = np.hstack((Xy[9], Xy[1], Xy[3], Xy[5], Xy[7]))
        sample_id = sample_id.astype(np.uint32)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(sample_id, sample_id_path, compress=9)

        # delete archives
        for f in files:
            f.close()
            remove(f.name)
    else:
        X = joblib.load(samples_path)
        sample_id = joblib.load(sample_id_path)


    # load target (y), categories, and sample_id_bis
    if download_if_missing and (not exists(sample_topics_path) or
                                not exists(topics_path)):
        logger.info("Downloading %s" % TOPICS_METADATA.url)
        topics_archive_path = _fetch_remote(TOPICS_METADATA,
                                            dirname=rcv1_dir)

        # parse the target file
        n_cat = -1
        n_doc = -1
        doc_previous = -1
        y = np.zeros((N_SAMPLES, N_CATEGORIES), dtype=np.uint8)
        sample_id_bis = np.zeros(N_SAMPLES, dtype=np.int32)
        category_names = {}
        with GzipFile(filename=topics_archive_path, mode='rb') as f:
            for line in f:
                line_components = line.decode("ascii").split(u" ")
                if len(line_components) == 3:
                    cat, doc, _ = line_components
                    if cat not in category_names:
                        n_cat += 1
                        category_names[cat] = n_cat

                    doc = int(doc)
                    if doc != doc_previous:
                        doc_previous = doc
                        n_doc += 1
                        sample_id_bis[n_doc] = doc
                    y[n_doc, category_names[cat]] = 1

        # delete archive
        remove(topics_archive_path)

        # Samples in X are ordered with sample_id,
        # whereas in y, they are ordered with sample_id_bis.
        permutation = _find_permutation(sample_id_bis, sample_id)
        y = y[permutation, :]

        # save category names in a list, with same order than y
        categories = np.empty(N_CATEGORIES, dtype=object)
        for k in category_names.keys():
            categories[category_names[k]] = k

        # reorder categories in lexicographic order
        order = np.argsort(categories)
        categories = categories[order]
        y = sp.csr_matrix(y[:, order])

        joblib.dump(y, sample_topics_path, compress=9)
        joblib.dump(categories, topics_path, compress=9)
    else:
        y = joblib.load(sample_topics_path)
        categories = joblib.load(topics_path)

    if subset == 'all':
        pass
    elif subset == 'train':
        X = X[:N_TRAIN, :]
        y = y[:N_TRAIN, :]
        sample_id = sample_id[:N_TRAIN]
    elif subset == 'test':
        X = X[N_TRAIN:, :]
        y = y[N_TRAIN:, :]
        sample_id = sample_id[N_TRAIN:]
    else:
        raise ValueError("Unknown subset parameter. Got '%s' instead of one"
                         " of ('all', 'train', test')" % subset)

    if shuffle:
        X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)

    return Bunch(data=X, target=y, sample_id=sample_id,
                 target_names=categories, DESCR=__doc__)


def _inverse_permutation(p):
    
```
**14 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
# ...


def fetch_kddcup99(subset=None, data_home=None, shuffle=False,
                   random_state=None,
                   percent10=True, download_if_missing=True, return_X_y=False):
    """Load the kddcup99 dataset (classification).

    Download it if necessary.

    =================   ====================================
    Classes                                               23
    Samples total                                    4898431
    Dimensionality                                        41
    Features            discrete (int) or continuous (float)
    =================   ====================================

    Read more in the :ref:`User Guide <kddcup99_dataset>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    subset : None, 'SA', 'SF', 'http', 'smtp'
        To return the corresponding classical subsets of kddcup 99.
        If None, return the entire kddcup 99 dataset.

    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.
        .. versionadded:: 0.19

    shuffle : bool, default=False
        Whether to shuffle dataset.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and for
        selection of abnormal samples if `subset='SA'`. Pass an int for
        reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.20

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
         - 'data', the data to learn.
         - 'target', the regression target for each sample.
         - 'DESCR', a description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    kddcup99 = _fetch_brute_kddcup99(data_home=data_home,
                                     percent10=percent10,
                                     download_if_missing=download_if_missing)

    data = kddcup99.data
    target = kddcup99.target

    if subset == 'SA':
        s = target == b'normal.'
        t = np.logical_not(s)
        normal_samples = data[s, :]
        normal_targets = target[s]
        abnormal_samples = data[t, :]
        abnormal_targets = target[t]

        n_samples_abnormal = abnormal_samples.shape[0]
        # selected abnormal samples:
        random_state = check_random_state(random_state)
        r = random_state.randint(0, n_samples_abnormal, 3377)
        abnormal_samples = abnormal_samples[r]
        abnormal_targets = abnormal_targets[r]

        data = np.r_[normal_samples, abnormal_samples]
        target = np.r_[normal_targets, abnormal_targets]

    
```
15 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# We had some issues with CSR matrices with unsorted indices (e.g. #1501),
    # so sort them here, but first make sure we don't modify the user's X.
    # TODO We can do this cheaper; sorted_indices copies the whole matrix.
    if yval is y and hasattr(yval, "sorted_indices"):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, "sort_indices"):
            y.sort_indices()

    if Xval is X and hasattr(Xval, "sorted_indices"):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, "sort_indices"):
            X.sort_indices()

    if query_id is not None:
        query_id = np.asarray(query_id)
        if query_id.shape[0] != y.shape[0]:
            raise ValueError("expected query_id of shape (n_samples,), got %r"
                             % (query_id.shape,))

    one_based = not zero_based

    if hasattr(f, "write"):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, "wb") as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
```
16 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# We had some issues with CSR matrices with unsorted indices (e.g. #1501),
    # so sort them here, but first make sure we don't modify the user's X.
    # TODO We can do this cheaper; sorted_indices copies the whole matrix.
    if yval is y and hasattr(yval, "sorted_indices"):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, "sort_indices"):
            y.sort_indices()

    if Xval is X and hasattr(Xval, "sorted_indices"):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, "sort_indices"):
            X.sort_indices()

    if query_id is not None:
        query_id = np.asarray(query_id)
        if query_id.shape[0] != y.shape[0]:
            raise ValueError("expected query_id of shape (n_samples,), got %r"
                             % (query_id.shape,))

    one_based = not zero_based

    if hasattr(f, "write"):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, "wb") as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
```
17 - /tmp/repos/scikit-learn/sklearn/datasets/openml.py
```python
# ...


def fetch_openml(name=None, version='active', data_id=None, data_home=None,
                 target_column='default-target', cache=True, return_X_y=False):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes in future releases.

    Parameters
    ----------
    name : str or None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : integer or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int or None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : string, list or None, default 'default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations)

    cache : boolean, default=True
        Whether to cache downloaded datasets using joblib.

    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

    Returns
    -------

    data : Bunch
        Dictionary-like object, with attributes:

        data : np.array or scipy.sparse.csr_matrix of floats
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical.
        DESCR : str
            The full description of the dataset
        feature_names : list
            The names of the dataset columns
        categories : dict
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list.
        details : dict
            More metadata from OpenML

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target)
    
```
**18 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names = data_train.target_names

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, target_names=target_names)
```
**19 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names = data_train.target_names

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, target_names=target_names)
```
20 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/binning.py
```python
class _BinMapper(BaseEstimator, TransformerMixin):


    def transform(self, X):
        """Bin data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to bin.

        Returns
        -------
        X_binned : array-like, shape (n_samples, n_features)
            The binned data (fortran-aligned).
        """
        X = check_array(X, dtype=[X_DTYPE])
        check_is_fitted(self, ['bin_thresholds_', 'actual_n_bins_'])
        if X.shape[1] != self.actual_n_bins_.shape[0]:
            raise ValueError(
                'This estimator was fitted with {} features but {} got passed '
                'to transform()'.format(self.actual_n_bins_.shape[0],
                                        X.shape[1])
            )
        binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order='F')
        _map_to_bins(X, self.bin_thresholds_, binned)
        return binned
```
21 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
22 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class _CVIterableWrapper(BaseCrossValidator):


    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        for train, test in self.cv:
            yield train, test
```
23 - /tmp/repos/scikit-learn/sklearn/feature_extraction/dict_vectorizer.py
```python
class DictVectorizer(BaseEstimator, TransformerMixin):


    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X), but does not require
        materializing X in memory.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        return self._transform(X, fitting=True)
```
**24 - /tmp/repos/scikit-learn/sklearn/datasets/california_housing.py**:
```python
# ...


def fetch_california_housing(data_home=None, download_if_missing=True,
                             return_X_y=False):
    
```
**25 - /tmp/repos/scikit-learn/sklearn/datasets/california_housing.py**:
```python
# ...


def fetch_california_housing(data_home=None, download_if_missing=True,
                             return_X_y=False):
    
```
26 - /tmp/repos/scikit-learn/sklearn/isotonic.py
```python
def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            Training data.

        y : array-like, shape=(n_samples,)
            Training target.

        sample_weight : array-like, shape=(n_samples,), optional, default: None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as `transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False,
                            dtype=[np.float64, np.float32])
        X = check_array(X, **check_params)
        y = check_array(y, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self._necessary_X_, self._necessary_y_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def transform(self, T):
        """Transform new data by linear interpolation

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            The transformed data
        """

        if hasattr(self, '_necessary_X_'):
            dtype = self._necessary_X_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False)

        if len(T.shape) != 1:
            raise ValueError("Isotonic regression input should be a 1d array")

        # Handle the out_of_bounds argument by clipping if needed
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError("The argument ``out_of_bounds`` must be in "
                             "'nan', 'clip', 'raise'; got {0}"
                             .format(self.out_of_bounds))

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            Transformed data.
        """
        return self.transform(T)

    def __getstate__(self):
        """Pickle-protocol - return state of the estimator. """
        state = super().__getstate__()
        # remove interpolation method
        state.pop('f_', None)
        return state

    
```
27 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# ...


def load_svmlight_file(f, n_features=None, dtype=np.float64,
                       multilabel=False, zero_based="auto", query_id=False,
                       offset=0, length=-1):
    """Load datasets in the svmlight / libsvm format into sparse CSR matrix

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    This format is used as the default format for both svmlight and the
    libsvm command line programs.

    Parsing a text based source can be expensive. When working on
    repeatedly on the same dataset, it is recommended to wrap this
    loader with joblib.Memory.cache to store a memmapped backup of the
    CSR results of the first call and benefit from the near instantaneous
    loading of memmapped structures for the subsequent calls.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    This implementation is written in Cython and is reasonably fast.
    However, a faster API-compatible loader is also available at:

      https://github.com/mblondel/svmlight-loader

    Parameters
    ----------
    f : {str, file-like, int}
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.

    n_features : int or None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.

    dtype : numpy data type, default np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : boolean, optional, default False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

    zero_based : boolean or "auto", optional, default "auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : boolean, default False
        If True, will return the query_id array for each file.

    offset : integer, optional, default 0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : integer, optional, default -1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)

    y : ndarray of shape (n_samples,), or, in the multilabel a list of
        tuples of length n_samples.

    query_id : array of shape (n_samples,)
       query_id for each sample. Only returned when query_id is set to
       True.

    See also
    --------
    load_svmlight_files: similar function for loading multiple files in this
                         format, enforcing the same number of features/columns
                         on all of them.

    Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from joblib import Memory
        from .datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]

        X, y = get_data()
    
```
28 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# ...


def load_svmlight_file(f, n_features=None, dtype=np.float64,
                       multilabel=False, zero_based="auto", query_id=False,
                       offset=0, length=-1):
    """Load datasets in the svmlight / libsvm format into sparse CSR matrix

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    This format is used as the default format for both svmlight and the
    libsvm command line programs.

    Parsing a text based source can be expensive. When working on
    repeatedly on the same dataset, it is recommended to wrap this
    loader with joblib.Memory.cache to store a memmapped backup of the
    CSR results of the first call and benefit from the near instantaneous
    loading of memmapped structures for the subsequent calls.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    This implementation is written in Cython and is reasonably fast.
    However, a faster API-compatible loader is also available at:

      https://github.com/mblondel/svmlight-loader

    Parameters
    ----------
    f : {str, file-like, int}
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.

    n_features : int or None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.

    dtype : numpy data type, default np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : boolean, optional, default False
        Samples may have several labels each (see
        http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

    zero_based : boolean or "auto", optional, default "auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : boolean, default False
        If True, will return the query_id array for each file.

    offset : integer, optional, default 0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : integer, optional, default -1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)

    y : ndarray of shape (n_samples,), or, in the multilabel a list of
        tuples of length n_samples.

    query_id : array of shape (n_samples,)
       query_id for each sample. Only returned when query_id is set to
       True.

    See also
    --------
    load_svmlight_files: similar function for loading multiple files in this
    format, enforcing the same number of features/columns on all of them.

    Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from sklearn.externals.joblib import Memory
        from sklearn.datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]

        X, y = get_data()
    
```
29 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
                        'predict_log_proba']
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=np.int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
# ...
```
30 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# ...


def dump_svmlight_file(X, y, f,  zero_based=True, comment=None, query_id=None,
                       multilabel=False):
    
```
31 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
# ...


def dump_svmlight_file(X, y, f,  zero_based=True, comment=None, query_id=None,
                       multilabel=False):
    
```
**32 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
"""
    data_home = get_data_home(data_home=data_home)
    kddcup99 = _fetch_brute_kddcup99(data_home=data_home,
                                     percent10=percent10,
                                     download_if_missing=download_if_missing)

    data = kddcup99.data
    target = kddcup99.target

    if subset == 'SA':
        s = target == b'normal.'
        t = np.logical_not(s)
        normal_samples = data[s, :]
        normal_targets = target[s]
        abnormal_samples = data[t, :]
        abnormal_targets = target[t]

        n_samples_abnormal = abnormal_samples.shape[0]
        # selected abnormal samples:
        random_state = check_random_state(random_state)
        r = random_state.randint(0, n_samples_abnormal, 3377)
        abnormal_samples = abnormal_samples[r]
        abnormal_targets = abnormal_targets[r]

        data = np.r_[normal_samples, abnormal_samples]
        target = np.r_[normal_targets, abnormal_targets]

    if subset == 'SF' or subset == 'http' or subset == 'smtp':
        # select all samples with positive logged_in attribute:
        s = data[:, 11] == 1
        data = np.c_[data[s, :11], data[s, 12:]]
        target = target[s]

        data[:, 0] = np.log((data[:, 0] + 0.1).astype(float))
        data[:, 4] = np.log((data[:, 4] + 0.1).astype(float))
        data[:, 5] = np.log((data[:, 5] + 0.1).astype(float))

        if subset == 'http':
            s = data[:, 2] == b'http'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]

        if subset == 'smtp':
            s = data[:, 2] == b'smtp'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]

        if subset == 'SF':
            data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]

    if shuffle:
        data, target = shuffle_method(data, target, random_state=random_state)

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target)
# ...
```
33 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class BaseHistGradientBoosting(BaseEstimator, ABC):


    def _bin_data(self, X, rng, is_training_data):
        """Bin data X.

        If is_training_data, then set the bin_mapper_ attribute.
        Else, the binned data is converted to a C-contiguous array.
        """

        description = 'training' if is_training_data else 'validation'
        if self.verbose:
            print("Binning {:.3f} GB of {} data: ".format(
                X.nbytes / 1e9, description), end="", flush=True)
        tic = time()
        if is_training_data:
            X_binned = self.bin_mapper_.fit_transform(X)  # F-aligned array
        else:
            X_binned = self.bin_mapper_.transform(X)  # F-aligned array
            # We convert the array to C-contiguous since predicting is faster
            # with this layout (training is faster on F-arrays though)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time()
        if self.verbose:
            duration = toc - tic
            print("{:.3f} s".format(duration))

        return X_binned
```
**34 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
if download_if_missing and not available:
        _mkdirp(kddcup_dir)
        logger.info("Downloading %s" % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir)
        dt = [('duration', int),
              ('protocol_type', 'S4'),
              ('service', 'S11'),
              ('flag', 'S6'),
              ('src_bytes', int),
              ('dst_bytes', int),
              ('land', int),
              ('wrong_fragment', int),
              ('urgent', int),
              ('hot', int),
              ('num_failed_logins', int),
              ('logged_in', int),
              ('num_compromised', int),
              ('root_shell', int),
              ('su_attempted', int),
              ('num_root', int),
              ('num_file_creations', int),
              ('num_shells', int),
              ('num_access_files', int),
              ('num_outbound_cmds', int),
              ('is_host_login', int),
              ('is_guest_login', int),
              ('count', int),
              ('srv_count', int),
              ('serror_rate', float),
              ('srv_serror_rate', float),
              ('rerror_rate', float),
              ('srv_rerror_rate', float),
              ('same_srv_rate', float),
              ('diff_srv_rate', float),
              ('srv_diff_host_rate', float),
              ('dst_host_count', int),
              ('dst_host_srv_count', int),
              ('dst_host_same_srv_rate', float),
              ('dst_host_diff_srv_rate', float),
              ('dst_host_same_src_port_rate', float),
              ('dst_host_srv_diff_host_rate', float),
              ('dst_host_serror_rate', float),
              ('dst_host_srv_serror_rate', float),
              ('dst_host_rerror_rate', float),
              ('dst_host_srv_rerror_rate', float),
              ('labels', 'S16')]
        DT = np.dtype(dt)
        logger.debug("extracting archive")
        archive_path = join(kddcup_dir, archive.filename)
        file_ = GzipFile(filename=archive_path, mode='r')
        Xy = []
        for line in file_.readlines():
            if six.PY3:
                line = line.decode()
            Xy.append(line.replace('\n', '').split(','))
        file_.close()
        logger.debug('extraction done')
        os.remove(archive_path)

        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:, :-1]
        y = Xy[:, -1]
        # XXX bug when compress!=0:
        # (error: 'Incorrect data length while decompressing[...] the file
        #  could be corrupted.')

        joblib.dump(X, samples_path, compress=0)
        joblib.dump(y, targets_path, compress=0)
    elif not available:
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    return Bunch(data=X, target=y, DESCR=__doc__)
# ...
```
**35 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
if download_if_missing and not available:
        _mkdirp(kddcup_dir)
        logger.info("Downloading %s" % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir)
        dt = [('duration', int),
              ('protocol_type', 'S4'),
              ('service', 'S11'),
              ('flag', 'S6'),
              ('src_bytes', int),
              ('dst_bytes', int),
              ('land', int),
              ('wrong_fragment', int),
              ('urgent', int),
              ('hot', int),
              ('num_failed_logins', int),
              ('logged_in', int),
              ('num_compromised', int),
              ('root_shell', int),
              ('su_attempted', int),
              ('num_root', int),
              ('num_file_creations', int),
              ('num_shells', int),
              ('num_access_files', int),
              ('num_outbound_cmds', int),
              ('is_host_login', int),
              ('is_guest_login', int),
              ('count', int),
              ('srv_count', int),
              ('serror_rate', float),
              ('srv_serror_rate', float),
              ('rerror_rate', float),
              ('srv_rerror_rate', float),
              ('same_srv_rate', float),
              ('diff_srv_rate', float),
              ('srv_diff_host_rate', float),
              ('dst_host_count', int),
              ('dst_host_srv_count', int),
              ('dst_host_same_srv_rate', float),
              ('dst_host_diff_srv_rate', float),
              ('dst_host_same_src_port_rate', float),
              ('dst_host_srv_diff_host_rate', float),
              ('dst_host_serror_rate', float),
              ('dst_host_srv_serror_rate', float),
              ('dst_host_rerror_rate', float),
              ('dst_host_srv_rerror_rate', float),
              ('labels', 'S16')]
        DT = np.dtype(dt)
        logger.debug("extracting archive")
        archive_path = join(kddcup_dir, archive.filename)
        file_ = GzipFile(filename=archive_path, mode='r')
        Xy = []
        for line in file_.readlines():
            if six.PY3:
                line = line.decode()
            Xy.append(line.replace('\n', '').split(','))
        file_.close()
        logger.debug('extraction done')
        os.remove(archive_path)

        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:, :-1]
        y = Xy[:, -1]
        # XXX bug when compress!=0:
        # (error: 'Incorrect data length while decompressing[...] the file
        #  could be corrupted.')

        joblib.dump(X, samples_path, compress=0)
        joblib.dump(y, targets_path, compress=0)
    elif not available:
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    return Bunch(data=X, target=y, DESCR=__doc__)
# ...
```
36 - /tmp/repos/scikit-learn/sklearn/base.py
```python
class ClusterMixin(object):


    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X)
        return self.labels_
```
**37 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
if download_if_missing and not available:
        _mkdirp(kddcup_dir)
        logger.info("Downloading %s" % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir)
        dt = [('duration', int),
              ('protocol_type', 'S4'),
              ('service', 'S11'),
              ('flag', 'S6'),
              ('src_bytes', int),
              ('dst_bytes', int),
              ('land', int),
              ('wrong_fragment', int),
              ('urgent', int),
              ('hot', int),
              ('num_failed_logins', int),
              ('logged_in', int),
              ('num_compromised', int),
              ('root_shell', int),
              ('su_attempted', int),
              ('num_root', int),
              ('num_file_creations', int),
              ('num_shells', int),
              ('num_access_files', int),
              ('num_outbound_cmds', int),
              ('is_host_login', int),
              ('is_guest_login', int),
              ('count', int),
              ('srv_count', int),
              ('serror_rate', float),
              ('srv_serror_rate', float),
              ('rerror_rate', float),
              ('srv_rerror_rate', float),
              ('same_srv_rate', float),
              ('diff_srv_rate', float),
              ('srv_diff_host_rate', float),
              ('dst_host_count', int),
              ('dst_host_srv_count', int),
              ('dst_host_same_srv_rate', float),
              ('dst_host_diff_srv_rate', float),
              ('dst_host_same_src_port_rate', float),
              ('dst_host_srv_diff_host_rate', float),
              ('dst_host_serror_rate', float),
              ('dst_host_srv_serror_rate', float),
              ('dst_host_rerror_rate', float),
              ('dst_host_srv_rerror_rate', float),
              ('labels', 'S16')]
        DT = np.dtype(dt)
        logger.debug("extracting archive")
        archive_path = join(kddcup_dir, archive.filename)
        file_ = GzipFile(filename=archive_path, mode='r')
        Xy = []
        for line in file_.readlines():
            if six.PY3:
                line = line.decode()
            Xy.append(line.replace('\n', '').split(','))
        file_.close()
        logger.debug('extraction done')
        os.remove(archive_path)

        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:, :-1]
        y = Xy[:, -1]
        # XXX bug when compress!=0:
        # (error: 'Incorrect data length while decompressing[...] the file
        #  could be corrupted.')

        joblib.dump(X, samples_path, compress=0)
        joblib.dump(y, targets_path, compress=0)
    elif not available:
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    return Bunch(data=X, target=y, DESCR=__doc__)
# ...
```
**38 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
data_test = fetch_20newsgroups(data_home=data_home,
                                   subset='test',
                                   categories=None,
                                   shuffle=True,
                                   random_state=12,
                                   remove=remove,
                                   download_if_missing=download_if_missing)

    if os.path.exists(target_file):
        X_train, X_test = joblib.load(target_file)
    else:
        vectorizer = CountVectorizer(dtype=np.int16)
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        joblib.dump((X_train, X_test), target_file, compress=9)

    # the data is stored as int16 for compactness
    # but normalize needs floats
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names = data_train.target_names

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'twenty_newsgroups.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 target_names=target_names,
                 DESCR=fdescr)
```
**39 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
# ...


def fetch_20newsgroups_vectorized(subset="train", remove=(), data_home=None,
                                  download_if_missing=True, return_X_y=False):
    """Load the 20 newsgroups dataset and vectorize it into token counts \
    ssification).

    Download it if necessary.

    This is a convenience function; the transformation is done using the
    default settings for
    :class:`sklearn.feature_extraction.text.CountVectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom
    :class:`sklearn.feature_extraction.text.CountVectorizer`,
    :class:`sklearn.feature_extraction.text.HashingVectorizer`,
    :class:`sklearn.feature_extraction.text.TfidfTransformer` or
    :class:`sklearn.feature_extraction.text.TfidfVectorizer`.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality          130107
    Features                  real
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

    data_home : optional, default: None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    Returns
    -------
    bunch : Bunch object with the following attribute:
        - bunch.data: sparse matrix, shape [n_samples, n_features]
        - bunch.target: array, shape [n_samples]
        - bunch.target_names: a list of categories of the returned data,
          length [n_classes].
        - bunch.DESCR: a description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    filebase = '20newsgroup_vectorized'
    if remove:
        filebase += 'remove-' + ('-'.join(remove))
    target_file = _pkl_filepath(data_home, filebase + ".pkl")

    # we shuffle but use a fixed seed for the memoization
    data_train = fetch_20newsgroups(data_home=data_home,
                                    subset='train',
                                    categories=None,
                                    shuffle=True,
                                    random_state=12,
                                    remove=remove,
                                    download_if_missing=download_if_missing)

    
```
40 - /tmp/repos/scikit-learn/sklearn/datasets/__init__.py
```python
# ..."""
The :mod:`sklearn.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
from .base import load_breast_cancer
from .base import load_boston
from .base import load_diabetes
from .base import load_digits
from .base import load_files
from .base import load_iris
from .base import load_linnerud
from .base import load_sample_images
from .base import load_sample_image
from .base import load_wine
from .base import get_data_home
from .base import clear_data_home
from .covtype import fetch_covtype
from .kddcup99 import fetch_kddcup99
from .mlcomp import load_mlcomp
from .lfw import fetch_lfw_pairs
from .lfw import fetch_lfw_people
from .twenty_newsgroups import fetch_20newsgroups
from .twenty_newsgroups import fetch_20newsgroups_vectorized
from .mldata import fetch_mldata, mldata_filename
from .samples_generator import make_classification
from .samples_generator import make_multilabel_classification
from .samples_generator import make_hastie_10_2
from .samples_generator import make_regression
from .samples_generator import make_blobs
from .samples_generator import make_moons
from .samples_generator import make_circles
from .samples_generator import make_friedman1
from .samples_generator import make_friedman2
from .samples_generator import make_friedman3
from .samples_generator import make_low_rank_matrix
from .samples_generator import make_sparse_coded_signal
from .samples_generator import make_sparse_uncorrelated
from .samples_generator import make_spd_matrix
from .samples_generator import make_swiss_roll
from .samples_generator import make_s_curve
from .samples_generator import make_sparse_spd_matrix
from .samples_generator import make_gaussian_quantiles
from .samples_generator import make_biclusters
from .samples_generator import make_checkerboard
from .svmlight_format import load_svmlight_file
from .svmlight_format import load_svmlight_files
from .svmlight_format import dump_svmlight_file
from .olivetti_faces import fetch_olivetti_faces
from .species_distributions import fetch_species_distributions
from .california_housing import fetch_california_housing
from .rcv1 import fetch_rcv1



```
41 - /tmp/repos/scikit-learn/sklearn/manifold/isomap.py
```python
class Isomap(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, precomputed tree, or NearestNeighbors
            object.

        y: Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self
```
42 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/predictor.py
```python
class TreePredictor:


    def predict_binned(self, X):
        """Predict raw values for binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_binned_data(self.nodes, X, out)
        return out
```
43 - /tmp/repos/scikit-learn/sklearn/manifold/isomap.py
```python
class Isomap(BaseEstimator, TransformerMixin):


    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y: Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_
```
**44 - /tmp/repos/scikit-learn/sklearn/datasets/kddcup99.py**:
```python
if download_if_missing and not available:
        _mkdirp(kddcup_dir)
        logger.info("Downloading %s" % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir)
        dt = [('duration', int),
              ('protocol_type', 'S4'),
              ('service', 'S11'),
              ('flag', 'S6'),
              ('src_bytes', int),
              ('dst_bytes', int),
              ('land', int),
              ('wrong_fragment', int),
              ('urgent', int),
              ('hot', int),
              ('num_failed_logins', int),
              ('logged_in', int),
              ('num_compromised', int),
              ('root_shell', int),
              ('su_attempted', int),
              ('num_root', int),
              ('num_file_creations', int),
              ('num_shells', int),
              ('num_access_files', int),
              ('num_outbound_cmds', int),
              ('is_host_login', int),
              ('is_guest_login', int),
              ('count', int),
              ('srv_count', int),
              ('serror_rate', float),
              ('srv_serror_rate', float),
              ('rerror_rate', float),
              ('srv_rerror_rate', float),
              ('same_srv_rate', float),
              ('diff_srv_rate', float),
              ('srv_diff_host_rate', float),
              ('dst_host_count', int),
              ('dst_host_srv_count', int),
              ('dst_host_same_srv_rate', float),
              ('dst_host_diff_srv_rate', float),
              ('dst_host_same_src_port_rate', float),
              ('dst_host_srv_diff_host_rate', float),
              ('dst_host_serror_rate', float),
              ('dst_host_srv_serror_rate', float),
              ('dst_host_rerror_rate', float),
              ('dst_host_srv_rerror_rate', float),
              ('labels', 'S16')]
        DT = np.dtype(dt)
        logger.debug("extracting archive")
        archive_path = join(kddcup_dir, archive.filename)
        file_ = GzipFile(filename=archive_path, mode='r')
        Xy = []
        for line in file_.readlines():
            line = line.decode()
            Xy.append(line.replace('\n', '').split(','))
        file_.close()
        logger.debug('extraction done')
        os.remove(archive_path)

        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:, :-1]
        y = Xy[:, -1]
        # XXX bug when compress!=0:
        # (error: 'Incorrect data length while decompressing[...] the file
        #  could be corrupted.')

        joblib.dump(X, samples_path, compress=0)
        joblib.dump(y, targets_path, compress=0)
    elif not available:
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    return Bunch(data=X, target=y)
# ...
```
45 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""
    if (offset != 0 or length > 0) and zero_based == "auto":
        # disable heuristic search to avoid getting inconsistent results on
        # different segments of the file
        zero_based = True

    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError(
            "n_features is required when offset or length is specified.")

    r = [_open_and_load(f, dtype, multilabel, bool(zero_based), bool(query_id),
                        offset=offset, length=length)
         for f in files]

    if (zero_based is False or
            zero_based == "auto" and all(len(tmp[1]) and np.min(tmp[1]) > 0
                                         for tmp in r)):
        for _, indices, _, _, _ in r:
            indices -= 1

    n_f = max(ind[1].max() if len(ind[1]) else 0 for ind in r) + 1

    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError("n_features was set to {},"
                         " but input file contains {} features"
                         .format(n_features, n_f))

    result = []
    for data, indices, indptr, y, query_values in r:
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()
        result += X, y
        if query_id:
            result.append(query_values)

    return result
# ...
```
46 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""
    if (offset != 0 or length > 0) and zero_based == "auto":
        # disable heuristic search to avoid getting inconsistent results on
        # different segments of the file
        zero_based = True

    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError(
            "n_features is required when offset or length is specified.")

    r = [_open_and_load(f, dtype, multilabel, bool(zero_based), bool(query_id),
                        offset=offset, length=length)
         for f in files]

    if (zero_based is False or
            zero_based == "auto" and all(len(tmp[1]) and np.min(tmp[1]) > 0
                                         for tmp in r)):
        for _, indices, _, _, _ in r:
            indices -= 1

    n_f = max(ind[1].max() if len(ind[1]) else 0 for ind in r) + 1

    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError("n_features was set to {},"
                         " but input file contains {} features"
                         .format(n_features, n_f))

    result = []
    for data, indices, indptr, y, query_values in r:
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()
        result += X, y
        if query_id:
            result.append(query_values)

    return result
# ...
```
47 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None, batch_size=None):
    """Euclidean distances between X and Y

    Assumes X and Y have float32 dtype.
    Assumes XX and YY have float64 dtype or are None.

    X and Y are upcast to float64 by chunks, which size is chosen to limit
    memory increase by approximately 10% (at least 10MiB).
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)

    if batch_size is None:
        x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
        y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1

        # Allow 10% more memory than X, Y and the distance matrix take (at
        # least 10MiB)
        maxmem = max(
            ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
             + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
            10 * 2 ** 17)

        # The increase amount of memory in 8-byte blocks is:
        # - x_density * batch_size * n_features (copy of chunk of X)
        # - y_density * batch_size * n_features (copy of chunk of Y)
        # - batch_size * batch_size (chunk of distance matrix)
        # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
        #                                 xd=x_density and yd=y_density
        tmp = (x_density + y_density) * n_features
        batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
        batch_size = max(int(batch_size), 1)

    x_batches = gen_batches(n_samples_X, batch_size)

    for i, x_slice in enumerate(x_batches):
        X_chunk = X[x_slice].astype(np.float64)
        if XX is None:
            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]

        y_batches = gen_batches(n_samples_Y, batch_size)

        for j, y_slice in enumerate(y_batches):
            if X is Y and j < i:
                # when X is Y the distance matrix is symmetric so we only need
                # to compute half of it.
                d = distances[y_slice, x_slice].T

            else:
                Y_chunk = Y[y_slice].astype(np.float64)
                if YY is None:
                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
                else:
                    YY_chunk = YY[:, y_slice]

                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
                d += XX_chunk
                d += YY_chunk

            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)

    return distances


def _argmin_min_reduce(dist, start):
    indices = dist.argmin(axis=1)
    values = dist[np.arange(dist.shape[0]), indices]
    return indices, values



```
**48 - /tmp/repos/scikit-learn/sklearn/datasets/twenty_newsgroups.py**:
```python
if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)

    return Bunch(data=data, target=target, target_names=target_names)
```
49 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : {array-like, sparse matrix}, shape = [n_samples (, n_labels)]
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : string or file-like in binary mode
        If string, specifies the path that will contain the data.
        If file-like, data will be written to f. f should be opened in binary
        mode.

    zero_based : boolean, optional
        Whether column indices should be written zero-based (True) or one-based
        (False).

    comment : string, optional
        Comment to insert at the top of the file. This should be either a
        Unicode string, which will be encoded as UTF-8, or an ASCII byte
        string.
        If a comment is given, then it will be preceded by one that identifies
        the file as having been dumped by scikit-learn. Note that not all
        tools grok comments in SVMlight files.

    query_id : array-like, shape = [n_samples]
        Array containing pairwise preference constraints (qid in svmlight
        format).

    multilabel : boolean, optional
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

        .. versionadded:: 0.17
           parameter *multilabel* to support multilabel datasets.
    """
    if comment is not None:
        # Convert comment string to list of lines in UTF-8.
        # If a byte string is passed, then check whether it's ASCII;
        # if a user wants to get fancy, they'll have to decode themselves.
        # Avoid mention of str and unicode types for Python 3.x compat.
        if isinstance(comment, bytes):
            comment.decode("ascii")  # just for the exception
        else:
            comment = comment.encode("utf-8")
        if b"\0" in comment:
            raise ValueError("comment string contains NUL byte")

    yval = check_array(y, accept_sparse='csr', ensure_2d=False)
    if sp.issparse(yval):
        if yval.shape[1] != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples, 1),"
                             " got %r" % (yval.shape,))
    else:
        if yval.ndim != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples,), got %r"
                             % (yval.shape,))

    Xval = check_array(X, accept_sparse='csr')
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError("X.shape[0] and y.shape[0] should be the same, got"
                         " %r and %r instead." % (Xval.shape[0], yval.shape[0]))

    
```
50 - /tmp/repos/scikit-learn/sklearn/cluster/affinity_propagation_.py
```python
def fit_predict(self, X, y=None):
        """Fit the clustering from features or affinity matrix, and return
        cluster labels.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            array-like, shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
```
51 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
52 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
53 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
54 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
55 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
56 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
# ...
```
57 - /tmp/repos/scikit-learn/sklearn/neighbors/base.py
```python
class SupervisedFloatMixin(object):
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values, array of float values, shape = [n_samples]
             or [n_samples, n_outputs]
        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)
        self._y = y
        return self._fit(X)
```
58 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)
    out = parallel(delayed(_fit_and_score)(
        clone(estimator), X, y, scorer, train, test, verbose,
        parameters={param_name: v}, fit_params=None, return_train_score=True,
        error_score=error_score)
        # NOTE do not change order of iteration to allow one time cv splitters
        for train, test in cv.split(X, y, groups) for v in param_range)
    out = np.asarray(out)
    n_params = len(param_range)
    n_cv_folds = out.shape[0] // n_params
    out = out.reshape(n_cv_folds, n_params, 2).transpose((2, 1, 0))

    return out[0], out[1]
# ...
```
59 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
**60 - /tmp/repos/scikit-learn/sklearn/datasets/covtype.py**:
```python
"""Load the covertype dataset (classification).

    Download it if necessary.

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

    Read more in the :ref:`User Guide <covtype_dataset>`.

    Parameters
    ----------
    data_home : string, optional
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : boolean, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (581012, 54)
        Each row corresponds to the 54 features in the dataset.

    dataset.target : numpy array of shape (581012,)
        Each value corresponds to one of the 7 forest covertypes with values
        ranging between 1 to 7.

    dataset.DESCR : string
        Description of the forest covertype dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """

    data_home = get_data_home(data_home=data_home)
    covtype_dir = join(data_home, "covertype")
    samples_path = _pkl_filepath(covtype_dir, "samples")
    targets_path = _pkl_filepath(covtype_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        if not exists(covtype_dir):
            makedirs(covtype_dir)
        logger.info("Downloading %s" % ARCHIVE.url)

        archive_path = _fetch_remote(ARCHIVE, dirname=covtype_dir)
        Xy = np.genfromtxt(GzipFile(filename=archive_path), delimiter=',')
        # delete archive
        remove(archive_path)

        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32, copy=False)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")
    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    module_path = dirname(__file__)
    with open(join(module_path, 'descr', 'covtype.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, DESCR=fdescr)
```
61 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
"""Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : {array-like, sparse matrix}, shape = [n_samples (, n_labels)]
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : string or file-like in binary mode
        If string, specifies the path that will contain the data.
        If file-like, data will be written to f. f should be opened in binary
        mode.

    zero_based : boolean, optional
        Whether column indices should be written zero-based (True) or one-based
        (False).

    comment : string, optional
        Comment to insert at the top of the file. This should be either a
        Unicode string, which will be encoded as UTF-8, or an ASCII byte
        string.
        If a comment is given, then it will be preceded by one that identifies
        the file as having been dumped by scikit-learn. Note that not all
        tools grok comments in SVMlight files.

    query_id : array-like, shape = [n_samples]
        Array containing pairwise preference constraints (qid in svmlight
        format).

    multilabel : boolean, optional
        Samples may have several labels each (see
        http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

        .. versionadded:: 0.17
           parameter *multilabel* to support multilabel datasets.
    """
    if comment is not None:
        # Convert comment string to list of lines in UTF-8.
        # If a byte string is passed, then check whether it's ASCII;
        # if a user wants to get fancy, they'll have to decode themselves.
        # Avoid mention of str and unicode types for Python 3.x compat.
        if isinstance(comment, bytes):
            comment.decode("ascii")     # just for the exception
        else:
            comment = comment.encode("utf-8")
        if six.b("\0") in comment:
            raise ValueError("comment string contains NUL byte")

    yval = check_array(y, accept_sparse='csr', ensure_2d=False)
    if sp.issparse(yval):
        if yval.shape[1] != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples, 1),"
                             " got %r" % (yval.shape,))
    else:
        if yval.ndim != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples,), got %r"
                             % (yval.shape,))

    Xval = check_array(X, accept_sparse='csr')
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError("X.shape[0] and y.shape[0] should be the same, got"
                         " %r and %r instead." % (Xval.shape[0], yval.shape[0]))

    
```
62 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py
```python
def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples = X.shape[0]
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in xrange(n_samples):
        row = np.zeros(X.shape[1])
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row
```
63 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/predictor.py
```python
class TreePredictor:


    def predict(self, X):
        """Predict raw values for non-binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_numeric_data(self.nodes, X, out)
        return out
```
64 - /tmp/repos/scikit-learn/sklearn/model_selection/_split.py
```python
class BaseShuffleSplit(with_metaclass(ABCMeta)):


    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""
```
65 - /tmp/repos/scikit-learn/sklearn/metrics/pairwise.py
```python
# ...# -*- coding: utf-8 -*-

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Philippe Gervais <philippe.gervais@inria.fr>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
# License: BSD 3 clause

import itertools
from functools import partial
import warnings

import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..externals.joblib import Parallel
from ..externals.joblib import delayed
from ..externals.joblib import cpu_count

from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan


# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype



```
66 - /tmp/repos/scikit-learn/sklearn/datasets/svmlight_format.py
```python
def _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id):
    X_is_sp = int(hasattr(X, "tocsr"))
    y_is_sp = int(hasattr(y, "tocsr"))
    if X.dtype.kind == 'i':
        value_pattern = u("%d:%d")
    else:
        value_pattern = u("%d:%.16g")

    if y.dtype.kind == 'i':
        label_pattern = u("%d")
    else:
        label_pattern = u("%.16g")

    line_pattern = u("%s")
    if query_id is not None:
        line_pattern += u(" qid:%d")
    line_pattern += u(" %s\n")

    if comment:
        f.write(b("# Generated by dump_svmlight_file from scikit-learn %s\n"
                % __version__))
        f.write(b("# Column indices are %s-based\n"
                  % ["zero", "one"][one_based]))

        f.write(b("#\n"))
        f.writelines(b("# %s\n" % line) for line in comment.splitlines())

    for i in range(X.shape[0]):
        if X_is_sp:
            span = slice(X.indptr[i], X.indptr[i + 1])
            row = zip(X.indices[span], X.data[span])
        else:
            nz = X[i] != 0
            row = zip(np.where(nz)[0], X[i, nz])

        s = " ".join(value_pattern % (j + one_based, x) for j, x in row)

        if multilabel:
            if y_is_sp:
                nz_labels = y[i].nonzero()[1]
            else:
                nz_labels = np.where(y[i] != 0)[0]
            labels_str = ",".join(label_pattern % j for j in nz_labels)
        else:
            if y_is_sp:
                labels_str = label_pattern % y.data[i]
            else:
                labels_str = label_pattern % y[i]

        if query_id is not None:
            feat = (labels_str, query_id[i], s)
        else:
            feat = (labels_str, s)

        f.write((line_pattern % feat).encode('ascii'))
```
67 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True)
        for train, test in cv.split(X, y, groups))

    if return_train_score:
        train_scores, test_scores, fit_times, score_times = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret
# ...
```
68 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True)
        for train, test in cv.split(X, y, groups))

    if return_train_score:
        train_scores, test_scores, fit_times, score_times = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret
# ...
```
69 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
70 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
71 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
72 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
"""
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
# ...
```
73 - /tmp/repos/scikit-learn/sklearn/neighbors/lof.py
```python
def fit(self, X, y=None):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        if self.contamination != 'auto':
            if not(0. < self.contamination <= .5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % self.contamination)

        super().fit(X)

        n_samples = self._fit_X.shape[0]
        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = (
            self.kneighbors(None, n_neighbors=self.n_neighbors_))

        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_)

        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = (self._lrd[_neighbors_indices_fit_X_] /
                            self._lrd[:, np.newaxis])

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        if self.contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_outlier_factor_,
                                         100. * self.contamination)

        return self

    @property
    def predict(self):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        This method allows to generalize prediction to *new observations* (not
        in the training set). Only available for novelty detection (when
        novelty is set to True).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        if not self.novelty:
            msg = ('predict is not available when novelty=False, use '
                   'fit_predict if you want to predict on training data. Use '
                   'novelty=True if you want to use LOF for novelty detection '
                   'and predict on new unseen data.')
            raise AttributeError(msg)

        return self._predict

    
```
74 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret
# ...
```
75 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret
# ...
```
76 - /tmp/repos/scikit-learn/sklearn/model_selection/_validation.py
```python
"""
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator)
        for train, test in cv.split(X, y, groups))

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret
# ...
```
77 - /tmp/repos/scikit-learn/sklearn/isotonic.py
```python
class IsotonicRegression(BaseEstimator, TransformerMixin, RegressorMixin):


    def _check_fit_data(self, X, y, sample_weight=None):
        if len(X.shape) != 1:
            raise ValueError("X should be a 1d array")
```
78 - /tmp/repos/scikit-learn/sklearn/feature_selection/univariate_selection.py
```python
class SelectKBest(_BaseFilter):


    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features = %d; got %r. "
                             "Use k='all' to return all features."
                             % (X.shape[1], self.k))
```
79 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
# ...


def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=None, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean or None, optional (default=None)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

        .. deprecated:: 0.21
            ``warn_on_dtype`` is deprecated in version 0.21 and will be
             removed in 0.23.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    
```
80 - /tmp/repos/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
```python
class HistGradientBoostingRegressor(BaseHistGradientBoosting, RegressorMixin):


    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """
        # Return raw predictions after converting shape
        # (n_samples, 1) to (n_samples,)
        return self._raw_predict(X).ravel()
```
81 - /tmp/repos/scikit-learn/sklearn/feature_selection/univariate_selection.py
```python
class _BaseFilter(BaseEstimator, SelectorMixin):


    def _check_params(self, X, y):
        pass
```
82 - /tmp/repos/scikit-learn/sklearn/cluster/birch.py
```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):


    def transform(self, X):
        """
        Transform X into subcluster centroids dimension.

        Each dimension represents the distance from the sample point to each
        cluster centroid.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_trans : {array-like, sparse matrix}, shape (n_samples, n_clusters)
            Transformed data.
        """
        check_is_fitted(self, 'subcluster_centers_')
        return euclidean_distances(X, self.subcluster_centers_)
```
83 - /tmp/repos/scikit-learn/sklearn/utils/validation.py
```python
# ...


def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2d and sparse y.  If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    
```
**84 - /tmp/repos/scikit-learn/sklearn/datasets/lfw.py**:
```python
# ...


def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
                     min_faces_per_person=0, color=False,
                     slice_=(slice(70, 195), slice(78, 172)),
                     download_if_missing=True, return_X_y=False):
    
```
